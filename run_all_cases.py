#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run a specified solver on all (or selected) cases in the configured data directory.

Usage:
    python run_all_cases.py                                   # quick set, 5 cores
    python run_all_cases.py A题_problem2_solver.py ./results/p2 all
    python run_all_cases.py A题_problem2_solver.py ./results/p2 quick --cores 2 3 4 5
    python run_all_cases.py --cases case_001 case_010 --time-limit 60
    python run_all_cases.py --limit 10 --max-ops 5000 --workers 4
    python run_all_cases.py A题_problem2_solver.py ./results/p2 all --resume

Positional arguments (same order as the original runner):
    solver       solver script (default: A题_problem2_solver.py)
    output_dir   output root (default: ./results/<solver name>)
    mode         quick: cases whose historical V8 runtime <= 1000s (default)
                 all:   every case

Case selection (applied after mode): --cases, --max-ops, --limit.
Core counts: --cores (default 5). For each case the core counts run in
ascending order and the previous plan warm-starts the next one (--no-warm to
disable), so the speedup curve is non-decreasing in the core count.

Single-core baselines are cached in <output_dir>/singlecore_cache.json and
passed to the solver with --single; large cases need minutes to compute it.

Optional environment variables:
    HUAWEI_DATA_DIR
    HUAWEI_CODE_DIR
    V8_MAX_WORKERS
"""

import os
import sys
import glob
import json
import csv
import time
import threading
import subprocess
import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed


ROOT = os.path.dirname(os.path.abspath(__file__))

ATTACHMENT = os.path.join(
    ROOT,
    "通用神经网络处理器下的多核调度问题  附件"
)

DATA = os.environ.get(
    "HUAWEI_DATA_DIR",
    os.path.join(ATTACHMENT, "data")
)

CODE = os.environ.get(
    "HUAWEI_CODE_DIR",
    os.path.join(ATTACHMENT, "code")
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a solver on all Huawei multicore scheduling cases."
    )

    parser.add_argument(
        "solver",
        nargs="?",
        default="A题_problem2_solver.py",
        help=(
            "Solver Python file. "
            "Can be a filename relative to this runner, or an absolute path. "
            "Default: A题_problem2_solver.py"
        ),
    )

    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help=(
            "Output root directory. "
            "Default: ./results/<solver name>."
        ),
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["quick", "all"],
        default="quick",
        help=(
            "quick: run cases whose historical V8 runtime <= 1000s "
            "(default); all: run every case."
        ),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        help="Explicit case names, e.g. case_001 case_010 (overrides mode).",
    )
    parser.add_argument(
        "--max-ops",
        type=int,
        help="Only cases with at most this many ops.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only the first N selected cases in filename order.",
    )
    parser.add_argument(
        "--cores",
        nargs="+",
        type=int,
        default=[5],
        help="Core counts to solve (default: 5). Example: --cores 2 3 4 5",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get(
            "V8_MAX_WORKERS",
            str(max(1, min(20, (os.cpu_count() or 2) - 1))),
        )),
        help="Number of cases solved concurrently.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=20000.0,
        help="Per-(case, core count) search budget in seconds (default 300).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed rows of an earlier run and solve only missing ones.",
    )
    parser.add_argument(
        "--no-warm",
        action="store_true",
        help="Do not pass the previous core count's plan as a seed.",
    )
    parser.add_argument(
        "--seed-dir",
        help=(
            "Optional directory with <case>/<k>cores/plan.json seed plans "
            "(e.g. results/problem1_v4_2); its singlecore_result.json files "
            "also fill the single-core cache."
        ),
    )
    return parser.parse_args()


ARGS = parse_args()


# ---------------------------------------------------------
# Solver path
# ---------------------------------------------------------

if os.path.isabs(ARGS.solver):
    SOLVER = ARGS.solver
else:
    SOLVER = os.path.join(ROOT, ARGS.solver)

SOLVER = os.path.abspath(SOLVER)

if not os.path.isfile(SOLVER):
    raise FileNotFoundError(
        f"Solver not found: {SOLVER}"
    )


# Solver filename without ".py"
SOLVER_NAME = os.path.splitext(
    os.path.basename(SOLVER)
)[0]


# ---------------------------------------------------------
# Output paths
# ---------------------------------------------------------

OUTPUT_ROOT = os.path.abspath(
    ARGS.output_dir or os.path.join(ROOT, "results", SOLVER_NAME)
)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

OUT_DIR = os.path.join(
    OUTPUT_ROOT,
    f"{SOLVER_NAME}_allcase_plans"
)

CSV_PATH = os.path.join(
    OUTPUT_ROOT,
    f"{SOLVER_NAME}_allcase_results.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_ROOT,
    f"{SOLVER_NAME}_allcase_summary.json"
)

SINGLE_CACHE_PATH = os.path.join(
    OUTPUT_ROOT,
    "singlecore_cache.json"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Cases
# ---------------------------------------------------------

ALL_CASES = sorted(
    os.path.basename(fp)[:-5]
    for fp in glob.glob(
        os.path.join(DATA, "case_*.json")
    )
)


# V8 historical runtime (seconds).
# 用来：
# 1. quick 模式过滤 >1000s 的超慢 case（与图规模强相关）
# 2. 将较慢 case 优先提交给线程池（LPT）
V8_HISTORY_SECONDS = {
    "case_001": 2.72,
    "case_002": 9.81,
    "case_003": 261.35,
    "case_004": 3.95,
    "case_005": 42.46,
    "case_006": 3.61,
    "case_007": 33.63,
    "case_008": 4.45,
    "case_009": 76.13,
    "case_010": 5.16,
    "case_011": 6.44,
    "case_012": 6.79,
    "case_013": 30.04,
    "case_014": 8133.06,
    "case_015": 6.78,
    "case_016": 1089.64,
    "case_017": 15.60,
    "case_018": 42.21,
    "case_019": 2.77,
    "case_020": 49.76,
    "case_021": 92.87,
    "case_022": 8.32,
    "case_023": 5.53,
    "case_024": 95.00,
    "case_025": 850.99,
    "case_026": 5.20,
    "case_027": 20.05,
    "case_028": 818.92,
    "case_029": 7.27,
    "case_030": 369.12,
    "case_031": 148.12,
    "case_032": 4.67,
    "case_033": 66.05,
    "case_034": 60.98,
    "case_035": 33.88,
    "case_036": 20.50,
    "case_037": 5.93,
    "case_038": 194.04,
    "case_039": 174.39,
    "case_040": 26.15,
    "case_041": 1696.98,
    "case_042": 108.68,
    "case_043": 140.73,
    "case_044": 11.93,
    "case_045": 5.87,
    "case_046": 8.87,
    "case_047": 348.38,
    "case_048": 12.78,
    "case_049": 40.63,
    "case_050": 51.88,
    "case_051": 14.22,
    "case_052": 10.94,
    "case_053": 260.32,
    "case_054": 381.94,
    "case_055": 7.95,
    "case_056": 93.93,
    "case_057": 6.94,
    "case_058": 964.82,
    "case_059": 42.44,
    "case_060": 108.26,
    "case_061": 7.34,
    "case_062": 820.58,
    "case_063": 85.40,
    "case_064": 9.46,
    "case_065": 3.85,
    "case_066": 70.98,
    "case_067": 113.83,
    "case_068": 65.68,
    "case_069": 10.53,
    "case_070": 7.67,
    "case_071": 7.59,
    "case_072": 4085.24,
    "case_073": 49.72,
    "case_074": 64.05,
    "case_075": 122.42,
    "case_076": 7759.09,
    "case_077": 70.49,
    "case_078": 8.65,
    "case_079": 1555.58,
    "case_080": 10.49,
    "case_081": 34.38,
    "case_082": 45.34,
    "case_083": 28.63,
    "case_084": 208.27,
    "case_085": 872.17,
    "case_086": 58.13,
    "case_087": 2458.66,
    "case_088": 37.57,
    "case_089": 33.60,
    "case_090": 10.30,
    "case_091": 5872.61,
    "case_092": 669.75,
    "case_093": 3.40,
    "case_094": 3.38,
    "case_095": 38.74,
    "case_096": 6.20,
    "case_097": 197.56,
    "case_098": 8.13,
    "case_099": 172.06,
    "case_100": 8.24,
}


QUICK_MAX_SECONDS = 1000.0


def op_count(case):
    with open(os.path.join(DATA, case + ".json"), encoding="utf-8") as f:
        return len(json.load(f)["ops"])


if ARGS.cases:
    unknown = [c for c in ARGS.cases if c not in ALL_CASES]
    if unknown:
        raise SystemExit(f"Unknown cases: {unknown}")
    CASES = sorted(set(ARGS.cases))
elif ARGS.mode == "all":
    CASES = list(ALL_CASES)
else:
    # 默认 quick：历史耗时 <= 1000 秒的全部运行。
    CASES = [
        c
        for c in ALL_CASES
        if V8_HISTORY_SECONDS.get(c, 0.0)
        <= QUICK_MAX_SECONDS
    ]

if ARGS.max_ops is not None:
    CASES = [c for c in CASES if op_count(c) <= ARGS.max_ops]

if ARGS.limit is not None:
    CASES = CASES[:ARGS.limit]

CORES = sorted(set(k for k in ARGS.cores if k >= 1))
if not CORES:
    raise SystemExit("--cores must contain at least one positive core count")

# Longest Processing Time first (LPT)：较慢任务先占 worker，减少尾部等待。
CASES.sort(
    key=lambda c: V8_HISTORY_SECONDS.get(c, 0.0),
    reverse=True,
)

MAX_WORKERS = max(1, ARGS.workers)

COLS = [
    "case",
    "cores",
    "status",
    "seconds",
    "single",
    "makespan",
    "speedup",
    "added_copy_bytes",
    "used_cores",
    "official_evaluations",
    "selected",
    "plan",
    "error",
]


# ---------------------------------------------------------
# Resume / caches
# ---------------------------------------------------------

LOCK = threading.Lock()
ROWS = {}

if ARGS.resume and os.path.isfile(CSV_PATH):
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok" or not os.path.isfile(row.get("plan", "")):
                continue
            for key in ("cores", "single", "makespan", "added_copy_bytes",
                        "used_cores", "official_evaluations"):
                if row.get(key) not in (None, ""):
                    row[key] = int(float(row[key]))
            for key in ("seconds", "speedup"):
                if row.get(key) not in (None, ""):
                    row[key] = float(row[key])
            ROWS[(row["case"], row["cores"])] = row

SINGLE_CACHE = {}
if os.path.isfile(SINGLE_CACHE_PATH):
    with open(SINGLE_CACHE_PATH, encoding="utf-8") as f:
        SINGLE_CACHE.update(json.load(f))
if ARGS.seed_dir:
    for case in CASES:
        path = os.path.join(ARGS.seed_dir, case, "singlecore_result.json")
        if case not in SINGLE_CACHE and os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                SINGLE_CACHE[case] = json.load(f)["makespan"]


def write():
    rows = sorted(ROWS.values(), key=lambda r: (r["case"], int(r["cores"])))
    tmp = CSV_PATH + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, CSV_PATH)
    tmp = SINGLE_CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(SINGLE_CACHE, f, indent=1, sort_keys=True)
    os.replace(tmp, SINGLE_CACHE_PATH)


def plan_path(case, cores):
    return os.path.join(OUT_DIR, f"{case}_{cores}core.json")


# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------

def run_one(case, cores, warm_plan):
    graph = os.path.join(DATA, case + ".json")
    plan = plan_path(case, cores)

    env = os.environ.copy()
    env["HUAWEI_CODE_DIR"] = CODE

    cmd = [
        sys.executable,
        SOLVER,
        graph,
        str(cores),
        plan,
        "--time-limit",
        str(ARGS.time_limit),
    ]
    with LOCK:
        single = SINGLE_CACHE.get(case)
    if single is not None:
        cmd += ["--single", str(single)]
    if warm_plan and os.path.isfile(warm_plan):
        cmd += ["--seed-plan", warm_plan]
    if ARGS.seed_dir:
        seed = os.path.join(ARGS.seed_dir, case, f"{cores}cores", "plan.json")
        if os.path.isfile(seed):
            cmd += ["--seed-plan", seed]

    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    sec = time.time() - t

    base = {"case": case, "cores": cores, "seconds": sec}
    if p.returncode != 0:
        return dict(base, status="error", error=p.stderr[-1000:])
    try:
        x = json.loads(p.stdout)
    except Exception as e:
        return dict(
            base,
            status="error",
            error=(
                f"Failed to parse solver stdout as JSON: {repr(e)}\n"
                f"stdout tail:\n{p.stdout[-1000:]}\n"
                f"stderr tail:\n{p.stderr[-1000:]}"
            ),
        )
    return dict(
        base,
        status="ok",
        single=x["single"],
        makespan=x["best_makespan"],
        speedup=x["speedup"],
        added_copy_bytes=x.get("added_copy_bytes"),
        used_cores=x.get("used_cores"),
        official_evaluations=x.get("official_evaluations"),
        selected=x["selected"],
        plan=plan,
    )


def run_case(case):
    """Solve all requested core counts of one case in ascending order."""
    previous = None
    for cores in CORES:
        with LOCK:
            done = ROWS.get((case, cores))
        if done is not None:
            previous = done["plan"]
            continue
        row = run_one(case, cores, None if ARGS.no_warm else previous)
        with LOCK:
            ROWS[(case, cores)] = row
            if row["status"] == "ok":
                SINGLE_CACHE[case] = row["single"]
            write()
        if row["status"] == "ok":
            previous = row["plan"]
            print(
                case,
                f"{cores}c",
                "speedup=",
                round(row["speedup"], 4),
                "makespan=",
                row["makespan"],
                "sec=",
                round(row["seconds"], 1),
                "selected=",
                row["selected"][:80],
                flush=True,
            )
        else:
            print(case, f"{cores}c", row["status"], row["error"][-300:], flush=True)
    return case


print("=" * 70)
print("Solver       :", SOLVER)
print("Solver name  :", SOLVER_NAME)
print("Data dir     :", DATA)
print("Code dir     :", CODE)
print("Output root  :", OUTPUT_ROOT)
print("Plan dir     :", OUT_DIR)
print("CSV          :", CSV_PATH)
print("Summary      :", SUMMARY_PATH)
print("Cases        :", len(CASES))
print("Cores        :", CORES)
print("Time limit   :", ARGS.time_limit)
print("Workers      :", MAX_WORKERS)
print("Resumed rows :", len(ROWS))
print("=" * 70)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    fs = [ex.submit(run_case, c) for c in CASES]
    for f in as_completed(fs):
        f.result()


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

def stats(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "avg": None, "median": None, "min": None, "max": None,
                "gt4": 0, "gt45": 0}
    return {
        "n": len(values),
        "avg": sum(values) / len(values),
        "median": (values[(len(values) - 1) // 2] + values[len(values) // 2]) / 2,
        "min": values[0],
        "max": values[-1],
        "gt4": sum(x >= 4 for x in values),
        "gt45": sum(x >= 4.5 for x in values),
    }


selected_rows = [ROWS[(c, k)] for c in CASES for k in CORES if (c, k) in ROWS]
ok = [r for r in selected_rows if r["status"] == "ok"]
per_cores = {
    str(k): stats([r["speedup"] for r in ok if int(r["cores"]) == k])
    for k in CORES
}
# Speedup curve over cases that finished every requested core count.
complete = [
    c for c in CASES
    if all((c, k) in ROWS and ROWS[(c, k)]["status"] == "ok" for k in CORES)
]
curve = {"1": 1.0}
for k in CORES:
    values = [ROWS[(c, k)]["speedup"] for c in complete]
    curve[str(k)] = sum(values) / len(values) if values else None

summary = {
    "solver": SOLVER_NAME,
    "solver_path": SOLVER,
    "time_limit": ARGS.time_limit,
    "cores": CORES,
    "requested": len(CASES) * len(CORES),
    "ok": len(ok),
    "failed": len(CASES) * len(CORES) - len(ok),
    "cases": sorted(CASES),
    "per_cores": per_cores,
    "avg_speedup_curve_complete_cases": curve,
    "complete_cases": len(complete),
    "failed_items": sorted(
        f"{r['case']}@{r['cores']}" for r in selected_rows if r["status"] != "ok"
    ),
    "output_csv": CSV_PATH,
    "output_plans": OUT_DIR,
}

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
