#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run a specified solver on every case for one or more core counts.

Examples:
    # Backward-compatible default: 5 cores, quick mode
    python run_all_cases_multicore.py

    # Single core-count selection
    python run_all_cases_multicore.py V10_final.py ./results quick --cores 5

    # Multiple core counts
    python run_all_cases_multicore.py V10_final.py ./results quick --cores 2 3 4 5

    # Range syntax
    python run_all_cases_multicore.py V10_final.py ./results quick --cores 2-5

    # Mixed syntax also works
    python run_all_cases_multicore.py V10_final.py ./results all --cores 2,4-5

Optional environment variables:
    HUAWEI_DATA_DIR
    HUAWEI_CODE_DIR
    V8_MAX_WORKERS

Notes:
    - --cores defaults to 5, so old usage still works.
    - Supported core counts are 2, 3, 4, 5.
    - With multiple core counts, every (case, core_count) pair is submitted as
      an independent job to the thread pool.
"""

import os
import sys
import glob
import json
import csv
import time
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


def parse_core_specs(values):
    """Parse core specifications such as:

        ["5"]
        ["2", "3", "4", "5"]
        ["2-5"]
        ["2,4-5"]

    Returns a sorted unique list of integers.
    """
    cores = set()

    for raw in values:
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue

            if "-" in token:
                parts = token.split("-", 1)
                if len(parts) != 2:
                    raise argparse.ArgumentTypeError(
                        f"Invalid core range: {token}"
                    )
                try:
                    lo = int(parts[0])
                    hi = int(parts[1])
                except ValueError:
                    raise argparse.ArgumentTypeError(
                        f"Invalid core range: {token}"
                    )

                if lo > hi:
                    lo, hi = hi, lo

                for k in range(lo, hi + 1):
                    cores.add(k)
            else:
                try:
                    cores.add(int(token))
                except ValueError:
                    raise argparse.ArgumentTypeError(
                        f"Invalid core count: {token}"
                    )

    if not cores:
        raise argparse.ArgumentTypeError("At least one core count is required.")

    invalid = sorted(k for k in cores if k < 2 or k > 5)
    if invalid:
        raise argparse.ArgumentTypeError(
            "Only core counts 2-5 are supported; invalid: "
            + ", ".join(map(str, invalid))
        )

    return sorted(cores)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a solver on Huawei multicore scheduling cases for one or "
            "more core counts."
        )
    )

    parser.add_argument(
        "solver",
        nargs="?",
        default="V8_solver_50case.py",
        help=(
            "Solver Python file. Can be a filename relative to this runner, "
            "or an absolute path. Default: V8_solver_50case.py"
        ),
    )

    parser.add_argument(
        "output_dir",
        nargs="?",
        default=ROOT,
        help="Output root directory. Default: current script directory.",
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
        "--cores",
        nargs="+",
        default=["5"],
        metavar="N",
        help=(
            "Core counts to run. Supports: '--cores 5', '--cores 2 3 4 5', "
            "'--cores 2-5', or '--cores 2,4-5'. Default: 5."
        ),
    )

    args = parser.parse_args()
    try:
        args.cores = parse_core_specs(args.cores)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))
    return args


ARGS = parse_args()
CORES = ARGS.cores


# ---------------------------------------------------------
# Solver path
# ---------------------------------------------------------

if os.path.isabs(ARGS.solver):
    SOLVER = ARGS.solver
else:
    SOLVER = os.path.join(ROOT, ARGS.solver)

SOLVER = os.path.abspath(SOLVER)

if not os.path.isfile(SOLVER):
    raise FileNotFoundError(f"Solver not found: {SOLVER}")

SOLVER_NAME = os.path.splitext(os.path.basename(SOLVER))[0]


# ---------------------------------------------------------
# Output paths
# ---------------------------------------------------------

OUTPUT_ROOT = os.path.abspath(ARGS.output_dir)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

CORE_TAG = "-".join(map(str, CORES)) + "core"

OUT_DIR = os.path.join(
    OUTPUT_ROOT,
    f"{SOLVER_NAME}_{CORE_TAG}_allcase_plans"
)

CSV_PATH = os.path.join(
    OUTPUT_ROOT,
    f"{SOLVER_NAME}_{CORE_TAG}_allcase_results.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_ROOT,
    f"{SOLVER_NAME}_{CORE_TAG}_allcase_summary.json"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Cases
# ---------------------------------------------------------

ALL_CASES = sorted(
    os.path.basename(fp)[:-5]
    for fp in glob.glob(os.path.join(DATA, "case_*.json"))
)


# Historical 5-core V8 runtime.  It is used only as a rough case-size proxy for
# quick filtering and LPT ordering; it does NOT affect solver evaluation.
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

if ARGS.mode == "all":
    CASES = sorted(
        ALL_CASES,
        key=lambda c: V8_HISTORY_SECONDS.get(c, 0.0),
        reverse=True,
    )
else:
    CASES = [
        c
        for c in ALL_CASES
        if V8_HISTORY_SECONDS.get(c, 0.0) <= QUICK_MAX_SECONDS
    ]
    CASES.sort(
        key=lambda c: V8_HISTORY_SECONDS.get(c, 0.0),
        reverse=True,
    )

MAX_WORKERS = int(os.environ.get("V8_MAX_WORKERS", "20"))


# ---------------------------------------------------------
# Worker
# ---------------------------------------------------------

def run_one(case, cores):
    graph = os.path.join(DATA, case + ".json")

    plan = os.path.join(
        OUT_DIR,
        f"{case}_{cores}core.json"
    )

    env = os.environ.copy()
    env["HUAWEI_CODE_DIR"] = CODE

    t = time.time()

    p = subprocess.run(
        [
            sys.executable,
            SOLVER,
            graph,
            str(cores),
            plan,
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    sec = time.time() - t

    if p.returncode != 0:
        return {
            "case": case,
            "cores": cores,
            "status": "error",
            "seconds": sec,
            "error": p.stderr[-1000:],
        }

    try:
        x = json.loads(p.stdout)
    except Exception as e:
        return {
            "case": case,
            "cores": cores,
            "status": "error",
            "seconds": sec,
            "error": (
                f"Failed to parse solver stdout as JSON: {repr(e)}\n"
                f"stdout tail:\n{p.stdout[-1000:]}\n"
                f"stderr tail:\n{p.stderr[-1000:]}"
            ),
        }

    return {
        "case": case,
        "cores": cores,
        "status": "ok",
        "seconds": sec,
        "single": x["single"],
        "makespan": x["best_makespan"],
        "speedup": x["speedup"],
        "selected": x["selected"],
        "plan": plan,
    }


# ---------------------------------------------------------
# Output helpers
# ---------------------------------------------------------

def write(rows):
    cols = [
        "case",
        "cores",
        "status",
        "seconds",
        "single",
        "makespan",
        "speedup",
        "selected",
        "plan",
        "error",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=cols,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(
            sorted(
                rows,
                key=lambda r: (r.get("cores", 0), r["case"])
            )
        )


def calc_stats(rows):
    ok = [r for r in rows if r.get("status") == "ok"]
    sp = sorted(r["speedup"] for r in ok)
    return {
        "requested": len(rows),
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "avg": (sum(sp) / len(sp)) if sp else None,
        "median": (
            (sp[(len(sp) - 1) // 2] + sp[len(sp) // 2]) / 2
            if sp else None
        ),
        "gt4": sum(x >= 4 for x in sp),
        "gt45": sum(x >= 4.5 for x in sp),
        "min": min(sp) if sp else None,
        "max": max(sp) if sp else None,
    }


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

TOTAL_JOBS = len(CASES) * len(CORES)

print("=" * 76)
print("Solver       :", SOLVER)
print("Solver name  :", SOLVER_NAME)
print("Data dir     :", DATA)
print("Code dir     :", CODE)
print("Output root  :", OUTPUT_ROOT)
print("Plan dir     :", OUT_DIR)
print("CSV          :", CSV_PATH)
print("Summary      :", SUMMARY_PATH)
print("Mode         :", ARGS.mode)
print("Cases        :", len(CASES))
print("Core counts  :", CORES)
print("Total jobs   :", TOTAL_JOBS)
print("Workers      :", MAX_WORKERS)
print("=" * 76)

rows = []

# Submit slow cases first for every selected core count.  CASES is already LPT
# ordered; nesting case -> cores therefore gives every large case an early slot.
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    fs = {
        ex.submit(run_one, case, cores): (case, cores)
        for case in CASES
        for cores in CORES
    }

    for f in as_completed(fs):
        r = f.result()
        rows.append(r)

        # Refresh CSV after every completed (case, core) job so interrupted runs
        # still preserve all completed results.
        write(rows)

        if r["status"] == "ok":
            print(
                f"[{r['cores']}core]",
                r["case"],
                "speedup=",
                round(r["speedup"], 4),
                "selected=",
                r["selected"],
                "sec=",
                round(r["seconds"], 1),
                flush=True,
            )
        else:
            print(
                f"[{r['cores']}core]",
                r["case"],
                r["status"],
                flush=True,
            )


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

by_core = {}
for cores in CORES:
    core_rows = [r for r in rows if r.get("cores") == cores]
    stats = calc_stats(core_rows)
    stats["cases"] = [r["case"] for r in sorted(core_rows, key=lambda x: x["case"])]
    by_core[str(cores)] = stats

overall = calc_stats(rows)

summary = {
    "solver": SOLVER_NAME,
    "solver_path": SOLVER,
    "mode": ARGS.mode,
    "core_counts": CORES,
    "requested_cases": len(CASES),
    "requested_jobs": TOTAL_JOBS,
    "completed_jobs": len(rows),
    "overall": overall,
    "by_core": by_core,
    "output_csv": CSV_PATH,
    "output_plans": OUT_DIR,
}

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(
    "SUMMARY",
    json.dumps(summary, ensure_ascii=False),
    flush=True,
)


# Examples:
#   python run_all_cases.py V10_final.py ./V10_5core_result quick --cores 5
#   python run_all_cases.py V10_final.py ./V10_2to5_result quick --cores 2-5
#   python run_all_cases.py V10_final.py ./V10_result all --cores 2 3 4 5
