#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run a specified solver on every case in the configured data directory.

Usage:
    python run_V8_50cases.py
    python run_V8_50cases.py V8_solver_50case.py
    python run_V8_50cases.py V8_solver_50case.py ./results

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
        default="V8_solver_50case.py",
        help=(
            "Solver Python file. "
            "Can be a filename relative to this runner, or an absolute path. "
            "Default: V8_solver_50case.py"
        ),
    )

    parser.add_argument(
        "output_dir",
        nargs="?",
        default=ROOT,
        help=(
            "Output root directory. "
            "Default: current script directory."
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
#
# V8_solver_50case.py
# ->
# V8_solver_50case
SOLVER_NAME = os.path.splitext(
    os.path.basename(SOLVER)
)[0]


# ---------------------------------------------------------
# Output paths
# ---------------------------------------------------------

OUTPUT_ROOT = os.path.abspath(ARGS.output_dir)
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

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Cases
# ---------------------------------------------------------

CASES = sorted(
    os.path.basename(fp)[:-5]
    for fp in glob.glob(
        os.path.join(DATA, "case_*.json")
    )
)

MAX_WORKERS = int(
    os.environ.get("V8_MAX_WORKERS", "12")
)


def run_one(case):
    graph = os.path.join(
        DATA,
        case + ".json"
    )

    plan = os.path.join(
        OUT_DIR,
        case + "_5core.json"
    )

    env = os.environ.copy()
    env["HUAWEI_CODE_DIR"] = CODE

    t = time.time()

    p = subprocess.run(
        [
            sys.executable,
            SOLVER,
            graph,
            "5",
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
            "status": "error",
            "seconds": sec,
            "error": p.stderr[-1000:],
        }

    try:
        x = json.loads(p.stdout)
    except Exception as e:
        return {
            "case": case,
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
        "status": "ok",
        "seconds": sec,
        "single": x["single"],
        "makespan": x["best_makespan"],
        "speedup": x["speedup"],
        "selected": x["selected"],
        "plan": plan,
    }


def write(rows):
    cols = [
        "case",
        "status",
        "seconds",
        "single",
        "makespan",
        "speedup",
        "selected",
        "plan",
        "error",
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=cols,
            extrasaction="ignore",
        )

        w.writeheader()

        w.writerows(
            sorted(
                rows,
                key=lambda r: r["case"]
            )
        )


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
print("Workers      :", MAX_WORKERS)
print("=" * 70)


rows = []

with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as ex:

    fs = {
        ex.submit(run_one, c): c
        for c in CASES
    }

    for f in as_completed(fs):

        r = f.result()

        rows.append(r)

        # 每完成一个 case 就刷新 CSV，
        # 中途终止也不会丢掉已经完成的结果。
        write(rows)

        if r["status"] == "ok":

            print(
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
                r["case"],
                r["status"],
                flush=True,
            )


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

ok = [
    r
    for r in rows
    if r["status"] == "ok"
]

sp = sorted(
    r["speedup"]
    for r in ok
)

summary = {
    "solver": SOLVER_NAME,
    "solver_path": SOLVER,

    "requested": len(CASES),
    "ok": len(ok),
    "failed": len(CASES) - len(ok),

    "cases": CASES,

    "avg": (
        sum(sp) / len(sp)
        if sp else None
    ),

    "median": (
        (
            sp[(len(sp) - 1) // 2]
            + sp[len(sp) // 2]
        ) / 2
        if sp else None
    ),

    "gt4": sum(
        x >= 4
        for x in sp
    ),

    "gt45": sum(
        x >= 4.5
        for x in sp
    ),

    "min": (
        min(sp)
        if sp else None
    ),

    "max": (
        max(sp)
        if sp else None
    ),

    "output_csv": CSV_PATH,
    "output_plans": OUT_DIR,
}


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2,
    )


print(
    "SUMMARY",
    json.dumps(
        summary,
        ensure_ascii=False
    ),
    flush=True,
)

'''
python run_all_cases.py V8.py ./V8_result

python run_all_cases.py V9_1.py ./V9_1_result
python run_all_cases.py V9_2.py ./V9_2_result
'''