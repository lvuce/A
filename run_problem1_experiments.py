#!/usr/bin/env python3
"""Run and resume the official Problem 1 evaluation for 100 cases and 2–5 cores."""

import argparse
import csv
import fcntl
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from A题_problem1_solver_v4_2 import evaluate_baseline, solve


ROOT = Path(__file__).resolve().parent
DATA = ROOT / '通用神经网络处理器下的多核调度问题  附件' / 'data'
FIELDS = ('case', 'cores', 'status', 'seconds', 'selected', 'single', 'makespan',
          'speedup', 'added_copy', 'subgraphs', 'error')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    os.replace(temporary, path)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_or_evaluate_baseline(case, graph, config, output_dir):
    case_dir = Path(output_dir) / case
    case_dir.mkdir(parents=True, exist_ok=True)
    cache = case_dir / 'singlecore_result.json'
    with (case_dir / 'singlecore_result.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache.is_file():
            result = json.loads(cache.read_text(encoding='utf-8'))
        else:
            result = evaluate_baseline(graph, config)
            write_json(cache, result)
        return result


def run_one(case, cores, data_dir, config, output_dir):
    started = time.monotonic()
    job_dir = Path(output_dir) / case / f'{cores}cores'
    metric_path = job_dir / 'metrics.json'
    try:
        with (Path(data_dir) / f'{case}.json').open(encoding='utf-8') as handle:
            graph = json.load(handle)
        baseline = load_or_evaluate_baseline(case, graph, config, output_dir)
        plan, info, result = solve(graph, cores, config, return_result=True,
                                   baseline_result=baseline)
        if result['makespan'] != info['best_makespan']:
            raise ValueError('Solver and official evaluator makespans differ')
        if len(plan['core_schedules']) != cores:
            raise ValueError('Plan has an incorrect number of cores')
        row = dict(case=case, cores=cores, status='ok', seconds=time.monotonic()-started,
                   selected=info['selected'], single=info['single_makespan'],
                   makespan=result['makespan'], speedup=info['speedup'],
                   added_copy=result['data_movement_bytes']['added_copy_bytes'],
                   subgraphs=len(set(plan['node_to_subgraph'].values())), error='')
        write_json(job_dir / 'plan.json', plan)
        write_json(job_dir / 'official_result.json', result)
        write_json(job_dir / 'solver_info.json', info)
        write_json(metric_path, row)
        return row
    except Exception:
        row = dict(case=case, cores=cores, status='error', seconds=time.monotonic()-started,
                   selected='', single='', makespan='', speedup='', added_copy='',
                   subgraphs='', error=traceback.format_exc())
        write_json(metric_path, row)
        return row


def completed_row(output_dir, case, cores):
    job_dir = output_dir / case / f'{cores}cores'
    paths = [job_dir / name for name in ('plan.json', 'official_result.json',
                                        'solver_info.json', 'metrics.json')]
    if not all(path.is_file() for path in paths):
        return None
    try:
        row = json.loads(paths[-1].read_text(encoding='utf-8'))
        if row.get('status') != 'ok':
            return None
        result = json.loads(paths[1].read_text(encoding='utf-8'))
        if result['makespan'] != row['makespan']:
            return None
        return row
    except (OSError, ValueError, KeyError):
        return None


def summarize(output_dir, cases, cores_list, rows):
    ordered = [rows[(case, cores)] for case in cases for cores in cores_list
               if (case, cores) in rows]
    write_csv(output_dir / 'per_case.csv', ordered, FIELDS)
    baselines = {}
    for row in ordered:
        if row['status'] == 'ok':
            case = row['case']
            if case in baselines and baselines[case] != row['single']:
                raise ValueError(f'Inconsistent single-core baseline for {case}')
            baselines[case] = row['single']
    write_csv(output_dir / 'singlecore_baseline.csv',
              [dict(case=case, makespan=baselines[case]) for case in cases if case in baselines],
              ('case', 'makespan'))
    summary = [dict(cores=1, cases_expected=len(cases), cases_ok=len(baselines),
                    average_speedup=1.0 if baselines else '', total_seconds='',
                    total_added_copy_bytes='')]
    for cores in cores_list:
        successes = [row for row in ordered if row['cores'] == cores and row['status'] == 'ok']
        summary.append(dict(cores=cores, cases_expected=len(cases), cases_ok=len(successes),
                            average_speedup=(sum(row['speedup'] for row in successes) /
                                             len(successes) if successes else ''),
                            total_seconds=sum(row['seconds'] for row in successes),
                            total_added_copy_bytes=sum(row['added_copy'] for row in successes)))
    write_csv(output_dir / 'summary.csv', summary,
              ('cores', 'cases_expected', 'cases_ok', 'average_speedup',
               'total_seconds', 'total_added_copy_bytes'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'results' / 'problem1_v4_2')
    parser.add_argument('--cores', type=int, nargs='+', default=[2, 3, 4, 5])
    parser.add_argument('--cases', nargs='+', help='Optional case names, e.g. case_001 case_002')
    parser.add_argument('--max-ops', type=int, help='Keep cases with at most this many ops')
    parser.add_argument('--max-edges', type=int, help='Keep cases with at most this many edges')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--force', action='store_true', help='Rerun successful jobs')
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    config = (args.config or data_dir / 'config.txt').resolve()
    output_dir = args.out.resolve()
    if not config.is_file():
        parser.error(f'Configuration file not found: {config}')
    candidates = sorted(args.cases if args.cases else
                        (path.stem for path in data_dir.glob('case_*.json')))
    if not candidates or any(not (data_dir / f'{case}.json').is_file() for case in candidates):
        parser.error('No valid case JSON files were selected')
    selection = []
    for case in candidates:
        with (data_dir / f'{case}.json').open(encoding='utf-8') as handle:
            graph = json.load(handle)
        ops, edges = len(graph['ops']), len(graph['edges'])
        selected = ((args.max_ops is None or ops <= args.max_ops) and
                    (args.max_edges is None or edges <= args.max_edges))
        selection.append(dict(case=case, ops=ops, edges=edges, selected=int(selected)))
    cases = [row['case'] for row in selection if row['selected']]
    if not cases:
        parser.error('No cases satisfy the size limits')
    cores_list = sorted(set(args.cores))
    if any(cores < 2 or cores > 5 for cores in cores_list):
        parser.error('Core counts must be between 2 and 5')
    if args.workers < 1:
        parser.error('--workers must be positive')
    write_csv(output_dir / 'selection.csv', selection, ('case', 'ops', 'edges', 'selected'))
    rows = {}
    pending = []
    for case in cases:
        for cores in cores_list:
            row = None if args.force else completed_row(output_dir, case, cores)
            if row is None:
                pending.append((case, cores))
            else:
                rows[(case, cores)] = row
    pending.sort(key=lambda pair: (pair[1], pair[0]))
    summarize(output_dir, cases, cores_list, rows)
    print(f'{len(cases)} cases, {len(cores_list)} core counts, '
          f'{len(rows)} resumed, {len(pending)} pending; output={output_dir}', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, case, cores, data_dir, config, output_dir): (case, cores)
                   for case, cores in pending}
        for future in as_completed(futures):
            case, cores = futures[future]
            row = future.result()
            rows[(case, cores)] = row
            summarize(output_dir, cases, cores_list, rows)
            print(f'{len(rows)}/{len(cases)*len(cores_list)} {case} {cores} cores '
                  f'{row["status"]} makespan={row["makespan"]} '
                  f'speedup={row["speedup"]} seconds={row["seconds"]:.1f}', flush=True)
    errors = [row for row in rows.values() if row['status'] != 'ok']
    if errors:
        print(f'{len(errors)} jobs failed; see per_case.csv and metrics.json', flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
