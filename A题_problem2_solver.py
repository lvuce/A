#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Problem 2 / Scene B solver (V3).

Scene B merges every core into one Task, so the decisions that matter are
(a) which core each operation runs on and (b) the order of operations on each
core: Step3 issues every Pipe strictly in the projected sequence order, so a
cross-core COPY_IN placed too early blocks the whole MTE2 pipe of its core.

Pipeline (every accepted plan is scored by the official Problem-2 evaluator):

P0  Seed portfolio: Problem-1 (V8) structural plans evaluated natively, plus
    optional external seed plans, ranked by a cheap list-schedule estimate.
    Each promising core map is then re-realised with several per-core orders
    (list-schedule time, time windows x Step1-DFS, pure DFS).
P1  No heavy proxy: the only screen is an O(E log V) list-schedule estimate;
    candidates are accepted on official makespan only.
P2  Slack-gated co-location clustering: an edge is forced onto one core only
    when its cross-core latency (500 + 2B/BW) cannot be hidden by its slack,
    or when it is a large transfer in a DDR-bound graph.  Clusters never
    absorb two critical ops that overlap in time on the same pipe.
P3  Slack-aware cluster mapping: HEFT-like placement with lookahead load,
    communication cut, and a penalty for stacking overlapping critical work on
    one core (the Scene-B analogue of slack level packing).
P4  Critical-path local search on the official trace: cross-core edges and
    pipe contention on the critical chain drive unit moves, which are
    screened by the estimate and realised in official start-time order.

Usage:
    python A题_problem2_solver.py graph.json [num_cores=5] [output_plan.json]
        [--config config.txt] [--time-limit 300] [--single N]
        [--seed-plan plan.json ...] [--verbose]
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ATTACHMENT = ROOT / "通用神经网络处理器下的多核调度问题  附件"
OFFICIAL_CODE = Path(os.environ.get("HUAWEI_CODE_DIR", ATTACHMENT / "code"))
if not OFFICIAL_CODE.is_dir():
    raise FileNotFoundError(f"Official evaluator directory is missing: {OFFICIAL_CODE}")
sys.path.insert(0, str(OFFICIAL_CODE))
# V8 reads HUAWEI_CODE_DIR at import time; point it at the same evaluator.
os.environ.setdefault("HUAWEI_CODE_DIR", str(OFFICIAL_CODE))

from evaluation_validation import read_evaluation_config  # noqa: E402
from multicore_cut_evaluate_problem_2 import (  # noqa: E402
    evaluate_scene_b,
    read_scene_b_config,
)
from schedule_step1 import step1_schedule  # noqa: E402
from singlecore_evaluate import evaluate_singlecore  # noqa: E402

try:  # Problem-1 structure generators (optional, read-only reuse).
    if str(ROOT) not in sys.path:
        sys.path.insert(1, str(ROOT))
    import V8 as _V8  # noqa: E402
except Exception:  # pragma: no cover - solver still works without V8
    _V8 = None


EXCLUDED = {"COPY_IN", "COPY_OUT"}
MASK_BUCKETS = 256


def popcount(value):
    """int.bit_count() needs Python 3.10; keep 3.9 compatibility."""
    return bin(value).count("1")


def log(verbose, *parts):
    if verbose:
        print(*parts, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Graph model
# ---------------------------------------------------------------------------

class Model:
    """Contracted op DAG (COPY ops removed) plus tensor/traffic views."""

    def __init__(self, graph):
        self.graph = graph
        self.op_by = {int(op["id"]): op for op in graph["ops"]}
        self.tensor_by = {int(t["id"]): t for t in graph["tensors"]}
        op_ids = set(self.op_by)
        producers = defaultdict(set)
        consumers = defaultdict(set)
        direct = []
        for edge in graph["edges"]:
            source, target = int(edge["source"]), int(edge["target"])
            source_is_op, target_is_op = source in op_ids, target in op_ids
            if source_is_op and not target_is_op:
                producers[target].add(source)
            elif target_is_op and not source_is_op:
                consumers[source].add(target)
            elif source_is_op and target_is_op and source != target:
                direct.append((source, target, max(0, int(edge.get("data_size", 0)))))

        self.nodes = tuple(sorted(
            op_id for op_id, op in self.op_by.items()
            if op.get("op") not in EXCLUDED
        ))
        self.node_set = set(self.nodes)
        self.n = len(self.nodes)
        self.index = {op_id: i for i, op_id in enumerate(self.nodes)}
        self.dur = {
            op_id: max(1.0, float(self.op_by[op_id].get("cycles", 0)))
            for op_id in self.nodes
        }
        self.pipe = {op_id: self.op_by[op_id].get("pipe") for op_id in self.nodes}

        self.pred, self.succ = self._contract(op_ids, producers, consumers, direct)
        self.topological = self._topological_order()

        self.edge_bytes = defaultdict(int)
        for tid, tensor in self.tensor_by.items():
            size = int(tensor.get("size", 0))
            for source in producers[tid] & self.node_set:
                for target in consumers[tid] & self.node_set:
                    if source != target:
                        self.edge_bytes[(source, target)] += size
        for source, target, size in direct:
            if source in self.node_set and target in self.node_set:
                self.edge_bytes[(source, target)] += size
        for source in self.nodes:
            for target in self.succ[source]:
                self.edge_bytes[(source, target)] += 0

        # Mandatory DDR traffic: graph inputs (read) and final outputs (write).
        self.inputs = {op_id: [] for op_id in self.nodes}
        self.finals = {op_id: [] for op_id in self.nodes}
        self.input_consumers = {}
        mandatory = 0
        for tid, tensor in self.tensor_by.items():
            size = int(tensor.get("size", 0))
            eligible_producers = producers[tid] & self.node_set
            eligible_consumers = consumers[tid] & self.node_set
            if eligible_consumers and not eligible_producers:
                for op_id in eligible_consumers:
                    self.inputs[op_id].append((tid, size))
                self.input_consumers[tid] = sorted(eligible_consumers)
                mandatory += size
            has_copy_out = any(
                self.op_by[op_id].get("op") == "COPY_OUT"
                for op_id in consumers[tid] if op_id in self.op_by
            )
            if eligible_producers and (has_copy_out or not eligible_consumers):
                for op_id in eligible_producers:
                    self.finals[op_id].append((tid, size))
                mandatory += size
        self.mandatory_bytes = mandatory
        self.pipe_total = defaultdict(float)
        for op_id in self.nodes:
            self.pipe_total[self.pipe[op_id]] += self.dur[op_id]

        self.dfs_pos = self._dfs_positions(graph)
        self._zero_comm_levels()

    def _contract(self, op_ids, producers, consumers, direct):
        succ0 = {op_id: set() for op_id in op_ids}
        for source, target, _ in direct:
            succ0[source].add(target)
        for tid, sources in producers.items():
            for source in sources:
                for target in consumers.get(tid, ()):
                    if source != target:
                        succ0[source].add(target)
        pred = {op_id: set() for op_id in self.nodes}
        succ = {op_id: set() for op_id in self.nodes}
        for source in self.nodes:
            stack = list(succ0[source])
            seen = set()
            while stack:
                target = stack.pop()
                if target in self.node_set:
                    if target != source:
                        succ[source].add(target)
                        pred[target].add(source)
                elif target not in seen:
                    seen.add(target)
                    stack.extend(succ0.get(target, ()))
        return pred, succ

    def _topological_order(self):
        indegree = {op_id: len(self.pred[op_id]) for op_id in self.nodes}
        ready = [op_id for op_id in self.nodes if indegree[op_id] == 0]
        heapq.heapify(ready)
        order = []
        while ready:
            source = heapq.heappop(ready)
            order.append(source)
            for target in self.succ[source]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    heapq.heappush(ready, target)
        if len(order) != self.n:
            raise ValueError("contracted operation graph contains a cycle")
        return tuple(order)

    def _dfs_positions(self, graph):
        """Positions in the official Step1 order (memory-friendly DFS)."""
        try:
            sequence = [op_id for op_id in step1_schedule(graph)
                        if op_id in self.node_set]
        except Exception:
            sequence = []
        if len(sequence) != self.n or not self.is_topological(sequence):
            sequence = list(self.topological)
        return {op_id: i for i, op_id in enumerate(sequence)}

    def _zero_comm_levels(self):
        """Pipe-agnostic, communication-free longest paths used for slack."""
        est_end = {}
        for op_id in self.topological:
            est_end[op_id] = self.dur[op_id] + max(
                (est_end[p] for p in self.pred[op_id]), default=0.0)
        bottom = {}
        for op_id in reversed(self.topological):
            bottom[op_id] = self.dur[op_id] + max(
                (bottom[s] for s in self.succ[op_id]), default=0.0)
        self.est_end = est_end
        self.bottom = bottom
        self.critical_length = max(1.0, max(est_end.values(), default=1.0))

    def is_topological(self, order):
        position = {op_id: i for i, op_id in enumerate(order)}
        if len(position) != self.n:
            return False
        return all(
            position[source] < position[target]
            for source in self.nodes for target in self.succ[source]
        )


class Grouping:
    """A partition of ops into subgraphs, with a cached quotient DAG."""

    def __init__(self, model, members):
        self.members = [tuple(group) for group in members]
        self.owner = {}
        for gid, group in enumerate(self.members):
            for op_id in group:
                self.owner[op_id] = gid
        self.model = model
        self._quotient = None

    def __len__(self):
        return len(self.members)

    @property
    def is_fine(self):
        return len(self.members) == self.model.n

    def quotient(self):
        if self._quotient is None:
            qsucc = [set() for _ in self.members]
            for source in self.model.nodes:
                left = self.owner[source]
                for target in self.model.succ[source]:
                    right = self.owner[target]
                    if left != right:
                        qsucc[left].add(right)
            indegree = [0] * len(self.members)
            for targets in qsucc:
                for target in targets:
                    indegree[target] += 1
            self._quotient = (qsucc, indegree)
        return self._quotient


@dataclass
class Evaluated:
    name: str
    plan: dict
    result: dict
    objective: tuple
    grouping: Grouping
    group_core: tuple
    family: str


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class Solver:
    def __init__(self, graph, num_cores, config_path, time_limit=300.0,
                 seed_plans=(), patience=None, verbose=False):
        self.graph = graph
        self.model = Model(graph)
        self.K = int(num_cores)
        self.time_limit = max(5.0, float(time_limit))
        self.verbose = verbose
        common = read_evaluation_config(config_path)
        scene = read_scene_b_config(config_path)
        self.bandwidth = common["bandwidth"]
        self.capacity = dict(common["capacity"])
        self.cross_delay = scene["cross_core_copy_delay_cycles"]
        self.eval_kwargs = {
            "bandwidth": self.bandwidth,
            "capacity": self.capacity,
            "cross_core_copy_delay": self.cross_delay,
        }
        self.bw = float(self.bandwidth)
        self.delay = float(self.cross_delay)
        self.seed_plans = list(seed_plans)
        self.patience = patience
        self.fine = Grouping(self.model, [(op_id,) for op_id in self.model.nodes])

        compute_bound = max(self.model.pipe_total.values(), default=1.0) / max(1, self.K)
        ddr_bound = self.model.mandatory_bytes / self.bw
        self.ddr_pressure = ddr_bound / max(1.0, compute_bound)
        self.gamma = 1.0 + 0.5 * min(max(0, self.K - 1), self.ddr_pressure * self.K)

        self.best = None
        self.pool = []  # (objective, name, grouping, group_core)
        self.seen = {}
        self.records = []
        self.eval_times = []
        self.accepted = defaultdict(int)
        self._cluster_cache = {}

    # -- basic helpers ------------------------------------------------------

    def core_of(self, grouping, group_core):
        return {op_id: group_core[grouping.owner[op_id]] for op_id in self.model.nodes}

    def fine_core(self, core):
        return tuple(core[op_id] for op_id in self.model.nodes)

    def structure_from_plan(self, plan):
        mapping = {int(k): int(v) for k, v in plan["node_to_subgraph"].items()}
        sg_core = {
            int(sg): core_id
            for core_id, schedule in enumerate(plan["core_schedules"])
            for sg in schedule
        }
        subgraphs = sorted(set(mapping.values()))
        if len(subgraphs) == self.model.n:
            core = {op_id: sg_core[mapping[op_id]] for op_id in self.model.nodes}
            return self.fine, self.fine_core(core)
        index = {sg: i for i, sg in enumerate(subgraphs)}
        members = [[] for _ in subgraphs]
        for op_id in self.model.nodes:
            members[index[mapping[op_id]]].append(op_id)
        return Grouping(self.model, members), tuple(sg_core[sg] for sg in subgraphs)

    def avg_eval_time(self):
        recent = self.eval_times[-6:]
        return sum(recent) / len(recent) if recent else 1.0

    def affordable(self, until):
        return time.monotonic() + 0.8 * self.avg_eval_time() <= until

    # -- ordering -----------------------------------------------------------

    def op_keys(self, family, start=None, official=None):
        dfs = self.model.dfs_pos
        if family == "dfs":
            return {op_id: (dfs[op_id],) for op_id in self.model.nodes}
        if family == "time":
            return {op_id: (start[op_id], dfs[op_id]) for op_id in self.model.nodes}
        if family.startswith("win"):
            buckets = max(1, int(family[3:]))
            horizon = max(start.values(), default=0.0) + 1.0
            width = horizon / buckets
            return {op_id: (int(start[op_id] // width), dfs[op_id])
                    for op_id in self.model.nodes}
        if family == "otime":
            return {op_id: (official.get(op_id, 0.0), dfs[op_id])
                    for op_id in self.model.nodes}
        raise ValueError(family)

    def realize(self, grouping, group_core, keys):
        """Kahn order of the subgraph quotient, highest priority = smallest key."""
        qsucc, qindegree = grouping.quotient()
        priority = [min(keys[op_id] for op_id in group) for group in grouping.members]
        indegree = list(qindegree)
        heap = [(priority[g], g) for g in range(len(grouping)) if indegree[g] == 0]
        heapq.heapify(heap)
        schedules = [[] for _ in range(self.K)]
        placed = 0
        while heap:
            _, gid = heapq.heappop(heap)
            schedules[group_core[gid]].append(gid)
            placed += 1
            for target in qsucc[gid]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    heapq.heappush(heap, (priority[target], target))
        if placed != len(grouping):
            return None
        return {
            "node_to_subgraph": {
                str(op_id): grouping.owner[op_id] for op_id in self.model.nodes
            },
            "core_schedules": schedules,
        }

    # -- cheap estimate (replaces the old full Step1/2/3 proxy) -------------

    def comm(self, source, target):
        return self.delay + 2.0 * self.gamma * self.model.edge_bytes[(source, target)] / self.bw

    def upward_rank(self, differ):
        model = self.model
        rank = {}
        for op_id in reversed(model.topological):
            best = 0.0
            for target in model.succ[op_id]:
                value = rank[target]
                if differ(op_id, target):
                    value += self.comm(op_id, target)
                if value > best:
                    best = value
            rank[op_id] = model.dur[op_id] + best
        return rank

    def estimate(self, core):
        """List schedule with fixed placement: pipes, input loads, latency."""
        model = self.model
        gamma, bw = self.gamma, self.bw
        rank = self.upward_rank(lambda a, b: core[a] != core[b])
        indegree = {op_id: len(model.pred[op_id]) for op_id in model.nodes}
        heap = [(-rank[op_id], op_id) for op_id in model.nodes if indegree[op_id] == 0]
        heapq.heapify(heap)
        pipe_free = defaultdict(float)
        mte2 = defaultdict(float)
        mte3 = defaultdict(float)
        loaded = {}
        finish, start = {}, {}
        end = 0.0
        while heap:
            _, op_id = heapq.heappop(heap)
            core_id = core[op_id]
            ready = 0.0
            for source in model.pred[op_id]:
                value = finish[source]
                if core[source] != core_id:
                    value += self.comm(source, op_id)
                if value > ready:
                    ready = value
            for tid, size in model.inputs[op_id]:
                key = (core_id, tid)
                if key not in loaded:
                    mte2[core_id] += gamma * size / bw
                    loaded[key] = mte2[core_id]
                ready = max(ready, loaded[key])
            key = (core_id, model.pipe[op_id])
            begin = max(ready, pipe_free[key])
            done = begin + model.dur[op_id]
            pipe_free[key] = done
            start[op_id], finish[op_id] = begin, done
            end = max(end, done)
            for tid, size in model.finals[op_id]:
                mte3[core_id] = max(mte3[core_id], done) + gamma * size / bw
                end = max(end, mte3[core_id])
            for target in model.succ[op_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    heapq.heappush(heap, (-rank[target], target))
        return end, start

    # -- official evaluation ------------------------------------------------

    def calibrate(self, result):
        actual = 0.0
        for core_entry in result.get("per_core_timeline", []):
            for entry in core_entry.get("ops", []):
                if entry.get("op") in EXCLUDED:
                    actual += float(entry["duration"])
        exclusive = result["data_movement_bytes"]["scheduled_copy_bytes"] / self.bw
        if exclusive > 0:
            self.gamma = min(max(1.0, actual / exclusive), max(1.0, float(self.K)))

    def evaluate(self, name, plan, grouping=None, group_core=None, family="native"):
        if plan is None:
            return None
        digest = hashlib.sha1(json.dumps(
            plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if digest in self.seen:
            return self.seen[digest]
        began = time.monotonic()
        try:
            result = evaluate_scene_b(self.graph, plan, **self.eval_kwargs)
        except Exception as error:
            self.seen[digest] = None
            self.records.append({"name": name, "error": repr(error)[:300]})
            return None
        elapsed = time.monotonic() - began
        self.eval_times.append(elapsed)
        movement = result["data_movement_bytes"]
        objective = (result["makespan"], movement["added_copy_bytes"])
        self.seen[digest] = objective
        self.records.append({
            "name": name, "makespan": objective[0],
            "added_copy_bytes": objective[1], "seconds": round(elapsed, 3),
        })
        if grouping is None:
            grouping, group_core = self.structure_from_plan(plan)
        self.pool.append((objective, name, grouping, tuple(group_core)))
        self.pool.sort(key=lambda item: (item[0], item[1]))
        del self.pool[12:]
        if self.best is None or objective < self.best.objective:
            self.best = Evaluated(name, plan, result, objective, grouping,
                                  tuple(group_core), family)
            self.calibrate(result)
            log(self.verbose, f"[best] {objective[0]} {name} "
                f"({elapsed:.2f}s, gamma={self.gamma:.2f})")
        return objective

    # -- P0: seed portfolio -------------------------------------------------

    def v8_seeds(self):
        if _V8 is None or self.K < 2:
            return []
        graph, K, n = self.graph, self.K, self.model.n
        seeds = []

        def pad(plan):
            plan["core_schedules"] = list(plan["core_schedules"]) + [
                [] for _ in range(K - len(plan["core_schedules"]))]
            return plan

        def attempt(name, builder):
            try:
                seeds.append((name, pad(builder())))
            except Exception as error:
                self.records.append({"name": name, "error": repr(error)[:300]})

        for active in range(K, 1, -1):
            try:
                plan, components = _V8.wcc_plan(graph, active)
                if components >= 2:
                    seeds.append((f"v8_wcc_active{active}", pad(plan)))
            except Exception as error:
                self.records.append({"name": "v8_wcc", "error": repr(error)[:300]})

        def level(mode, max_ops, work_factor, quantile, bins):
            op, nodes, _, succ, edge_bytes, clusters = _V8.chain_clusters(
                graph, K, max_ops, work_factor, mode, quantile)
            groups = _V8.level_pack(clusters, op, nodes, succ, edge_bytes, K, bins)
            return _V8.heft_plan(graph, groups, K)

        for setting in (("none", 80, .75, .05, 20), ("aggressive", 96, .9, .10, 30),
                        ("strong", 128, .75, .40, 12), ("aggressive", 160, .75, .15, 20),
                        ("strong", 256, 1.5, .50, 10), ("strong", 1024, 1.5, .50, 8)):
            attempt("v8_level_{}_{}_{}".format(setting[0], setting[1], setting[4]),
                    lambda s=setting: level(*s))
        attempt("v8_active3_strong96", lambda: _V8.v7_active3_strong96(graph, K))
        attempt("v8_micro64_b40", lambda: _V8.v7_micro64_b40(graph, K))
        for setting in ((64, .75, "aggressive", .1, .2), (128, 1.0, "strong", .4, .1)):
            attempt(f"v8_clusterphase_{setting[2]}_{setting[0]}",
                    lambda s=setting: _V8.clusterphase_plan(graph, K, *s, 1.3, 2))
        if n <= 7000:
            attempt("v8_frontier6", lambda: _V8.frontier_plan(graph, K, 6))
            for stick, cap in ((.1, 1.1), (.5, 1.2)):
                attempt(f"v8_corephase_{stick}_{cap}",
                        lambda s=stick, c=cap: _V8.corephase_plan(graph, K, s, c))
        return seeds

    def external_seeds(self):
        seeds = []
        for path in self.seed_plans:
            try:
                plan = json.loads(Path(path).read_text(encoding="utf-8"))
                schedules = [list(s) for s in plan["core_schedules"]]
                if any(schedules[k] for k in range(self.K, len(schedules))):
                    continue
                schedules = schedules[:self.K] + [
                    [] for _ in range(self.K - len(schedules))]
                seeds.append((f"seed:{Path(path).name}", {
                    "node_to_subgraph": plan["node_to_subgraph"],
                    "core_schedules": schedules,
                }))
            except Exception as error:
                self.records.append({"name": f"seed:{path}", "error": repr(error)[:300]})
        return seeds

    def phase_seeds(self, until):
        candidates = self.external_seeds() + self.v8_seeds()
        if self.K == 1 or self.model.n <= 3000 or not candidates:
            candidates.append(("single_core_dfs", self.realize(
                self.fine, (0,) * self.model.n, self.op_keys("dfs"))))
        ranked = []
        for index, (name, plan) in enumerate(candidates):
            if plan is None:
                continue
            try:
                grouping, group_core = self.structure_from_plan(plan)
                estimate = self.estimate(self.core_of(grouping, group_core))[0]
            except Exception as error:
                self.records.append({"name": name, "error": repr(error)[:300]})
                continue
            # External seeds first: they are usually already strong.
            ranked.append((0 if name.startswith("seed:") else 1, estimate, index,
                           name, plan, grouping, group_core))
        ranked.sort(key=lambda item: item[:3])
        best_estimate = min((item[1] for item in ranked), default=0.0)
        for _, estimate, _, name, plan, grouping, group_core in ranked:
            if self.best is not None and not self.affordable(until):
                break
            # Hopeless seeds (e.g. near single-core packings of a big graph)
            # can take minutes to evaluate; skip them once something exists.
            if (self.best is not None and not name.startswith("seed:")
                    and estimate > 2.5 * best_estimate):
                continue
            self.evaluate(name, plan, grouping, group_core, "native")

    def distinct_structures(self, limit):
        chosen, seen = [], set()
        for objective, name, grouping, group_core in self.pool:
            core = self.fine_core(self.core_of(grouping, group_core))
            if core in seen:
                continue
            seen.add(core)
            chosen.append((name, grouping, group_core))
            if len(chosen) >= limit:
                break
        return chosen

    def order_portfolio(self, name, grouping, group_core, families, until):
        core = self.core_of(grouping, group_core)
        fine_core = self.fine_core(core)
        _, start = self.estimate(core)
        for family in families:
            if not self.affordable(until):
                return
            keys = self.op_keys(family, start)
            self.evaluate(f"{name}|fine_{family}",
                          self.realize(self.fine, fine_core, keys),
                          self.fine, fine_core, family)
        if not grouping.is_fine and self.affordable(until):
            self.evaluate(f"{name}|coarse_time",
                          self.realize(grouping, group_core, self.op_keys("time", start)),
                          grouping, group_core, "time")

    # -- P2: slack-gated co-location clustering -----------------------------

    def critical_masks(self):
        model = self.model
        length = model.critical_length
        scale = MASK_BUCKETS / length
        threshold = 0.03 * length
        masks = {}
        for op_id in model.nodes:
            begin = model.est_end[op_id] - model.dur[op_id]
            slack = length - (begin + model.bottom[op_id])
            if slack > threshold:
                continue
            first = min(MASK_BUCKETS - 1, int(begin * scale))
            last = min(MASK_BUCKETS - 1, max(first, int((model.est_end[op_id] - 1e-9) * scale)))
            masks[op_id] = (model.pipe[op_id], ((1 << (last - first + 1)) - 1) << first)
        return masks

    def slack_clusters(self, tau, cap_factor, active):
        key = (tau, cap_factor, active)
        if key in self._cluster_cache:
            return self._cluster_cache[key]
        model = self.model
        length = model.critical_length
        ddr_weight = min(1.0, self.ddr_pressure)
        candidates = []
        for (source, target), size in model.edge_bytes.items():
            transfer = 2.0 * self.gamma * size / self.bw
            comm = self.delay + transfer
            slack = max(0.0, length - (model.est_end[source] + model.bottom[target]))
            exposed = max(0.0, comm - slack)
            ddr_gain = ddr_weight * transfer
            if exposed >= tau * comm or (ddr_weight >= 0.5 and transfer >= self.delay):
                candidates.append((exposed + ddr_gain, source, target))
        if ddr_weight >= 0.5:
            for tid, ops in model.input_consumers.items():
                size = int(model.tensor_by[tid].get("size", 0))
                gain = ddr_weight * self.gamma * size / self.bw
                if len(ops) < 2 or gain < 0.25 * self.delay:
                    continue
                ordered = sorted(ops, key=lambda op_id: (model.est_end[op_id], op_id))
                for first, second in zip(ordered, ordered[1:]):
                    candidates.append((gain, first, second))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

        parent = {op_id: op_id for op_id in model.nodes}
        work = {op_id: defaultdict(float, {model.pipe[op_id]: model.dur[op_id]})
                for op_id in model.nodes}
        masks = {op_id: defaultdict(int) for op_id in model.nodes}
        for op_id, (pipe, bits) in self.critical_masks().items():
            masks[op_id][pipe] = bits
        cap = cap_factor * max(model.pipe_total.values(), default=1.0) / max(1, active)

        def find(item):
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        for _, source, target in candidates:
            left, right = find(source), find(target)
            if left == right:
                continue
            pipes = set(work[left]) | set(work[right])
            if max(work[left][p] + work[right][p] for p in pipes) > cap:
                continue
            # SlackPack guard: never stack overlapping critical work on a pipe.
            if any(popcount(masks[left][p] & masks[right][p]) > 2
                   for p in set(masks[left]) & set(masks[right])):
                continue
            parent[right] = left
            for pipe, value in work.pop(right).items():
                work[left][pipe] += value
            for pipe, bits in masks.pop(right).items():
                masks[left][pipe] |= bits

        groups = defaultdict(list)
        for op_id in model.nodes:
            groups[find(op_id)].append(op_id)
        clusters = sorted(groups.values(), key=lambda group: min(model.dfs_pos[o] for o in group))
        owner = {op_id: cid for cid, group in enumerate(clusters) for op_id in group}
        cluster_masks = []
        for group in clusters:
            root = find(group[0])
            cluster_masks.append(dict(masks[root]))
        result = (owner, clusters, cluster_masks)
        self._cluster_cache[key] = result
        return result

    # -- P3: slack-aware cluster mapping ------------------------------------

    def map_clusters(self, owner, clusters, cluster_masks, active, spread_weight=1.0):
        model = self.model
        gamma, bw = self.gamma, self.bw
        bucket_length = model.critical_length / MASK_BUCKETS
        ddr_weight = min(1.0, self.ddr_pressure)
        rank = self.upward_rank(lambda a, b: owner[a] != owner[b])
        cluster_work = [defaultdict(float) for _ in clusters]
        adjacency = [defaultdict(float) for _ in clusters]
        for op_id in model.nodes:
            cluster_work[owner[op_id]][model.pipe[op_id]] += model.dur[op_id]
        for (source, target), size in model.edge_bytes.items():
            left, right = owner[source], owner[target]
            if left != right and size:
                adjacency[left][right] += size
                adjacency[right][left] += size

        cluster_core = [-1] * len(clusters)
        core_masks = [defaultdict(int) for _ in range(active)]
        core = {}
        indegree = {op_id: len(model.pred[op_id]) for op_id in model.nodes}
        heap = [(-rank[op_id], op_id) for op_id in model.nodes if indegree[op_id] == 0]
        heapq.heapify(heap)
        pipe_free = defaultdict(float)
        mte2 = defaultdict(float)
        loaded = {}
        finish = {}

        def ready_on(op_id, core_id):
            ready = 0.0
            for source in model.pred[op_id]:
                value = finish[source]
                if core[source] != core_id:
                    value += self.comm(source, op_id)
                ready = max(ready, value)
            for tid, size in model.inputs[op_id]:
                if (core_id, tid) in loaded:
                    ready = max(ready, loaded[(core_id, tid)])
                else:
                    ready = max(ready, mte2[core_id] + gamma * size / bw)
            return ready

        while heap:
            _, op_id = heapq.heappop(heap)
            cid = owner[op_id]
            pipe = model.pipe[op_id]
            if cluster_core[cid] < 0:
                best = None
                for core_id in range(active):
                    begin = max(ready_on(op_id, core_id), pipe_free[(core_id, pipe)])
                    eft = begin + model.dur[op_id]
                    projected = max(
                        pipe_free[(core_id, p)] + value
                        for p, value in cluster_work[cid].items())
                    overlap = sum(
                        popcount(core_masks[core_id][p] & bits)
                        for p, bits in cluster_masks[cid].items())
                    cut = sum(size for other, size in adjacency[cid].items()
                              if cluster_core[other] >= 0 and cluster_core[other] != core_id)
                    score = (max(eft, projected)
                             + spread_weight * overlap * bucket_length
                             + ddr_weight * 2.0 * gamma * cut / bw)
                    if best is None or (score, core_id) < best:
                        best = (score, core_id)
                cluster_core[cid] = best[1]
                for p, bits in cluster_masks[cid].items():
                    core_masks[best[1]][p] |= bits
            core_id = cluster_core[cid]
            core[op_id] = core_id
            for tid, size in model.inputs[op_id]:
                if (core_id, tid) not in loaded:
                    mte2[core_id] += gamma * size / bw
                    loaded[(core_id, tid)] = mte2[core_id]
            begin = max(ready_on(op_id, core_id), pipe_free[(core_id, pipe)])
            finish[op_id] = begin + model.dur[op_id]
            pipe_free[(core_id, pipe)] = finish[op_id]
            for target in model.succ[op_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    heapq.heappush(heap, (-rank[target], target))
        return core

    def phase_slack(self, until, families):
        actives = [self.K] + ([self.K - 1] if self.K >= 3 else [])
        for tau, cap in ((0.5, 1.0), (0.25, 0.6), (0.75, 1.5)):
            for active in actives:
                if not self.affordable(until):
                    return
                owner, clusters, masks = self.slack_clusters(tau, cap, active)
                core = self.map_clusters(owner, clusters, masks, active)
                fine_core = self.fine_core(core)
                _, start = self.estimate(core)
                for family in families:
                    if not self.affordable(until):
                        return
                    self.evaluate(
                        f"slack_t{tau}_c{cap}_a{active}|fine_{family}",
                        self.realize(self.fine, fine_core, self.op_keys(family, start)),
                        self.fine, fine_core, family)

    # -- P4: critical-path local search -------------------------------------

    def official_times(self, result):
        start, end, core = {}, {}, {}
        for core_entry in result.get("per_core_timeline", []):
            core_id = int(core_entry["core_id"])
            for entry in core_entry.get("ops", []):
                op_id = int(entry["op_id"])
                if op_id in self.model.node_set:
                    start[op_id] = float(entry["start"])
                    end[op_id] = float(entry["end"])
                    core[op_id] = core_id
        return start, end, core

    def critical_chain(self, result, limit=400):
        model = self.model
        start, end, core = self.official_times(result)
        if not end:
            return []
        previous = {}
        lanes = defaultdict(list)
        for op_id in start:
            lanes[(core[op_id], model.pipe[op_id])].append(op_id)
        for lane in lanes.values():
            lane.sort(key=lambda op_id: (start[op_id], op_id))
            for before, after in zip(lane, lane[1:]):
                previous[after] = before
        current = max(end, key=lambda op_id: (end[op_id], op_id))
        chain, seen = [], set()
        while current is not None and current not in seen and len(chain) < limit:
            seen.add(current)
            best = (-1.0, None, None)
            for source in model.pred[current]:
                arrival = end[source]
                kind = "data"
                if core[source] != core[current]:
                    arrival += self.comm(source, current)
                    kind = "cross"
                if arrival > best[0]:
                    best = (arrival, source, kind)
            before = previous.get(current)
            if before is not None and end[before] > best[0]:
                best = (end[before], before, "pipe")
            if best[1] is None:
                break
            chain.append((best[2], best[1], current))
            current = best[1]
        return chain

    def move_units(self, grouping):
        if not grouping.is_fine:
            return grouping.owner, grouping.members
        owner, clusters, _ = self.slack_clusters(0.5, 1.0, self.K)
        return owner, clusters

    def local_search(self, until):
        patience = self.patience or max(12, min(60, int(40 / max(0.05, self.avg_eval_time()))))
        failures = 0
        tried = set()
        while failures < patience and self.affordable(until):
            best = self.best
            grouping, group_core = best.grouping, list(best.group_core)
            core = self.core_of(grouping, group_core)
            official, _, _ = self.official_times(best.result)
            keys = self.op_keys("otime", official=official)
            unit_owner, units = self.move_units(grouping)

            moves = [("resort", None, None)]
            load = defaultdict(float)
            for op_id in self.model.nodes:
                load[(core[op_id], self.model.pipe[op_id])] += self.model.dur[op_id]
            for kind, source, target in self.critical_chain(best.result):
                if kind == "cross":
                    moves.append(("move", unit_owner[target], core[source]))
                    moves.append(("move", unit_owner[source], core[target]))
                elif kind == "pipe":
                    pipe = self.model.pipe[target]
                    others = sorted(
                        (k for k in range(self.K) if k != core[target]),
                        key=lambda k: (load[(k, pipe)], k))[:2]
                    for other in others:
                        moves.append(("move", unit_owner[target], other))
                        moves.append(("move", unit_owner[source], other))
            if best.family.startswith("win") or best.family in ("time", "dfs"):
                moves.append(("family", None, None))

            unique = []
            for move in moves:
                signature = (best.objective, move)
                if signature not in tried:
                    tried.add(signature)
                    unique.append(move)
                if len(unique) >= 16:
                    break
            if not unique:
                break

            scored = []
            for kind, unit, target_core in unique:
                if not self.affordable(until):
                    break
                if kind == "resort":
                    scored.append((-1.0, kind, grouping, tuple(group_core), None))
                    continue
                if kind == "family":
                    scored.append((-0.5, kind, grouping, tuple(group_core), None))
                    continue
                new_core = dict(core)
                for op_id in units[unit]:
                    new_core[op_id] = target_core
                if new_core == core:
                    continue
                estimate = self.estimate(new_core)[0]
                if grouping.is_fine or unit_owner is not grouping.owner:
                    new_grouping, new_group_core = self.fine, self.fine_core(new_core)
                else:
                    new_group_core = list(group_core)
                    new_group_core[unit] = target_core
                    new_grouping, new_group_core = grouping, tuple(new_group_core)
                scored.append((estimate, kind, new_grouping, new_group_core, (unit, target_core)))
            scored.sort(key=lambda item: item[0])

            improved = False
            for estimate, kind, new_grouping, new_group_core, info in scored[:4]:
                if not self.affordable(until):
                    break
                before = self.best.objective
                if kind == "family":
                    self.try_families(until)
                else:
                    plan = self.realize(new_grouping, new_group_core, keys)
                    self.evaluate(f"{best.name}+ls_{kind}" + (f"_u{info[0]}->c{info[1]}" if info else ""),
                                  plan, new_grouping, new_group_core, "otime")
                if self.best.objective < before:
                    self.accepted[kind] += 1
                    improved = True
                    break
                failures += 1
            if not improved and not scored:
                failures += 1

    def try_families(self, until):
        best = self.best
        family = best.family
        if family == "time":
            options = ["win256", "win128"]
        elif family == "dfs":
            options = ["win2", "win4"]
        else:
            buckets = int(family[3:])
            options = [f"win{max(1, buckets // 2)}", f"win{buckets * 2}"]
        core = self.core_of(best.grouping, best.group_core)
        fine_core = self.fine_core(core)
        _, start = self.estimate(core)
        for option in options:
            if not self.affordable(until):
                return
            self.evaluate(f"{best.name}~{option}",
                          self.realize(self.fine, fine_core, self.op_keys(option, start)),
                          self.fine, fine_core, option)

    # -- driver -------------------------------------------------------------

    def solve(self, single=None):
        began = time.monotonic()
        if single is None:
            single = evaluate_singlecore(
                self.graph, bandwidth=self.bandwidth, capacity=self.capacity)["makespan"]
        deadline = time.monotonic() + self.time_limit
        span = self.time_limit

        def mark(fraction):
            return min(deadline, time.monotonic() + fraction * span)

        self.phase_seeds(mark(0.30))
        log(self.verbose, f"[phase] seeds done: {self.best.objective if self.best else None}")
        families = ["time", "win64", "win16", "dfs", "win4"]
        order_until = mark(0.25)
        for name, grouping, group_core in self.distinct_structures(3):
            self.order_portfolio(name, grouping, group_core, families, order_until)
        log(self.verbose, f"[phase] orders done: {self.best.objective if self.best else None}")
        if self.K >= 2:
            best_family = self.best.family if self.best and self.best.family != "native" else "time"
            slack_families = list(dict.fromkeys([best_family, "time", "win64"]))
            self.phase_slack(mark(0.15), slack_families)
            log(self.verbose, f"[phase] slack done: {self.best.objective}")
            self.local_search(deadline)
            log(self.verbose, f"[phase] local search done: {self.best.objective}")

        if self.best is None:  # every candidate failed: guaranteed-valid fallback
            plan = self.realize(self.fine, (0,) * self.model.n, self.op_keys("dfs"))
            self.evaluate("fallback_single_core", plan, self.fine, (0,) * self.model.n, "dfs")
        if self.best is None:
            raise RuntimeError("no valid Problem-2 plan could be evaluated")

        best = self.best
        movement = best.result["data_movement_bytes"]
        used = sorted({k for k, schedule in enumerate(best.plan["core_schedules"]) if schedule})
        return {
            "single": single,
            "best_makespan": best.result["makespan"],
            "speedup": single / best.result["makespan"] if best.result["makespan"] else 0.0,
            "selected": best.name,
            "added_copy_bytes": movement["added_copy_bytes"],
            "data_movement_bytes": movement,
            "memory_peak_by_core": best.result["memory_peak_by_core"],
            "num_cores": self.K,
            "used_cores": len(used),
            "subgraphs": len(best.grouping),
            "official_evaluations": len(self.eval_times),
            "search_seconds": round(time.monotonic() - began, 2),
            "gamma": round(self.gamma, 3),
            "accepted_moves": dict(self.accepted),
            "plan": best.plan,
            "candidates": self.records,
        }


def resolve_config(graph_path, explicit):
    if explicit:
        return Path(explicit)
    beside_graph = Path(graph_path).resolve().parent / "config.txt"
    if beside_graph.exists():
        return beside_graph
    return ATTACHMENT / "data" / "config.txt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Scene-B multicore solver V3")
    parser.add_argument("graph")
    parser.add_argument("num_cores", nargs="?", type=int, default=5)
    parser.add_argument("output_plan", nargs="?")
    parser.add_argument("--config")
    parser.add_argument("--time-limit", type=float, default=300.0,
                        help="Search budget in seconds (single-core baseline excluded).")
    parser.add_argument("--single", type=float,
                        help="Known single-core makespan; skips the slow baseline run.")
    parser.add_argument("--seed-plan", action="append", default=[],
                        help="Extra seed plan JSON (repeatable), e.g. a Problem-1 plan.")
    parser.add_argument("--patience", type=int,
                        help="Local-search evaluations without improvement before stopping.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    graph_path = Path(args.graph)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    config = resolve_config(graph_path, args.config)
    solver = Solver(
        graph, args.num_cores, config,
        time_limit=args.time_limit,
        seed_plans=args.seed_plan,
        patience=args.patience,
        verbose=args.verbose,
    )
    single = None
    if args.single is not None:
        single = int(args.single) if float(args.single).is_integer() else args.single
    result = solver.solve(single=single)
    if args.output_plan:
        Path(args.output_plan).write_text(
            json.dumps(result["plan"], ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    printable = {k: v for k, v in result.items() if k not in {"plan", "candidates"}}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
