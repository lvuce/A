#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 1 / Scene A V14 cluster merge with measured-duration feedback.

Default: search from the input graph, then evaluate at most two V14 candidates.
No saved V10 results or environment variables are required.
New code is below the original V10 solve() function. No V12 search is included.

V8 combines:
1) DDR-aware WCC packing with adaptive active-core count;
2) communication-aware safe chain coarsening;
3) level packing with several granularities;
4) frontier growth for dense single-WCC graphs;
5) core/phase assignment on coarse graphs;
6) evaluator-calibrated exact-duration reassignment;
7) task-level simulated annealing for hard compact task DAGs;
8) extra edge-slack-aware cut candidates;
9) task-slack-guided simulated annealing as an additional refinement.

The final winner is always selected by the official Problem-1 evaluator.
This script expects the official attachment to be unzipped and its code directory
available on PYTHONPATH, or placed at /mnt/data/a/code as in the batch runner.
"""
import os, sys, json, math, random, heapq
from collections import defaultdict, deque

# V14: self-contained base search followed by cluster/feedback refinement.
import contextlib
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
_code_paths = [os.environ.get("HUAWEI_CODE_DIR")]
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    _code_paths.append(str(Path(sys.argv[1]).resolve().parent.parent / "code"))
_code_paths += [str(Path(HERE) / "通用神经网络处理器下的多核调度问题  附件" / "code"),
                "/mnt/data/a/code"]
for _path in reversed(_code_paths):
    if _path and os.path.isdir(_path):
        sys.path.insert(0, _path)

def _load_official():
    global evaluate_scene_a, evaluate_singlecore, build_singlecore_plan
    try:
        from multicore_cut_evaluate_problem_1 import evaluate_scene_a
        from singlecore_evaluate import evaluate_singlecore, build_singlecore_plan
    except ImportError as exc:
        raise RuntimeError("找不到官方评估器；请设置 HUAWEI_CODE_DIR 为附件/code 的绝对路径") from exc


BW = 60.0
EXCLUDED = {"COPY_IN", "COPY_OUT"}
_STATIC_SLACK_CACHE = {}


def eval_args():
    return dict(
        bandwidth=60,
        capacity={"L1": 524288, "UB": 131072},
        cross_core_wait=1000,
        same_core_wait=100,
    )


def build_views(g):
    op = {o["id"]: o for o in g["ops"]}
    op_ids = set(op)
    prod, cons = defaultdict(set), defaultdict(set)
    direct = []
    for e in g["edges"]:
        a, b = e["source"], e["target"]
        ai, bi = a in op_ids, b in op_ids
        if ai and not bi:
            prod[b].add(a)
        elif (not ai) and bi:
            cons[a].add(b)
        elif ai and bi and a != b:
            direct.append((a, b))

    nodes = [u for u, o in op.items() if o.get("op") not in EXCLUDED]
    eligible = set(nodes)

    succ0 = {u: set() for u in op_ids}
    for a, b in direct:
        succ0[a].add(b)
    for tid, ps in prod.items():
        for a in ps:
            for b in cons.get(tid, ()):
                if a != b:
                    succ0[a].add(b)

    pred = {u: set() for u in nodes}
    succ = {u: set() for u in nodes}
    for src in nodes:
        stack = list(succ0[src])
        seen = set()
        while stack:
            v = stack.pop()
            if v in eligible:
                if v != src:
                    succ[src].add(v)
                    pred[v].add(src)
            elif v not in seen:
                seen.add(v)
                stack.extend(succ0.get(v, ()))

    tensor_by = {t["id"]: t for t in g["tensors"]}
    edge_bytes = defaultdict(int)
    for tid, t in tensor_by.items():
        sz = int(t.get("size", 0))
        for a in prod.get(tid, ()):
            if a not in eligible:
                continue
            for b in cons.get(tid, ()):
                if b in eligible and a != b:
                    edge_bytes[(a, b)] += sz
    for u in nodes:
        for v in succ[u]:
            edge_bytes[(u, v)] += 0

    return op, nodes, pred, succ, edge_bytes, prod, cons, tensor_by


def topo(nodes, pred, succ):
    indeg = {u: len(pred[u]) for u in nodes}
    q = [u for u in nodes if indeg[u] == 0]
    heapq.heapify(q)
    out = []
    while q:
        u = heapq.heappop(q)
        out.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(q, v)
    if len(out) != len(nodes):
        raise RuntimeError("cycle in op DAG")
    return out


def static_edge_slack(g):
    """Compute cheap compute-only edge slack for cut heuristics only."""
    key=id(g)
    cached=_STATIC_SLACK_CACHE.get(key)
    if cached is not None and cached[0] is g:
        return cached[1]
    op, nodes, pred, succ, edge_bytes, *_ = build_views(g)
    order = topo(nodes, pred, succ)
    dur = {u: max(1.0, float(op[u].get("cycles", 0))) for u in nodes}
    ef = {}
    for u in order:
        ef[u] = dur[u] + max((ef[p] for p in pred[u]), default=0.0)
    cp = max(ef.values(), default=0.0)
    bottom = {}
    for u in reversed(order):
        bottom[u] = dur[u] + max((bottom[v] for v in succ[u]), default=0.0)
    latest_start = {u: cp - bottom[u] for u in nodes}
    edge_slack = {}
    for u in nodes:
        for v in succ[u]:
            edge_slack[(u, v)] = max(0.0, latest_start[v] - ef[u])
    _STATIC_SLACK_CACHE[key]=(g,edge_slack)
    return edge_slack


def cut_copy_work(bytes_):
    # Approximate COPY_OUT + COPY_IN work created by a subgraph boundary.
    # The official evaluator still determines true shared-DDR timing.
    return 2.0 * float(bytes_) / BW


def edge_cut_exposure(bytes_, slack):
    return max(0.0, cut_copy_work(bytes_) - float(slack))


def quotient(clusters, nodes, succ, edge_bytes):
    cid = {u: i for i, c in enumerate(clusters) for u in c}
    S = len(clusters)
    cp = [set() for _ in range(S)]
    cs = [set() for _ in range(S)]
    eb = defaultdict(int)
    for u in nodes:
        a = cid[u]
        for v in succ[u]:
            b = cid[v]
            if a != b:
                cs[a].add(b)
                cp[b].add(a)
                eb[(a, b)] += edge_bytes.get((u, v), 0)
    return cp, cs, eb


class DSU:
    def __init__(self, nodes):
        self.p = {x: x for x in nodes}
        self.sz = {x: 1 for x in nodes}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return a
        if self.sz[a] < self.sz[b]:
            a, b = b, a
        self.p[b] = a
        self.sz[a] += self.sz[b]
        return a


def chain_clusters(g, K, max_ops=256, work_factor=1.5, mode="none", strong_quantile=.5):
    op, nodes, pred, succ, edge_bytes, *_ = build_views(g)
    dsu = DSU(nodes)
    members = {u: [u] for u in nodes}
    work = {u: max(1, int(op[u].get("cycles", 0))) for u in nodes}
    total_work = sum(work.values())
    work_cap = max(500.0, work_factor * total_work / max(1, K))

    def can_merge(a, b):
        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb:
            return False
        return len(members[ra]) + len(members[rb]) <= max_ops and work[ra] + work[rb] <= work_cap

    def do_merge(a, b):
        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb or not can_merge(a, b):
            return False
        mm = members[ra] + members[rb]
        ww = work[ra] + work[rb]
        r = dsu.union(ra, rb)
        members[r] = mm
        work[r] = ww
        return True

    # Stage 1: exclusive serial chains.
    candidates = []
    for u in nodes:
        for v in succ[u]:
            if len(succ[u]) == 1 and len(pred[v]) == 1:
                candidates.append((edge_bytes.get((u, v), 0), u, v))
    for _, u, v in sorted(candidates, reverse=True):
        do_merge(u, v)

    roots = []
    seen = set()
    for u in nodes:
        r = dsu.find(u)
        if r not in seen:
            seen.add(r)
            roots.append(r)
    clusters = [members[r] for r in roots]

    if mode != "none":
        # Safe quotient merging only where source outdegree==1 or target indegree==1.
        for _ in range(30):
            cp, cs, eb = quotient(clusters, nodes, succ, edge_bytes)
            vals = sorted(x for x in eb.values() if x > 0)
            thr = vals[min(len(vals)-1, max(0, int(strong_quantile * (len(vals)-1))))] if vals else 10**30
            cwork = [sum(max(1, int(op[u].get("cycles", 0))) for u in c) for c in clusters]
            cand = []
            for (a, b), bytes_ in eb.items():
                if not (len(cs[a]) == 1 or len(cp[b]) == 1):
                    continue
                comm = 2.0 * bytes_ / BW
                comp = max(1.0, min(cwork[a], cwork[b]))
                ratio = comm / comp
                if mode == "strong":
                    ok = bytes_ >= thr and ratio >= .08
                else:  # aggressive
                    ok = bytes_ >= thr or ratio >= .22
                if ok:
                    cand.append((comm * (1.0 + ratio), a, b))
            if not cand:
                break
            used, pairs = set(), []
            for _, a, b in sorted(cand, reverse=True):
                if a in used or b in used:
                    continue
                if len(clusters[a]) + len(clusters[b]) <= max_ops and cwork[a] + cwork[b] <= work_cap:
                    used.update((a, b))
                    pairs.append((a, b))
            if not pairs:
                break
            mate = {}
            for a, b in pairs:
                mate[a] = b
                mate[b] = a
            new, done = [], set()
            for i, c in enumerate(clusters):
                if i in done:
                    continue
                if i in mate:
                    j = mate[i]
                    new.append(c + clusters[j])
                    done.update((i, j))
                else:
                    new.append(c)
                    done.add(i)
            clusters = new

    # Topologically relabel cluster DAG.
    cp, cs, eb = quotient(clusters, nodes, succ, edge_bytes)
    indeg = [len(x) for x in cp]
    q = [i for i, d in enumerate(indeg) if d == 0]
    heapq.heapify(q)
    order = []
    while q:
        a = heapq.heappop(q)
        order.append(a)
        for b in cs[a]:
            indeg[b] -= 1
            if indeg[b] == 0:
                heapq.heappush(q, b)
    if len(order) != len(clusters):
        raise RuntimeError("cluster quotient cycle")
    clusters = [clusters[i] for i in order]
    return op, nodes, pred, succ, edge_bytes, clusters


def slack_chain_clusters(g, K, max_ops=256, work_factor=1.5, mode="aggressive", strong_quantile=.5):
    """Additional slack-aware partition candidate; original V8 remains untouched."""
    op, nodes, pred, succ, edge_bytes, *_ = build_views(g)
    edge_slack = static_edge_slack(g)
    dsu = DSU(nodes)
    members = {u: [u] for u in nodes}
    work = {u: max(1, int(op[u].get("cycles", 0))) for u in nodes}
    total_work = sum(work.values())
    work_cap = max(500.0, work_factor * total_work / max(1, K))

    def can_merge(a, b):
        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb:
            return False
        return len(members[ra]) + len(members[rb]) <= max_ops and work[ra] + work[rb] <= work_cap

    def do_merge(a, b):
        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb or not can_merge(a, b):
            return False
        mm = members[ra] + members[rb]
        ww = work[ra] + work[rb]
        r = dsu.union(ra, rb)
        members[r] = mm
        work[r] = ww
        return True

    # Safe exclusive serial chains, ordered by least-hideable boundary cost.
    candidates = []
    for u in nodes:
        for v in succ[u]:
            if len(succ[u]) == 1 and len(pred[v]) == 1:
                b = edge_bytes.get((u, v), 0)
                exp = edge_cut_exposure(b, edge_slack.get((u, v), 0.0))
                candidates.append((exp, b, u, v))
    for _, _, u, v in sorted(candidates, reverse=True):
        do_merge(u, v)

    roots, seen = [], set()
    for u in nodes:
        r = dsu.find(u)
        if r not in seen:
            seen.add(r)
            roots.append(r)
    clusters = [members[r] for r in roots]

    for _ in range(30):
        cp, cs, eb = quotient(clusters, nodes, succ, edge_bytes)
        vals = sorted(x for x in eb.values() if x > 0)
        thr = vals[min(len(vals)-1, max(0, int(strong_quantile*(len(vals)-1))))] if vals else 10**30
        cwork = [sum(max(1, int(op[u].get("cycles", 0))) for u in c) for c in clusters]

        cid = {u: i for i, c in enumerate(clusters) for u in c}
        q_exp = defaultdict(float)
        q_copy = defaultdict(float)
        for u in nodes:
            a = cid[u]
            for v in succ[u]:
                b = cid[v]
                if a == b:
                    continue
                by = edge_bytes.get((u, v), 0)
                cw = cut_copy_work(by)
                q_copy[(a, b)] += cw
                q_exp[(a, b)] += max(0.0, cw - edge_slack.get((u, v), 0.0))

        cand = []
        for (a, b), bytes_ in eb.items():
            if not (len(cs[a]) == 1 or len(cp[b]) == 1):
                continue
            copy_work = q_copy.get((a, b), cut_copy_work(bytes_))
            exposed = q_exp.get((a, b), 0.0)
            comp = max(1.0, min(cwork[a], cwork[b]))
            ratio = exposed / comp
            frac = exposed / max(copy_work, 1e-9)
            if mode == "strong":
                ok = bytes_ >= thr and frac >= .20 and ratio >= .05
            else:
                ok = (bytes_ >= thr and frac >= .10) or ratio >= .20
            if ok:
                priority = exposed * (1.0 + ratio) + .10 * copy_work
                cand.append((priority, a, b))

        if not cand:
            break
        used, pairs = set(), []
        for _, a, b in sorted(cand, reverse=True):
            if a in used or b in used:
                continue
            if len(clusters[a]) + len(clusters[b]) <= max_ops and cwork[a] + cwork[b] <= work_cap:
                used.update((a, b))
                pairs.append((a, b))
        if not pairs:
            break
        mate = {}
        for a, b in pairs:
            mate[a] = b
            mate[b] = a
        new, done = [], set()
        for i, c in enumerate(clusters):
            if i in done:
                continue
            if i in mate:
                j = mate[i]
                new.append(c + clusters[j])
                done.update((i, j))
            else:
                new.append(c)
                done.add(i)
        clusters = new

    # Topological relabel.
    cp, cs, eb = quotient(clusters, nodes, succ, edge_bytes)
    indeg = [len(x) for x in cp]
    q = [i for i, d in enumerate(indeg) if d == 0]
    heapq.heapify(q)
    order = []
    while q:
        a = heapq.heappop(q)
        order.append(a)
        for b in cs[a]:
            indeg[b] -= 1
            if indeg[b] == 0:
                heapq.heappush(q, b)
    clusters = [clusters[i] for i in order]
    return op, nodes, pred, succ, edge_bytes, clusters


def level_pack(clusters, op, nodes, succ, edge_bytes, K, bins_per_level):
    cp, cs, eb = quotient(clusters, nodes, succ, edge_bytes)
    S = len(clusters)
    indeg = [len(x) for x in cp]
    q = [i for i, d in enumerate(indeg) if d == 0]
    heapq.heapify(q)
    level = [0] * S
    order = []
    while q:
        a = heapq.heappop(q)
        order.append(a)
        for b in cs[a]:
            level[b] = max(level[b], level[a] + 1)
            indeg[b] -= 1
            if indeg[b] == 0:
                heapq.heappush(q, b)
    M, V = [], []
    for c in clusters:
        M.append(sum(op[u].get("cycles", 0) for u in c if op[u].get("pipe") == "PIPE_M"))
        V.append(sum(op[u].get("cycles", 0) for u in c if op[u].get("pipe") == "PIPE_V"))
    groups = []
    for lv in range(max(level, default=-1) + 1):
        items = [i for i in range(S) if level[i] == lv]
        if not items:
            continue
        bcnt = min(len(items), max(1, bins_per_level))
        bins = [[] for _ in range(bcnt)]
        bm, bv = [0.0]*bcnt, [0.0]*bcnt
        items.sort(key=lambda i: max(M[i], V[i], .62*(M[i]+V[i])), reverse=True)
        for i in items:
            k = min(range(bcnt), key=lambda z: (max(bm[z]+M[i], bv[z]+V[i]), bm[z]+bv[z], z))
            bins[k].append(i)
            bm[k] += M[i]
            bv[k] += V[i]
        for b in bins:
            if b:
                groups.append(sum((clusters[i] for i in b), []))
    return groups


def heft_plan(g, clusters, K, local_durations=None):
    op, nodes, pred, succ, edge_bytes, *_ = build_views(g)
    cp, cs, eb = quotient(clusters, nodes, succ, edge_bytes)
    S = len(clusters)
    M, V = [], []
    for c in clusters:
        M.append(sum(op[u].get("cycles", 0) for u in c if op[u].get("pipe") == "PIPE_M"))
        V.append(sum(op[u].get("cycles", 0) for u in c if op[u].get("pipe") == "PIPE_V"))
    if local_durations is None:
        dur = [max(M[i], V[i], .62*(M[i]+V[i])) for i in range(S)]
    else:
        dur = list(map(float, local_durations))

    indeg = [len(x) for x in cp]
    q = [i for i, d in enumerate(indeg) if d == 0]
    heapq.heapify(q)
    top = []
    while q:
        a = heapq.heappop(q)
        top.append(a)
        for b in cs[a]:
            indeg[b] -= 1
            if indeg[b] == 0:
                heapq.heappush(q, b)
    rank = [0.0] * S
    for a in reversed(top):
        rank[a] = dur[a] + max((550.0 + rank[b] for b in cs[a]), default=0.0)

    indeg = [len(x) for x in cp]
    ready = {i for i, d in enumerate(indeg) if d == 0}
    av = [0.0] * K
    sched = [[] for _ in range(K)]
    ass, fin = {}, {}
    while ready:
        a = max(ready, key=lambda x: (rank[x], dur[x], -x))
        ready.remove(a)
        best = None
        for k in range(K):
            est = av[k] + (100 if sched[k] else 0)
            cross = 0
            for p in cp[a]:
                if ass[p] == k:
                    est = max(est, fin[p])
                else:
                    est = max(est, fin[p] + 1000)
                    cross += 1
            ft = est + dur[a]
            key = (ft, cross, av[k], k)
            if best is None or key < best[0]:
                best = (key, k, ft)
        _, k, ft = best
        ass[a] = k
        fin[a] = ft
        av[k] = ft
        sched[k].append(a)
        for b in cs[a]:
            indeg[b] -= 1
            if indeg[b] == 0:
                ready.add(b)
    return {
        "node_to_subgraph": {str(u): i for i, c in enumerate(clusters) for u in c},
        "core_schedules": sched,
    }


def wcc_plan(g, K):
    op, nodes, pred, succ, eb, prod, cons, tensor_by = build_views(g)
    seen, comps, comp_of = set(), [], {}
    for u in nodes:
        if u in seen:
            continue
        st, cc = [u], []
        seen.add(u)
        while st:
            x = st.pop()
            cc.append(x)
            for y in pred[x] | succ[x]:
                if y not in seen:
                    seen.add(y)
                    st.append(y)
        ci = len(comps)
        for x in cc:
            comp_of[x] = ci
        comps.append(cc)

    ddr = [0.0] * len(comps)
    for tid, t in tensor_by.items():
        sz = float(t.get("size", 0))
        has_in = any(op[p].get("op") == "COPY_IN" for p in prod.get(tid, ()) if p in op)
        has_out = any(op[c].get("op") == "COPY_OUT" for c in cons.get(tid, ()) if c in op)
        if has_in:
            for ci in {comp_of[c] for c in cons.get(tid, ()) if c in comp_of}:
                ddr[ci] += sz
        if has_out:
            for ci in {comp_of[p] for p in prod.get(tid, ()) if p in comp_of}:
                ddr[ci] += sz

    bcnt = min(K, len(comps))
    groups = [[] for _ in range(bcnt)]
    lm, lv, ld = [0.0]*bcnt, [0.0]*bcnt, [0.0]*bcnt
    stats = []
    for i, c in enumerate(comps):
        m = sum(op[u].get("cycles", 0) for u in c if op[u].get("pipe") == "PIPE_M")
        v = sum(op[u].get("cycles", 0) for u in c if op[u].get("pipe") == "PIPE_V")
        d = ddr[i] / BW
        stats.append((max(m, v, d), m, v, d, c))
    for _, m, v, d, c in sorted(stats, reverse=True, key=lambda x: x[0]):
        k = min(range(bcnt), key=lambda z: (max(lm[z]+m, lv[z]+v, ld[z]+d), lm[z]+lv[z]+ld[z], z))
        groups[k].extend(c)
        lm[k] += m
        lv[k] += v
        ld[k] += d
    mapping = {str(u): i for i, c in enumerate(groups) for u in c}
    sched = [[] for _ in range(K)]
    for i in range(len(groups)):
        sched[i].append(i)
    return {"node_to_subgraph": mapping, "core_schedules": sched}, len(comps)


def frontier_plan(g, K, waves=6):
    op, nodes, pred, succ, eb, *_ = build_views(g)
    order = topo(nodes, pred, succ)
    rank = {}
    for u in reversed(order):
        rank[u] = max(1, op[u].get("cycles", 0)) + max((rank[v] + 2*eb.get((u,v),0)/BW for v in succ[u]), default=0)
    total = sum(op[u].get("cycles", 0) for u in nodes)
    target = max(100.0, total / max(1, K*waves))
    left = {u: len(pred[u]) for u in nodes}
    ready = {u for u in nodes if left[u] == 0}
    assigned, clusters = set(), []
    while ready:
        seed = max(ready, key=lambda u: rank[u])
        ready.remove(seed)
        cur, curset = [], set()
        work = 0.0
        cand = {seed}
        while cand:
            def score(x):
                aff = sum(1000 + 2*eb.get((p,x),0)/BW for p in pred[x] if p in curset)
                return (aff + .02*rank[x], rank[x], -x)
            u = max(cand, key=score)
            cand.remove(u)
            if u in assigned:
                continue
            cyc = max(1, op[u].get("cycles", 0))
            affbytes = sum(eb.get((p,u),0) for p in pred[u] if p in curset)
            if cur and work + cyc > target and work > .35*target and affbytes < 4096:
                ready.add(u)
                break
            cur.append(u)
            curset.add(u)
            assigned.add(u)
            work += cyc
            for v in succ[u]:
                left[v] -= 1
                if left[v] == 0:
                    ready.add(v)
            for x in list(ready):
                if any(p in curset for p in pred[x]):
                    cand.add(x)
                    ready.discard(x)
        ready.update(x for x in cand if x not in assigned)
        if not cur:
            raise RuntimeError("empty frontier cluster")
        clusters.append(cur)
    return heft_plan(g, clusters, K)


def slack_frontier_plan(g, K, waves=6):
    """Frontier-growth partition using exposed cut cost; V8 HEFT is unchanged."""
    op, nodes, pred, succ, eb, *_ = build_views(g)
    edge_slack = static_edge_slack(g)
    order = topo(nodes, pred, succ)
    rank = {}
    for u in reversed(order):
        rank[u] = max(1, op[u].get("cycles", 0)) + max(
            (rank[v] + edge_cut_exposure(eb.get((u, v), 0), edge_slack.get((u, v), 0.0)) for v in succ[u]),
            default=0.0,
        )
    total = sum(op[u].get("cycles", 0) for u in nodes)
    target = max(100.0, total / max(1, K*waves))
    left = {u: len(pred[u]) for u in nodes}
    ready = {u for u in nodes if left[u] == 0}
    assigned, clusters = set(), []
    while ready:
        seed = max(ready, key=lambda u: rank[u])
        ready.remove(seed)
        cur, curset = [], set()
        work = 0.0
        cand = {seed}
        while cand:
            def score(x):
                aff = sum(
                    edge_cut_exposure(eb.get((p, x), 0), edge_slack.get((p, x), 0.0))
                    for p in pred[x] if p in curset
                )
                return (aff + .02*rank[x], rank[x], -x)
            u = max(cand, key=score)
            cand.remove(u)
            if u in assigned:
                continue
            cyc = max(1, op[u].get("cycles", 0))
            affbytes = sum(eb.get((p, u), 0) for p in pred[u] if p in curset)
            if cur and work + cyc > target and work > .35*target and affbytes < 4096:
                ready.add(u)
                break
            cur.append(u)
            curset.add(u)
            assigned.add(u)
            work += cyc
            for v in succ[u]:
                left[v] -= 1
                if left[v] == 0:
                    ready.add(v)
            for x in list(ready):
                if any(p in curset for p in pred[x]):
                    cand.add(x)
                    ready.discard(x)
        ready.update(x for x in cand if x not in assigned)
        if not cur:
            raise RuntimeError("empty slack frontier cluster")
        clusters.append(cur)
    return heft_plan(g, clusters, K)


def corephase_plan(g, K, stick=.2, cap=1.15):
    op, nodes, pred, succ, eb, *_ = build_views(g)
    order = topo(nodes, pred, succ)
    TM = sum(op[u].get("cycles",0) for u in nodes if op[u].get("pipe") == "PIPE_M")
    TV = sum(op[u].get("cycles",0) for u in nodes if op[u].get("pipe") == "PIPE_V")
    tM, tV = max(1.0, TM/K), max(1.0, TV/K)
    lm, lv = [0.0]*K, [0.0]*K
    ass = {}
    for u in order:
        cyc, pipe = float(op[u].get("cycles",0)), op[u].get("pipe")
        best = None
        for k in range(K):
            affinity = sum(1000 + 2*eb.get((p,u),0)/BW for p in pred[u] if ass.get(p) == k)
            nm = lm[k] + (cyc if pipe == "PIPE_M" else 0)
            nv = lv[k] + (cyc if pipe == "PIPE_V" else 0)
            ratio = max(nm/tM, nv/tV)
            over = max(0.0, ratio-cap)
            score = stick*affinity - 900*ratio - 5000*over
            key = (score, -ratio, -k)
            if best is None or key > best[0]:
                best = (key, k)
        k = best[1]
        ass[u] = k
        if pipe == "PIPE_M":
            lm[k] += cyc
        elif pipe == "PIPE_V":
            lv[k] += cyc
    phase = {}
    for u in order:
        phase[u] = max((phase[p] + (1 if ass[p] != ass[u] else 0) for p in pred[u]), default=0)
    groups = defaultdict(list)
    for u in order:
        groups[(phase[u], ass[u])].append(u)
    keys = sorted(groups)
    sid = {k:i for i,k in enumerate(keys)}
    sched = [[] for _ in range(K)]
    for key in keys:
        sched[key[1]].append(sid[key])
    return {
        "node_to_subgraph": {str(u): sid[(phase[u],ass[u])] for u in nodes},
        "core_schedules": sched,
    }


def clusterphase_plan(g, K, max_ops=128, work_factor=1.0, mode="strong", q=.4, stick=.1, cap=1.3, refine=2):
    op, nodes, pred, succ, eb, clusters = chain_clusters(g, K, max_ops, work_factor, mode, q)
    cp, cs, e2 = quotient(clusters, nodes, succ, eb)
    S = len(clusters)
    indeg = [len(x) for x in cp]
    hq = [i for i,d in enumerate(indeg) if d == 0]
    heapq.heapify(hq)
    order = []
    while hq:
        a = heapq.heappop(hq)
        order.append(a)
        for b in cs[a]:
            indeg[b] -= 1
            if indeg[b] == 0:
                heapq.heappush(hq,b)
    M, V = [], []
    for c in clusters:
        M.append(sum(op[u].get("cycles",0) for u in c if op[u].get("pipe") == "PIPE_M"))
        V.append(sum(op[u].get("cycles",0) for u in c if op[u].get("pipe") == "PIPE_V"))
    TM, TV = sum(M), sum(V)
    tM, tV = max(1.0, TM/K), max(1.0, TV/K)
    rank = [0.0]*S
    for a in reversed(order):
        local = max(M[a],V[a],.62*(M[a]+V[a]))
        rank[a] = local + max((1000 + 2*e2.get((a,b),0)/BW + rank[b] for b in cs[a]), default=0.0)
    lm, lv = [0.0]*K, [0.0]*K
    ass = {}
    indeg = [len(x) for x in cp]
    ready = {i for i,d in enumerate(indeg) if d == 0}
    seq = []
    while ready:
        a = max(ready, key=lambda i:(rank[i],max(M[i],V[i]),-i))
        ready.remove(a)
        seq.append(a)
        best = None
        for k in range(K):
            aff = sum(1000+2*e2.get((p,a),0)/BW for p in cp[a] if ass.get(p) == k)
            nm, nv = lm[k]+M[a], lv[k]+V[a]
            ratio = max(nm/tM,nv/tV)
            over = max(0.0,ratio-cap)
            score = stick*aff - 900*ratio - 5000*over
            key = (score,-ratio,-k)
            if best is None or key > best[0]:
                best=(key,k)
        k=best[1]
        ass[a]=k
        lm[k]+=M[a]
        lv[k]+=V[a]
        for b in cs[a]:
            indeg[b]-=1
            if indeg[b]==0:
                ready.add(b)
    for it in range(refine):
        changed=0
        walk=seq if it%2==0 else list(reversed(seq))
        for a in walk:
            old=ass[a]
            def affinity(k):
                z=0.0
                for b in cp[a]:
                    if ass[b]==k:z+=1000+2*e2.get((b,a),0)/BW
                for b in cs[a]:
                    if ass[b]==k:z+=1000+2*e2.get((a,b),0)/BW
                return z
            oldaff=affinity(old)
            choice=(0.0,old)
            for k in range(K):
                if k==old:continue
                nm,nv=lm[k]+M[a],lv[k]+V[a]
                if max(nm/tM,nv/tV)>cap:continue
                before=max(lm[old]/tM,lv[old]/tV)+max(lm[k]/tM,lv[k]/tV)
                after=max((lm[old]-M[a])/tM,(lv[old]-V[a])/tV)+max(nm/tM,nv/tV)
                gain=stick*(affinity(k)-oldaff)-500*(after-before)
                if gain>choice[0]:choice=(gain,k)
            if choice[1]!=old:
                k=choice[1]
                lm[old]-=M[a];lv[old]-=V[a]
                lm[k]+=M[a];lv[k]+=V[a]
                ass[a]=k;changed+=1
        if not changed:break
    phase={}
    for a in order:
        phase[a]=max((phase[p]+(1 if ass[p]!=ass[a] else 0) for p in cp[a]),default=0)
    groups=defaultdict(list)
    for a in order:
        groups[(phase[a],ass[a])].extend(clusters[a])
    keys=sorted(groups)
    sid={x:i for i,x in enumerate(keys)}
    sched=[[] for _ in range(K)]
    for x in keys:
        sched[x[1]].append(sid[x])
    return {
        "node_to_subgraph": {str(u):sid[(phase[a],ass[a])] for a in range(S) for u in clusters[a]},
        "core_schedules": sched,
    }


def exact_reassign(g, plan, res, K, alpha=1.0):
    sgs = sorted(set(int(v) for v in plan["node_to_subgraph"].values()))
    ren = {s:i for i,s in enumerate(sgs)}
    S = len(sgs)
    mapping = {str(k):ren[int(v)] for k,v in plan["node_to_subgraph"].items()}
    dur = [1.0]*S
    for k,x in res["step3_by_task"].items():
        kk = ren.get(int(k))
        if kk is not None:
            dur[kk]=float(x["local_makespan"])
    pred=[set() for _ in range(S)]
    succ=[set() for _ in range(S)]
    for e in res["task_dependencies"]:
        a,b=ren[int(e["source"])],ren[int(e["target"])]
        if a!=b:
            succ[a].add(b);pred[b].add(a)
    op,nodes,opred,osucc,eb,*_=build_views(g)
    nsg={int(k):int(v) for k,v in mapping.items()}
    e2=defaultdict(int)
    for u in nodes:
        a=nsg[u]
        for v in osucc[u]:
            b=nsg[v]
            if a!=b:e2[(a,b)]+=eb.get((u,v),0)
    indeg=[len(x) for x in pred]
    q=[i for i,d in enumerate(indeg) if d==0];heapq.heapify(q);top=[]
    while q:
        a=heapq.heappop(q);top.append(a)
        for b in succ[a]:
            indeg[b]-=1
            if indeg[b]==0:heapq.heappush(q,b)
    rank=[0.0]*S
    for a in reversed(top):
        rank[a]=dur[a]+max((1000+alpha*2*e2.get((a,b),0)/BW+rank[b] for b in succ[a]),default=0)
    indeg=[len(x) for x in pred]
    ready={i for i,d in enumerate(indeg) if d==0}
    av=[0.0]*K;sched=[[] for _ in range(K)];ass={};fin={}
    while ready:
        a=max(ready,key=lambda x:(rank[x],dur[x],-x));ready.remove(a)
        best=None
        for k in range(K):
            est=av[k]+(100 if sched[k] else 0);cross=0
            for p in pred[a]:
                if ass[p]==k:est=max(est,fin[p])
                else:est=max(est,fin[p]+1000);cross+=1
            ft=est+dur[a];key=(ft,cross,av[k],k)
            if best is None or key<best[0]:best=(key,k,ft)
        _,k,ft=best
        ass[a]=k;fin[a]=ft;av[k]=ft;sched[k].append(a)
        for b in succ[a]:
            indeg[b]-=1
            if indeg[b]==0:ready.add(b)
    return {"node_to_subgraph":mapping,"core_schedules":sched}


def v7_active3_strong96(g, K):
    active=min(3,K)
    op,nodes,pred,succ,eb,cl=chain_clusters(g,active,96,3.0,"strong",.25)
    groups=level_pack(cl,op,nodes,succ,eb,active,3)
    p=heft_plan(g,groups,active)
    p["core_schedules"] += [[] for _ in range(K-active)]
    return p


def v7_micro64_b40(g, K):
    op,nodes,pred,succ,eb,cl=chain_clusters(g,K,64,.4,"aggressive",0.0)
    groups=level_pack(cl,op,nodes,succ,eb,K,40)
    return heft_plan(g,groups,K)


def _task_extract(g, plan):
    res=evaluate_scene_a(g,plan,**eval_args())
    ids=sorted(int(k) for k in res["step3_by_task"].keys())
    S=max(ids)+1 if ids else 0
    dur=[1.0]*S
    for k,x in res["step3_by_task"].items():dur[int(k)]=float(x["local_makespan"])
    pred=[set() for _ in range(S)];succ=[set() for _ in range(S)]
    for e in res["task_dependencies"]:
        a,b=int(e["source"]),int(e["target"]);pred[b].add(a);succ[a].add(b)
    ass=[0]*S
    for k,ls in enumerate(plan["core_schedules"]):
        for t in ls:ass[int(t)]=k
    op,nodes,opred,osucc,eb,*_=build_views(g)
    nsg={int(k):int(v) for k,v in plan["node_to_subgraph"].items()};e2=defaultdict(int)
    for u in nodes:
        a=nsg[u]
        for v in osucc[u]:
            b=nsg[v]
            if a!=b:e2[(a,b)]+=eb.get((u,v),0)
    return res,dur,pred,succ,ass,e2


def _task_rank(dur,succ):
    S=len(dur);pred=[set() for _ in range(S)]
    for u in range(S):
        for v in succ[u]:pred[v].add(u)
    indeg=[len(x) for x in pred];q=[i for i,d in enumerate(indeg) if d==0];heapq.heapify(q);top=[]
    while q:
        u=heapq.heappop(q);top.append(u)
        for v in succ[u]:
            indeg[v]-=1
            if indeg[v]==0:heapq.heappush(q,v)
    rank=[0.0]*S
    for u in reversed(top):rank[u]=dur[u]+max((1000+rank[v] for v in succ[u]),default=0.0)
    return rank


def _task_schedule(dur,pred,succ,ass,K,rank):
    S=len(dur);indeg=[len(x) for x in pred];ready={i for i,d in enumerate(indeg) if d==0}
    av=[0.0]*K;first=[True]*K;fin=[0.0]*S;order=[[] for _ in range(K)]
    while ready:
        best=None
        for u in ready:
            k=ass[u];est=av[k]+(0 if first[k] else 100)
            for p in pred[u]:est=max(est,fin[p]+(0 if ass[p]==k else 1000))
            ft=est+dur[u];key=(ft,-rank[u],u)
            if best is None or key<best[0]:best=(key,u,k,ft)
        _,u,k,ft=best;ready.remove(u);fin[u]=ft;av[k]=ft;first[k]=False;order[k].append(u)
        for v in succ[u]:
            indeg[v]-=1
            if indeg[v]==0:ready.add(v)
    return max(fin,default=0.0),order


def _task_slack(dur, pred, succ):
    """Task-DAG slack from evaluator-calibrated local durations."""
    S = len(dur)
    indeg = [len(x) for x in pred]
    q = [i for i, d in enumerate(indeg) if d == 0]
    heapq.heapify(q)
    top = []
    while q:
        u = heapq.heappop(q)
        top.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(q, v)
    ef = [0.0] * S
    for u in top:
        ef[u] = dur[u] + max((ef[p] for p in pred[u]), default=0.0)
    cp = max(ef, default=0.0)
    bottom = [0.0] * S
    for u in reversed(top):
        bottom[u] = dur[u] + max((bottom[v] for v in succ[u]), default=0.0)
    return [max(0.0, cp - (ef[u] + bottom[u] - dur[u])) for u in range(S)]


def _weighted_pick(rng, weights):
    total = sum(weights)
    if total <= 0:
        return rng.randrange(len(weights))
    x = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if acc >= x:
            return i
    return len(weights) - 1


def task_sa_refine_guided(g, plan, K=5, iters=3000, seed=18801):
    """Extra SA candidate biased toward low-slack / critical Tasks."""
    base, dur, pred, succ, start, e2 = _task_extract(g, plan)
    S = len(dur)
    if S <= 1:
        return plan, base, "no_guided_refine"
    rank = _task_rank(dur, succ)
    slack = _task_slack(dur, pred, succ)
    positive = sorted(x for x in slack if x > 0)
    scale = positive[len(positive)//2] if positive else max(1.0, sum(dur)/max(1, S))
    scale = max(scale, 1.0)
    weights = [0.20 + 0.80/(1.0 + x/scale) for x in slack]
    rng = random.Random(seed)
    official = (base["makespan"], plan, "base")

    for beta in (0.0, .02, .1):
        ass = start[:]
        def score(a):
            ms, _ = _task_schedule(dur, pred, succ, a, K, rank)
            cut = sum(x for (u, v), x in e2.items() if a[u] != a[v])
            return ms + beta*cut/BW
        cur = score(ass)
        best = (cur, ass[:])
        elite = []
        T0 = max(20.0, .08*cur)
        for it in range(iters):
            T = T0 * (0.003 ** (it/max(1, iters-1)))
            if rng.random() < .80:
                u = _weighted_pick(rng, weights) if rng.random() < .82 else rng.randrange(S)
                old = ass[u]
                new = rng.randrange(K-1)
                new += new >= old
                ass[u] = new
                nv = score(ass)
                if nv < cur or rng.random() < math.exp(min(0.0, (cur-nv)/max(T, 1e-9))):
                    cur = nv
                else:
                    ass[u] = old
            else:
                if S < 2:
                    continue
                u = _weighted_pick(rng, weights) if rng.random() < .82 else rng.randrange(S)
                choices = [v for v in range(S) if v != u and ass[v] != ass[u]]
                if not choices:
                    continue
                if rng.random() < .75:
                    local_w = [weights[v] for v in choices]
                    v = choices[_weighted_pick(rng, local_w)]
                else:
                    v = rng.choice(choices)
                ass[u], ass[v] = ass[v], ass[u]
                nv = score(ass)
                if nv < cur or rng.random() < math.exp(min(0.0, (cur-nv)/max(T, 1e-9))):
                    cur = nv
                else:
                    ass[u], ass[v] = ass[v], ass[u]
            if cur < best[0]:
                best = (cur, ass[:])
            if it % 600 == 0:
                elite.append((cur, ass[:]))

        candidates = [best] + sorted(elite, key=lambda x: x[0])[:3]
        seen = set()
        for _, a in candidates:
            key = tuple(a)
            if key in seen:
                continue
            seen.add(key)
            _, order = _task_schedule(dur, pred, succ, a, K, rank)
            q = {"node_to_subgraph": plan["node_to_subgraph"], "core_schedules": order}
            try:
                r = evaluate_scene_a(g, q, **eval_args())
            except Exception:
                continue
            if r["makespan"] < official[0]:
                official = (r["makespan"], q, f"task_gsa_b{beta:g}")
    return official[1], evaluate_scene_a(g, official[1], **eval_args()), official[2]


def task_sa_refine(g, plan, K=5, iters=5000, seed=8801):
    base,dur,pred,succ,start,e2=_task_extract(g,plan)
    S=len(dur)
    if S<=1:
        return plan,base,"no_refine"
    rank=_task_rank(dur,succ);rng=random.Random(seed)
    official=(base["makespan"],plan,"base")
    for beta in (0.0,.02,.1,.5):
        ass=start[:]
        def score(a):
            ms,_=_task_schedule(dur,pred,succ,a,K,rank)
            cut=sum(x for (u,v),x in e2.items() if a[u]!=a[v])
            return ms+beta*cut/BW
        cur=score(ass);best=(cur,ass[:]);elite=[];T0=max(20.0,.08*cur)
        for it in range(iters):
            T=T0*(0.003**(it/max(1,iters-1)))
            if rng.random()<.78:
                u=rng.randrange(S);old=ass[u];new=rng.randrange(K-1);new+=new>=old;ass[u]=new
                nv=score(ass)
                if nv<cur or rng.random()<math.exp(min(0.0,(cur-nv)/max(T,1e-9))):cur=nv
                else:ass[u]=old
            else:
                if S<2:continue
                u,v=rng.sample(range(S),2)
                if ass[u]==ass[v]:continue
                ass[u],ass[v]=ass[v],ass[u];nv=score(ass)
                if nv<cur or rng.random()<math.exp(min(0.0,(cur-nv)/max(T,1e-9))):cur=nv
                else:ass[u],ass[v]=ass[v],ass[u]
            if cur<best[0]:best=(cur,ass[:])
            if it%800==0:elite.append((cur,ass[:]))
        candidates=[best]+sorted(elite,key=lambda x:x[0])[:4]
        seen=set()
        for _,a in candidates:
            key=tuple(a)
            if key in seen:continue
            seen.add(key)
            _,order=_task_schedule(dur,pred,succ,a,K,rank)
            q={"node_to_subgraph":plan["node_to_subgraph"],"core_schedules":order}
            try:r=evaluate_scene_a(g,q,**eval_args())
            except Exception:continue
            if r["makespan"]<official[0]:official=(r["makespan"],q,f"task_sa_b{beta:g}")
    return official[1],evaluate_scene_a(g,official[1],**eval_args()),official[2]



def _task_topology(pred, succ):
    S=len(pred)
    indeg=[len(x) for x in pred]
    q=[i for i,d in enumerate(indeg) if d==0]
    heapq.heapify(q)
    top=[]
    while q:
        u=heapq.heappop(q);top.append(u)
        for v in succ[u]:
            indeg[v]-=1
            if indeg[v]==0:heapq.heappush(q,v)
    return top


def exact_reassign_slackprio(g, plan, res, K, alpha=.1):
    """Round-1 candidate: use Task slack only to choose WHICH ready Task goes first.

    Core selection remains the original V8 model (same-core predecessor: 0 extra;
    cross-core predecessor: +1000; same-core task switch: +100).  This isolates the
    effect of Task criticality from V9.1's aggressive communication penalty.
    """
    sgs=sorted(set(int(v) for v in plan["node_to_subgraph"].values()))
    ren={sg:i for i,sg in enumerate(sgs)}
    S=len(sgs)
    mapping={str(k):ren[int(v)] for k,v in plan["node_to_subgraph"].items()}

    dur=[1.0]*S
    for k,x in res["step3_by_task"].items():
        kk=ren.get(int(k))
        if kk is not None:
            dur[kk]=float(x["local_makespan"])

    pred=[set() for _ in range(S)];succ=[set() for _ in range(S)]
    for e in res["task_dependencies"]:
        a,b=ren[int(e["source"])],ren[int(e["target"])]
        if a!=b:
            succ[a].add(b);pred[b].add(a)

    # Preserve V8's rank model used by exact_reassign.
    op,nodes,opred,osucc,eb,*_=build_views(g)
    nsg={int(k):int(v) for k,v in mapping.items()}
    e2=defaultdict(int)
    for u in nodes:
        a=nsg[u]
        for v in osucc[u]:
            b=nsg[v]
            if a!=b:e2[(a,b)]+=eb.get((u,v),0)

    top=_task_topology(pred,succ)
    rank=[0.0]*S
    for a in reversed(top):
        rank[a]=dur[a]+max((1000+alpha*2*e2.get((a,b),0)/BW+rank[b] for b in succ[a]),default=0.0)

    slack=_task_slack(dur,pred,succ)
    positive=sorted(x for x in slack if x>0)
    scale=positive[len(positive)//2] if positive else max(1.0,sum(dur)/max(1,S))
    scale=max(1.0,scale)
    criticality=[1.0/(1.0+x/scale) for x in slack]

    indeg=[len(x) for x in pred]
    ready={i for i,d in enumerate(indeg) if d==0}
    av=[0.0]*K;sched=[[] for _ in range(K)];ass={};fin={}
    while ready:
        # ONLY change relative ordering of ready Tasks.
        a=max(ready,key=lambda x:(criticality[x],rank[x],dur[x],-x));ready.remove(a)
        best=None
        for k in range(K):
            est=av[k]+(100 if sched[k] else 0);cross=0
            for p in pred[a]:
                if ass[p]==k:est=max(est,fin[p])
                else:est=max(est,fin[p]+1000);cross+=1
            ft=est+dur[a];key=(ft,cross,av[k],k)
            if best is None or key<best[0]:best=(key,k,ft)
        _,k,ft=best
        ass[a]=k;fin[a]=ft;av[k]=ft;sched[k].append(a)
        for b in succ[a]:
            indeg[b]-=1
            if indeg[b]==0:ready.add(b)
    return {"node_to_subgraph":mapping,"core_schedules":sched}


def _cluster_slack(clusters, op, nodes, succ, edge_bytes):
    """Compute cheap cluster-DAG slack using the same duration proxy as level_pack."""
    cp,cs,eb=quotient(clusters,nodes,succ,edge_bytes)
    S=len(clusters)
    M=[];V=[]
    for c in clusters:
        M.append(sum(op[u].get("cycles",0) for u in c if op[u].get("pipe")=="PIPE_M"))
        V.append(sum(op[u].get("cycles",0) for u in c if op[u].get("pipe")=="PIPE_V"))
    dur=[max(1.0,float(max(M[i],V[i],.62*(M[i]+V[i])))) for i in range(S)]
    top=_task_topology(cp,cs)
    ef=[0.0]*S
    for u in top:
        ef[u]=dur[u]+max((ef[p] for p in cp[u]),default=0.0)
    crit=max(ef,default=0.0)
    bottom=[0.0]*S
    for u in reversed(top):
        bottom[u]=dur[u]+max((bottom[v] for v in cs[u]),default=0.0)
    slack=[max(0.0,crit-(ef[u]+bottom[u]-dur[u])) for u in range(S)]
    return cp,cs,M,V,dur,slack


def level_pack_slack(clusters, op, nodes, succ, edge_bytes, K, bins_per_level):
    """Round-1 candidate: protect parallelism by spreading low-slack clusters.

    Original partition is unchanged.  Only the grouping of clusters at the same DAG
    level changes.  Critical clusters are spread across bins first, while high-slack
    clusters mostly act as load fillers.
    """
    cp,cs,M,V,dur,slack=_cluster_slack(clusters,op,nodes,succ,edge_bytes)
    S=len(clusters)
    indeg=[len(x) for x in cp]
    q=[i for i,d in enumerate(indeg) if d==0]
    heapq.heapify(q)
    level=[0]*S
    while q:
        a=heapq.heappop(q)
        for b in cs[a]:
            level[b]=max(level[b],level[a]+1)
            indeg[b]-=1
            if indeg[b]==0:heapq.heappush(q,b)

    positive=sorted(x for x in slack if x>0)
    scale=positive[len(positive)//2] if positive else max(1.0,sum(dur)/max(1,S))
    scale=max(scale,1.0)
    crit=[1.0/(1.0+x/scale) for x in slack]

    groups=[]
    for lv in range(max(level,default=-1)+1):
        items=[i for i in range(S) if level[i]==lv]
        if not items:continue
        bcnt=min(len(items),max(1,bins_per_level))
        bins=[[] for _ in range(bcnt)]
        bm=[0.0]*bcnt;bv=[0.0]*bcnt;bcrit=[0.0]*bcnt
        # Critical first, then large duration.
        items.sort(key=lambda i:(crit[i],dur[i]),reverse=True)
        for i in items:
            # Criticality balance is a soft first key; compute balance remains decisive
            # for near-equal criticality loads.
            k=min(range(bcnt),key=lambda z:(
                bcrit[z] + crit[i],
                max(bm[z]+M[i],bv[z]+V[i]),
                bm[z]+bv[z],
                z,
            ))
            bins[k].append(i);bm[k]+=M[i];bv[k]+=V[i];bcrit[k]+=crit[i]
        for b in bins:
            if b:groups.append(sum((clusters[i] for i in b),[]))
    return groups




def heft_plan_slackprio(g, clusters, K, gamma=.25, local_durations=None):
    """Round-2 experiment: Task/cluster slack affects only HEFT ready-task priority.

    Partition and core-cost model remain V8.  In particular, cross-core readiness is
    still +1000 only; DDR bytes are NOT added to EFT.  Slack is used as a modest
    priority boost so near-critical ready Tasks are considered earlier.
    """
    op,nodes,pred,succ,edge_bytes,*_=build_views(g)
    cp,cs,eb=quotient(clusters,nodes,succ,edge_bytes)
    S=len(clusters)
    M=[];V=[]
    for c in clusters:
        M.append(sum(op[u].get("cycles",0) for u in c if op[u].get("pipe")=="PIPE_M"))
        V.append(sum(op[u].get("cycles",0) for u in c if op[u].get("pipe")=="PIPE_V"))
    if local_durations is None:
        dur=[max(M[i],V[i],.62*(M[i]+V[i])) for i in range(S)]
    else:
        dur=list(map(float,local_durations))
    dur=[max(1.0,float(x)) for x in dur]

    top=_task_topology(cp,cs)
    rank=[0.0]*S
    for a in reversed(top):
        rank[a]=dur[a]+max((550.0+rank[b] for b in cs[a]),default=0.0)

    # Compute-only Task-DAG slack on the already-fixed partition.
    slack=_task_slack(dur,cp,cs)
    positive=sorted(x for x in slack if x>0)
    scale=positive[len(positive)//2] if positive else max(1.0,sum(dur)/max(1,S))
    scale=max(1.0,scale)
    crit=[1.0/(1.0+x/scale) for x in slack]
    priority=[rank[i]*(1.0+gamma*crit[i]) for i in range(S)]

    indeg=[len(x) for x in cp]
    ready={i for i,d in enumerate(indeg) if d==0}
    av=[0.0]*K;sched=[[] for _ in range(K)];ass={};fin={}
    while ready:
        a=max(ready,key=lambda x:(priority[x],rank[x],dur[x],-x));ready.remove(a)
        best=None
        for k in range(K):
            est=av[k]+(100 if sched[k] else 0);cross=0
            for pr in cp[a]:
                if ass[pr]==k:est=max(est,fin[pr])
                else:est=max(est,fin[pr]+1000);cross+=1
            ft=est+dur[a];key=(ft,cross,av[k],k)
            if best is None or key<best[0]:best=(key,k,ft)
        _,k,ft=best
        ass[a]=k;fin[a]=ft;av[k]=ft;sched[k].append(a)
        for b in cs[a]:
            indeg[b]-=1
            if indeg[b]==0:ready.add(b)
    return {
        "node_to_subgraph":{str(u):i for i,c in enumerate(clusters) for u in c},
        "core_schedules":sched,
    }


def slack_gate_chain_clusters(g, K, max_ops=160, work_factor=.75, mode="aggressive", strong_quantile=.15, gate=.25):
    """Round-1 candidate isolating Stage-1 slack gating.

    Unlike V10 slack_chain_clusters, an exclusive serial edge is NOT automatically
    merged.  It is merged only when at least `gate` of the boundary copy-work remains
    exposed after static edge slack.  Stage 2 intentionally reuses original V8 rules.
    """
    op,nodes,pred,succ,edge_bytes,*_=build_views(g)
    edge_slack=static_edge_slack(g)
    dsu=DSU(nodes)
    members={u:[u] for u in nodes}
    work={u:max(1,int(op[u].get("cycles",0))) for u in nodes}
    total_work=sum(work.values())
    work_cap=max(500.0,work_factor*total_work/max(1,K))

    def can_merge(a,b):
        ra,rb=dsu.find(a),dsu.find(b)
        return ra!=rb and len(members[ra])+len(members[rb])<=max_ops and work[ra]+work[rb]<=work_cap

    def do_merge(a,b):
        ra,rb=dsu.find(a),dsu.find(b)
        if ra==rb or not can_merge(a,b):return False
        mm=members[ra]+members[rb];ww=work[ra]+work[rb]
        r=dsu.union(ra,rb);members[r]=mm;work[r]=ww
        return True

    cand=[]
    for u in nodes:
        for v in succ[u]:
            if len(succ[u])==1 and len(pred[v])==1:
                by=edge_bytes.get((u,v),0)
                cw=cut_copy_work(by)
                ex=edge_cut_exposure(by,edge_slack.get((u,v),0.0))
                frac=ex/max(cw,1e-9) if cw>0 else 0.0
                if frac>=gate:
                    cand.append((ex,by,u,v))
    for _,_,u,v in sorted(cand,reverse=True):do_merge(u,v)

    roots=[];seen=set()
    for u in nodes:
        r=dsu.find(u)
        if r not in seen:seen.add(r);roots.append(r)
    clusters=[members[r] for r in roots]

    # Original V8 Stage-2 merge rules: this isolates the experiment to Stage 1.
    if mode!="none":
        for _ in range(30):
            cp,cs,eb=quotient(clusters,nodes,succ,edge_bytes)
            vals=sorted(x for x in eb.values() if x>0)
            thr=vals[min(len(vals)-1,max(0,int(strong_quantile*(len(vals)-1))))] if vals else 10**30
            cwork=[sum(max(1,int(op[u].get("cycles",0))) for u in c) for c in clusters]
            cand=[]
            for (a,b),bytes_ in eb.items():
                if not (len(cs[a])==1 or len(cp[b])==1):continue
                comm=2.0*bytes_/BW
                comp=max(1.0,min(cwork[a],cwork[b]));ratio=comm/comp
                if mode=="strong":ok=bytes_>=thr and ratio>=.08
                else:ok=bytes_>=thr or ratio>=.22
                if ok:cand.append((comm*(1.0+ratio),a,b))
            if not cand:break
            used=set();pairs=[]
            for _,a,b in sorted(cand,reverse=True):
                if a in used or b in used:continue
                if len(clusters[a])+len(clusters[b])<=max_ops and cwork[a]+cwork[b]<=work_cap:
                    used.update((a,b));pairs.append((a,b))
            if not pairs:break
            mate={}
            for a,b in pairs:mate[a]=b;mate[b]=a
            new=[];done=set()
            for i,c in enumerate(clusters):
                if i in done:continue
                if i in mate:
                    j=mate[i];new.append(c+clusters[j]);done.update((i,j))
                else:new.append(c);done.add(i)
            clusters=new

    cp,cs,eb=quotient(clusters,nodes,succ,edge_bytes)
    top=_task_topology(cp,cs)
    if len(top)!=len(clusters):raise RuntimeError("cluster quotient cycle")
    clusters=[clusters[i] for i in top]
    return op,nodes,pred,succ,edge_bytes,clusters



def slack_gate_chain_clusters_stage2(
    g, K, max_ops=160, work_factor=.75, mode="aggressive",
    strong_quantile=.15, gate1=.25, gate2=.25,
):
    """Round-3 experiment: apply slack as a merge/no-merge gate in BOTH stages.

    Stage 1 is the successful exclusive-serial-edge gate from Round 1/2.
    Stage 2 keeps the original V8 quotient-merge eligibility and communication
    heuristic, but additionally requires that enough boundary copy-work remains
    exposed after static op-edge slack.

    Important: slack is used only for the STRUCTURAL merge decision.  HEFT core
    placement remains the original V8 100/1000 model; DDR bytes are not added to
    cross-core EFT.
    """
    op,nodes,pred,succ,edge_bytes,*_=build_views(g)
    edge_slack=static_edge_slack(g)
    dsu=DSU(nodes)
    members={u:[u] for u in nodes}
    work={u:max(1,int(op[u].get("cycles",0))) for u in nodes}
    total_work=sum(work.values())
    work_cap=max(500.0,work_factor*total_work/max(1,K))

    def can_merge(a,b):
        ra,rb=dsu.find(a),dsu.find(b)
        return ra!=rb and len(members[ra])+len(members[rb])<=max_ops and work[ra]+work[rb]<=work_cap

    def do_merge(a,b):
        ra,rb=dsu.find(a),dsu.find(b)
        if ra==rb or not can_merge(a,b):
            return False
        mm=members[ra]+members[rb]
        ww=work[ra]+work[rb]
        r=dsu.union(ra,rb)
        members[r]=mm
        work[r]=ww
        return True

    # Stage 1: same semantics as the successful Round-1 gate.
    cand=[]
    for u in nodes:
        for v in succ[u]:
            if len(succ[u])==1 and len(pred[v])==1:
                by=edge_bytes.get((u,v),0)
                cw=cut_copy_work(by)
                ex=edge_cut_exposure(by,edge_slack.get((u,v),0.0))
                frac=ex/max(cw,1e-9) if cw>0 else 0.0
                if frac>=gate1:
                    cand.append((ex,by,u,v))
    for _,_,u,v in sorted(cand,reverse=True):
        do_merge(u,v)

    roots=[];seen=set()
    for u in nodes:
        r=dsu.find(u)
        if r not in seen:
            seen.add(r);roots.append(r)
    clusters=[members[r] for r in roots]

    if mode!="none":
        for _ in range(30):
            cp,cs,eb=quotient(clusters,nodes,succ,edge_bytes)
            vals=sorted(x for x in eb.values() if x>0)
            thr=vals[min(len(vals)-1,max(0,int(strong_quantile*(len(vals)-1))))] if vals else 10**30
            cwork=[sum(max(1,int(op[u].get("cycles",0))) for u in c) for c in clusters]

            # Aggregate copy-work and exposed copy-work across all op edges that
            # cross each current cluster boundary.  Summing per-edge exposure is
            # more faithful than subtracting one arbitrary slack from total bytes.
            cid={u:i for i,c in enumerate(clusters) for u in c}
            boundary_copy=defaultdict(float)
            boundary_exposed=defaultdict(float)
            for u in nodes:
                a=cid[u]
                for v in succ[u]:
                    b=cid[v]
                    if a==b:
                        continue
                    by=edge_bytes.get((u,v),0)
                    boundary_copy[(a,b)] += cut_copy_work(by)
                    boundary_exposed[(a,b)] += edge_cut_exposure(
                        by, edge_slack.get((u,v),0.0)
                    )

            cand=[]
            for (a,b),bytes_ in eb.items():
                # Preserve original V8 safe-topology constraint.
                if not (len(cs[a])==1 or len(cp[b])==1):
                    continue

                comm=2.0*bytes_/BW
                comp=max(1.0,min(cwork[a],cwork[b]))
                ratio=comm/comp
                if mode=="strong":
                    original_ok=bytes_>=thr and ratio>=.08
                else:
                    original_ok=bytes_>=thr or ratio>=.22
                if not original_ok:
                    continue

                cw=boundary_copy.get((a,b),0.0)
                ex=boundary_exposed.get((a,b),0.0)
                frac=ex/max(cw,1e-9) if cw>0 else 0.0
                if frac < gate2:
                    continue

                # Original V8 merge score, with exposed fraction only as a
                # tie-breaker.  Slack gates the merge; it does not redefine cost.
                cand.append((comm*(1.0+ratio),frac,ex,a,b))

            if not cand:
                break
            used=set();pairs=[]
            for _,_,_,a,b in sorted(cand,reverse=True):
                if a in used or b in used:
                    continue
                if len(clusters[a])+len(clusters[b])<=max_ops and cwork[a]+cwork[b]<=work_cap:
                    used.update((a,b));pairs.append((a,b))
            if not pairs:
                break

            mate={}
            for a,b in pairs:
                mate[a]=b;mate[b]=a
            new=[];done=set()
            for i,c in enumerate(clusters):
                if i in done:
                    continue
                if i in mate:
                    j=mate[i]
                    new.append(c+clusters[j])
                    done.update((i,j))
                else:
                    new.append(c)
                    done.add(i)
            clusters=new

    cp,cs,eb=quotient(clusters,nodes,succ,edge_bytes)
    top=_task_topology(cp,cs)
    if len(top)!=len(clusters):
        raise RuntimeError("cluster quotient cycle")
    clusters=[clusters[i] for i in top]
    return op,nodes,pred,succ,edge_bytes,clusters

def _critical_task_weights(dur,pred,succ):
    slack=_task_slack(dur,pred,succ)
    positive=sorted(x for x in slack if x>0)
    scale=positive[len(positive)//2] if positive else max(1.0,sum(dur)/max(1,len(dur)))
    scale=max(scale,1.0)
    crit=[1.0/(1.0+x/scale) for x in slack]
    return slack,crit


def task_sa_refine_edge_guided(g, plan, K=5, iters=1800, seed=31801):
    """Round-1 critical-edge SA.

    Keeps the partition fixed.  Low-slack Tasks are selected more often; target cores
    are biased toward predecessor/successor cores and the currently lightest core.
    """
    base,dur,pred,succ,start,e2=_task_extract(g,plan)
    S=len(dur)
    if S<=1:return plan,base,"no_edge_gsa"
    rank=_task_rank(dur,succ)
    slack,crit=_critical_task_weights(dur,pred,succ)
    weights=[.10+.90*c for c in crit]
    rng=random.Random(seed)
    official=(base["makespan"],plan,"base")

    for beta in (0.0,.02,.1):
        ass=start[:]
        def score(a):
            ms,_=_task_schedule(dur,pred,succ,a,K,rank)
            cut=sum(x for (u,v),x in e2.items() if a[u]!=a[v])
            return ms+beta*cut/BW
        cur=score(ass);best=(cur,ass[:]);elite=[];T0=max(20.0,.08*cur)
        for it in range(iters):
            T=T0*(0.003**(it/max(1,iters-1)))
            if rng.random()<.84:
                u=_weighted_pick(rng,weights) if rng.random()<.90 else rng.randrange(S)
                old=ass[u]
                # Meaningful target cores: critical neighbors + lightest static load.
                candidates={ass[p] for p in pred[u]}|{ass[v] for v in succ[u]}
                loads=[0.0]*K
                for t,k in enumerate(ass):loads[k]+=dur[t]
                candidates.add(min(range(K),key=lambda k:(loads[k],k)))
                candidates.discard(old)
                if not candidates:
                    candidates=set(range(K));candidates.discard(old)
                if not candidates:continue
                new=rng.choice(sorted(candidates))
                ass[u]=new;nv=score(ass)
                if nv<cur or rng.random()<math.exp(min(0.0,(cur-nv)/max(T,1e-9))):cur=nv
                else:ass[u]=old
            else:
                if S<2:continue
                u=_weighted_pick(rng,weights)
                near=list(pred[u]|succ[u])
                near=[v for v in near if ass[v]!=ass[u]]
                if near and rng.random()<.80:
                    v=max(near,key=lambda x:(crit[x],dur[x]))
                else:
                    choices=[v for v in range(S) if v!=u and ass[v]!=ass[u]]
                    if not choices:continue
                    v=rng.choice(choices)
                ass[u],ass[v]=ass[v],ass[u];nv=score(ass)
                if nv<cur or rng.random()<math.exp(min(0.0,(cur-nv)/max(T,1e-9))):cur=nv
                else:ass[u],ass[v]=ass[v],ass[u]
            if cur<best[0]:best=(cur,ass[:])
            if it%450==0:elite.append((cur,ass[:]))

        candidates=[best]+sorted(elite,key=lambda x:x[0])[:2]
        seen=set()
        for _,a in candidates:
            key=tuple(a)
            if key in seen:continue
            seen.add(key)
            _,order=_task_schedule(dur,pred,succ,a,K,rank)
            q={"node_to_subgraph":plan["node_to_subgraph"],"core_schedules":order}
            try:r=evaluate_scene_a(g,q,**eval_args())
            except Exception:continue
            if r["makespan"]<official[0]:official=(r["makespan"],q,f"r1_edgegsa_b{beta:g}")
    return official[1],evaluate_scene_a(g,official[1],**eval_args()),official[2]

def solve(g, K=5):
    ev=eval_args()
    single=evaluate_singlecore(g,**ev)["makespan"]
    base=build_singlecore_plan(g)
    base["core_schedules"] += [[] for _ in range(K-1)]
    best=[single,"single",base]
    records=[]
    eval_cache={}

    def plan_sig(plan):
        return json.dumps(plan,sort_keys=True,separators=(",",":"))

    def eval_cached(plan):
        sig=plan_sig(plan)
        if sig not in eval_cache:
            eval_cache[sig]=evaluate_scene_a(g,plan,**ev)
        return eval_cache[sig]

    def test(name, plan):
        nonlocal best
        try:r=eval_cached(plan)
        except Exception as e:
            records.append({"name":name,"error":repr(e)})
            return
        sp=single/r["makespan"]
        records.append({"name":name,"makespan":r["makespan"],"speedup":sp,"added_copy_bytes":r["data_movement_bytes"]["added_copy_bytes"]})
        if r["makespan"]<best[0]:best=[r["makespan"],name,plan]

    # Adaptive active-core WCC candidates.
    for active in range(2,K+1):
        p,_=wcc_plan(g,active);p["core_schedules"] += [[] for _ in range(K-active)]
        test(f"wcc_active{active}",p)

    # Multi-granularity level-packing portfolio.
    settings=[
        ("none",80,.75,.05,20),
        ("aggressive",96,.9,.10,30),
        ("strong",128,.75,.40,12),
        ("aggressive",160,.75,.15,20),
        ("strong",256,1.5,.50,10),
        ("strong",1024,1.5,.50,8),
    ]
    for mode,maxops,wf,q,bins in settings:
        try:
            op,nodes,pred,succ,eb,cl=chain_clusters(g,K,maxops,wf,mode,q)
            groups=level_pack(cl,op,nodes,succ,eb,K,bins)
            test(f"level_{mode}_{maxops}_{bins}",heft_plan(g,groups,K))
        except Exception:
            pass

    # V7 additions.
    try:test("v7_active3_strong96",v7_active3_strong96(g,K))
    except Exception:pass
    try:test("v7_micro64_b40",v7_micro64_b40(g,K))
    except Exception:pass

    # Coarse cluster-to-core mapping candidates.
    for mo,wf,mode,q,stick in [
        (64,.75,"aggressive",.1,.2),
        (96,.9,"aggressive",.1,.1),
        (128,1.0,"strong",.4,.1),
    ]:
        try:test(f"clusterphase_{mode}_{mo}_{stick}",clusterphase_plan(g,K,mo,wf,mode,q,stick,1.3,2))
        except Exception:pass

    op_count=sum(1 for o in g["ops"] if o.get("op") not in EXCLUDED)
    if op_count<=7000:
        try:test("frontier6",frontier_plan(g,K,6))
        except Exception:pass
        for stick,cap in [(.05,1.05),(.1,1.1),(.2,1.15),(.5,1.2),(1.0,1.3),(3.0,1.5)]:
            try:test(f"corephase_{stick}_{cap}",corephase_plan(g,K,stick,cap))
            except Exception:pass

    # Evaluator-calibrated reassignment of current best partition.
    if len(set(best[2]["node_to_subgraph"].values()))>1:
        try:
            seed_plan=best[2];seed_res=eval_cached(seed_plan)
            for alpha in (.1,1.0,2.0,3.0):
                q=exact_reassign(g,seed_plan,seed_res,K,alpha)
                test(f"reassign_{alpha:g}",q)
        except Exception:
            pass

    # Original V8 task-level SA, kept in the original position so the full V8
    # portfolio is preserved before any new slack candidate is allowed to affect best.
    speed=single/best[0]
    ntasks=len(set(best[2]["node_to_subgraph"].values()))
    if speed<2.65 and 4<=ntasks<=64:
        seed=(len(g.get("ops",[]))*1009 + len(g.get("edges",[]))*9173) & 0xffffffff
        try:
            q,rr,tag=task_sa_refine(g,best[2],K,iters=5000,seed=seed)
            records.append({"name":tag,"makespan":rr["makespan"],"speedup":single/rr["makespan"],"added_copy_bytes":rr["data_movement_bytes"]["added_copy_bytes"]})
            if rr["makespan"]<best[0]:best=[rr["makespan"],best[1]+"+"+tag,q]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # V10 additions begin here.  At this point the complete original V8 has
    # already run, so the extra candidates can only improve the final makespan.
    # ------------------------------------------------------------------
    slack_plans=[]

    # 1) Slack-aware versions of the two strongest V8 level configurations.
    for mode,maxops,wf,q,bins in [
        ("aggressive",160,.75,.15,20),
        ("strong",256,1.5,.50,10),
    ]:
        try:
            op,nodes,pred,succ,eb,cl=slack_chain_clusters(g,K,maxops,wf,mode,q)
            groups=level_pack(cl,op,nodes,succ,eb,K,bins)
            plan=heft_plan(g,groups,K)
            name=f"slack_level_{mode}_{maxops}_{bins}"
            test(name,plan)
            slack_plans.append((name,plan))
        except Exception:
            pass

    # 2) Slack-aware frontier is only tried on the same small/medium graphs on
    # which the original V8 frontier is enabled.
    if op_count<=7000:
        try:
            plan=slack_frontier_plan(g,K,6)
            test("slack_frontier6",plan)
            slack_plans.append(("slack_frontier6",plan))
        except Exception:
            pass

    # 3) The V8 statistics show alpha=0.1 is by far the strongest reassignment
    # setting.  Give each new slack partition one calibrated reassign_0.1 chance.
    for name,plan in slack_plans:
        try:
            if len(set(plan["node_to_subgraph"].values()))<=1:
                continue
            seed_res=eval_cached(plan)
            q=exact_reassign(g,plan,seed_res,K,.1)
            test(name+"+reassign_0.1",q)
        except Exception:
            pass

    # 4) Additional task-slack-guided SA.  Unlike edge slack, this uses the
    # evaluator-calibrated Task DAG and only biases which Tasks are moved/swapped.
    # It does not alter the objective and cannot replace the original V8 SA result.
    speed=single/best[0]
    ntasks=len(set(best[2]["node_to_subgraph"].values()))
    if speed<3.2 and 4<=ntasks<=64:
        seed=(len(g.get("ops",[]))*1009 + len(g.get("edges",[]))*9173) & 0xffffffff
        try:
            gseed=(seed ^ 0x9E3779B9) & 0xffffffff
            q,rr,tag=task_sa_refine_guided(g,best[2],K,iters=3000,seed=gseed)
            records.append({"name":tag,"makespan":rr["makespan"],"speedup":single/rr["makespan"],"added_copy_bytes":rr["data_movement_bytes"]["added_copy_bytes"]})
            if rr["makespan"]<best[0]:best=[rr["makespan"],best[1]+"+"+tag,q]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Round-1 Slack placement experiments.
    # Each candidate isolates one use of slack so the results are interpretable.
    # ------------------------------------------------------------------

    # R1-A: Task-level slack priority in evaluator-calibrated reassign.
    try:
        seed_plan=best[2]
        if len(set(seed_plan["node_to_subgraph"].values()))>1:
            seed_res=eval_cached(seed_plan)
            q=exact_reassign_slackprio(g,seed_plan,seed_res,K,.1)
            test("r1_reassign_slackprio_0.1",q)
    except Exception:
        pass

    # R1-B/C: keep original V8 partition, change only same-level packing.
    for mode,maxops,wf,qv,bins in [
        ("aggressive",160,.75,.15,20),
        ("strong",256,1.5,.50,10),
    ]:
        try:
            op,nodes,pred,succ,eb,cl=chain_clusters(g,K,maxops,wf,mode,qv)
            groups=level_pack_slack(cl,op,nodes,succ,eb,K,bins)
            test(f"r1_slackpack_{mode}_{maxops}_{bins}",heft_plan(g,groups,K))
        except Exception:
            pass

    # R1-D: isolate slack as a hard Stage-1 merge gate.
    try:
        op,nodes,pred,succ,eb,cl=slack_gate_chain_clusters(g,K,160,.75,"aggressive",.15,.25)
        groups=level_pack(cl,op,nodes,succ,eb,K,20)
        test("r1_slackgate_aggressive_160_20",heft_plan(g,groups,K))
    except Exception:
        pass

    # R1-E: critical-edge guided SA, only for compact difficult Task DAGs.
    speed=single/best[0]
    ntasks=len(set(best[2]["node_to_subgraph"].values()))
    if speed<3.2 and 4<=ntasks<=64:
        seed=(len(g.get("ops",[]))*1009 + len(g.get("edges",[]))*9173) & 0xffffffff
        try:
            eseed=(seed ^ 0x85EBCA6B) & 0xffffffff
            q,rr,tag=task_sa_refine_edge_guided(g,best[2],K,iters=1800,seed=eseed)
            records.append({"name":tag,"makespan":rr["makespan"],"speedup":single/rr["makespan"],"added_copy_bytes":rr["data_movement_bytes"]["added_copy_bytes"]})
            if rr["makespan"]<best[0]:best=[rr["makespan"],best[1]+"+"+tag,q]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Round-2 Slack experiments -- compare every candidate directly against V8.
    # R2 deliberately keeps the number of new variants small and tests:
    #   (1) Stage-1 gate threshold,
    #   (2) interaction between gate and packing,
    #   (3) whether a good slack-gated partition benefits from V8 reassign,
    #   (4) whether Task slack helps on that partition,
    #   (5) Slack used only as HEFT ready-task priority.
    # ------------------------------------------------------------------

    # R2-A/B: bracket the successful Stage-1 gate=.25 with more/less merging.
    for gate in (.10,.40):
        try:
            op,nodes,pred,succ,eb,cl=slack_gate_chain_clusters(
                g,K,160,.75,"aggressive",.15,gate
            )
            groups=level_pack(cl,op,nodes,succ,eb,K,20)
            test(f"r2_slackgate_g{int(round(gate*100)):02d}_aggressive_160_20",heft_plan(g,groups,K))
        except Exception:
            pass

    # R2-C: combine the two Round-1 winners: Stage-1 gate + slack-aware level packing.
    try:
        op,nodes,pred,succ,eb,cl=slack_gate_chain_clusters(
            g,K,160,.75,"aggressive",.15,.25
        )
        groups=level_pack_slack(cl,op,nodes,succ,eb,K,20)
        test("r2_gate025_plus_slackpack_aggressive_160_20",heft_plan(g,groups,K))
    except Exception:
        pass

    # R2-D/E: keep the successful gate partition and test placement refinement.
    # Normal V8 reassign is the control; slack-priority reassign differs only in
    # ready-Task ordering, so their difference isolates Task-level slack.
    try:
        op,nodes,pred,succ,eb,cl=slack_gate_chain_clusters(
            g,K,160,.75,"aggressive",.15,.25
        )
        groups=level_pack(cl,op,nodes,succ,eb,K,20)
        gate_plan=heft_plan(g,groups,K)
        gate_res=eval_cached(gate_plan)
        q=exact_reassign(g,gate_plan,gate_res,K,.1)
        test("r2_gate025_plus_reassign_0.1",q)
        q2=exact_reassign_slackprio(g,gate_plan,gate_res,K,.1)
        test("r2_gate025_plus_reassign_slackprio_0.1",q2)
    except Exception:
        pass

    # R2-F: new location -- use Task/cluster slack only in HEFT ready ordering.
    # Partition and core assignment cost are otherwise exactly V8-style.
    try:
        op,nodes,pred,succ,eb,cl=chain_clusters(g,K,160,.75,"aggressive",.15)
        groups=level_pack(cl,op,nodes,succ,eb,K,20)
        test("r2_heft_slackprio_g025_aggressive_160_20",heft_plan_slackprio(g,groups,K,.25))
    except Exception:
        pass


    # ------------------------------------------------------------------
    # Round-3 Slack validation.
    # Round-2 showed that almost all useful slack gain comes from structural
    # decisions: Stage-1 gate and level packing.  R3 therefore narrows the
    # search to four questions:
    #   A) Do Gate + SlackPack + original V8 Reassign stack?
    #   B) Is the successful gate=.25 locally better at .20 or .30?
    #   C) Should slack also gate Stage-2 quotient merges?
    #   D) If Stage-2 gating helps, does original V8 Reassign add more?
    # No R3 candidate changes V8's core-cost semantics.
    # ------------------------------------------------------------------

    # R3-A: strongest composition suggested by Round-2:
    # Stage-1 Gate(.25) -> SlackPack -> ORIGINAL V8 exact_reassign(.1).
    try:
        op,nodes,pred,succ,eb,cl=slack_gate_chain_clusters(
            g,K,160,.75,"aggressive",.15,.25
        )
        groups=level_pack_slack(cl,op,nodes,succ,eb,K,20)
        plan=heft_plan(g,groups,K)
        base_res=eval_cached(plan)
        q=exact_reassign(g,plan,base_res,K,.1)
        test("r3_gate025_slackpack_reassign01",q)
    except Exception:
        pass

    # R3-B/C: local threshold search around the successful .25 gate.  Keep the
    # rest identical so .20/.25/.30 is a clean ablation.
    for gate in (.20,.30):
        try:
            op,nodes,pred,succ,eb,cl=slack_gate_chain_clusters(
                g,K,160,.75,"aggressive",.15,gate
            )
            groups=level_pack_slack(cl,op,nodes,succ,eb,K,20)
            plan=heft_plan(g,groups,K)
            base_res=eval_cached(plan)
            q=exact_reassign(g,plan,base_res,K,.1)
            test(f"r3_gate{int(round(gate*100)):03d}_slackpack_reassign01",q)
        except Exception:
            pass

    # R3-D: use slack as a hard merge/no-merge gate in both clustering stages,
    # then use the successful slack-aware packing.  This directly tests whether
    # the structural interpretation of slack extends beyond exclusive chains.
    try:
        op,nodes,pred,succ,eb,cl=slack_gate_chain_clusters_stage2(
            g,K,160,.75,"aggressive",.15,.25,.25
        )
        groups=level_pack_slack(cl,op,nodes,succ,eb,K,20)
        plan=heft_plan(g,groups,K)
        test("r3_gate12_025_slackpack",plan)

        # R3-E: same partition/grouping, followed only by ORIGINAL V8 reassign.
        base_res=eval_cached(plan)
        q=exact_reassign(g,plan,base_res,K,.1)
        test("r3_gate12_025_slackpack_reassign01",q)
    except Exception:
        pass


    # ------------------------------------------------------------------
    # Final round: exploit the strongest structure found in R1-R3.
    #
    # Evidence so far points to:
    #   Stage-1 SlackGate(.25) -> SlackPack -> ORIGINAL V8 Reassign
    # as the strongest slack pipeline.  This final round does NOT search for
    # new slack locations.  It only asks three practical questions before we
    # freeze the final portfolio:
    #   1) Which V8 reassign alpha is best on this slack-structured partition?
    #   2) Is SlackPack slightly too coarse/fine at bins=20?
    #   3) Can a compact difficult Task DAG gain one last improvement from
    #      task-level guided SA after the best structural candidate?
    # No candidate changes the official cost model or adds bytes to cross-core EFT.
    # ------------------------------------------------------------------

    r4_struct_best = None   # (makespan, name, plan)

    def r4_track(name, plan):
        """Evaluate through the shared cache, record through test(), and keep
        the best *R4 structural* seed for the final guided-SA probe."""
        nonlocal r4_struct_best
        test(name, plan)
        try:
            rr = eval_cached(plan)
        except Exception:
            return
        item = (rr["makespan"], name, plan)
        if r4_struct_best is None or item[0] < r4_struct_best[0]:
            r4_struct_best = item

    # Build the proven structural seed once:
    # Gate(.25) -> SlackPack(bins=20) -> V8 HEFT.
    try:
        op,nodes,pred,succ,eb,cl = slack_gate_chain_clusters(
            g,K,160,.75,"aggressive",.15,.25
        )
        groups20 = level_pack_slack(cl,op,nodes,succ,eb,K,20)
        seed20 = heft_plan(g,groups20,K)
        seed20_res = eval_cached(seed20)

        # R4-A: reassign-alpha ablation.  alpha=.1 is the proven control;
        # alpha=0 isolates pure duration/topology rank, while 1 and 3 recover
        # the stronger communication-ranked variants that occasionally won in V8.
        for alpha, tag in [
            (0.0, "a0"),
            (0.1, "a01"),
            (1.0, "a1"),
            (3.0, "a3"),
        ]:
            q = exact_reassign(g, seed20, seed20_res, K, alpha)
            r4_track(f"r4_gate025_pack20_reassign_{tag}", q)
    except Exception:
        pass

    # R4-B/C: packing-granularity ablation around the successful bins=20.
    # bins=12 keeps more local reuse / fewer Task boundaries;
    # bins=30 preserves more same-level parallelism.  Reassign stays at the
    # proven alpha=.1 so this isolates only packing granularity.
    for bins in (12, 30):
        try:
            op,nodes,pred,succ,eb,cl = slack_gate_chain_clusters(
                g,K,160,.75,"aggressive",.15,.25
            )
            groups = level_pack_slack(cl,op,nodes,succ,eb,K,bins)
            p = heft_plan(g,groups,K)
            rr = eval_cached(p)
            q = exact_reassign(g,p,rr,K,.1)
            r4_track(f"r4_gate025_pack{bins}_reassign_a01", q)
        except Exception:
            pass

    # R4-D/E: one last local-search probe, but ONLY on compact difficult Task
    # DAGs and ONLY from the best structural seed found above.  This avoids
    # spending SA budget on large/high-speedup cases where it has historically
    # contributed little.
    if r4_struct_best is not None:
        r4_ms, r4_name, r4_plan = r4_struct_best
        r4_speed = single / r4_ms
        r4_ntasks = len(set(r4_plan["node_to_subgraph"].values()))
        r4_seed = (len(g.get("ops",[]))*1009 + len(g.get("edges",[]))*9173 + 0x51ED270B) & 0xffffffff

        # Critical-edge GSA: broader gate, still cheap enough for <=64 Tasks.
        if r4_speed < 4.0 and 4 <= r4_ntasks <= 64:
            try:
                q, rr, tag = task_sa_refine_edge_guided(
                    g, r4_plan, K, iters=2200, seed=(r4_seed ^ 0x85EBCA6B) & 0xffffffff
                )
                name = f"r4_struct_edgegsa_from_{r4_name}+{tag}"
                records.append({
                    "name": name,
                    "makespan": rr["makespan"],
                    "speedup": single/rr["makespan"],
                    "added_copy_bytes": rr["data_movement_bytes"]["added_copy_bytes"],
                })
                if rr["makespan"] < best[0]:
                    best = [rr["makespan"], name, q]
            except Exception:
                pass

        # The simpler task-slack GSA explores a different neighborhood.  Restrict
        # it further to genuinely hard/small cases so final-round runtime stays
        # controlled.
        if r4_speed < 3.5 and 4 <= r4_ntasks <= 48:
            try:
                q, rr, tag = task_sa_refine_guided(
                    g, r4_plan, K, iters=2600, seed=(r4_seed ^ 0x9E3779B9) & 0xffffffff
                )
                name = f"r4_struct_taskgsa_from_{r4_name}+{tag}"
                records.append({
                    "name": name,
                    "makespan": rr["makespan"],
                    "speedup": single/rr["makespan"],
                    "added_copy_bytes": rr["data_movement_bytes"]["added_copy_bytes"],
                })
                if rr["makespan"] < best[0]:
                    best = [rr["makespan"], name, q]
            except Exception:
                pass

    # Make candidate logs immediately useful for ablation analysis.
    for rec in records:
        if "makespan" in rec:
            rec["gap_to_best_pct"]=(float(rec["makespan"])/float(best[0])-1.0)*100.0

    return {
        "single":single,
        "best_makespan":best[0],
        "speedup":single/best[0],
        "selected":best[1],
        "plan":best[2],
        "candidates":records,
    }



# ==================== V13 bounded cluster probe ====================
# Traffic counts are tensor/task incidences, not sums over operation edges.
# Duration, liveness, and parallelism costs are proxies, never official scores.

def _v13_topology(pred, succ):
    order = _task_topology(pred, succ)
    if len(order) != len(pred):
        raise ValueError("cluster contraction or core order creates a cycle")
    return order


def _v13_schedule(dur, pred, succ, K, fixed=None):
    """Linear DAG passes plus heap-ready list scheduling; no DDR simulation."""
    n = len(dur)
    pp, ss = [set(p) for p in pred], [set(s) for s in succ]
    if fixed is not None:
        flat = [u for row in fixed for u in row]
        if len(flat) != n or set(flat) != set(range(n)):
            raise ValueError("seed core schedules do not cover tasks exactly once")
        owner = {u: k for k, row in enumerate(fixed) for u in row}
        for row in fixed:
            for a, b in zip(row, row[1:]):
                pp[b].add(a)
                ss[a].add(b)
        order = _v13_topology(pp, ss)
        start, finish = [0.0]*n, [0.0]*n
        prev = {b: a for row in fixed for a, b in zip(row, row[1:])}
        for u in order:
            start[u] = max((finish[p] + (1000 if owner[p] != owner[u] else 0)
                            for p in pred[u]), default=0.0)
            if u in prev:
                start[u] = max(start[u], finish[prev[u]] + 100)
            finish[u] = start[u] + dur[u]
        return fixed, start, finish, owner
    order = _v13_topology(pp, ss)
    rank = [0.0]*n
    for u in reversed(order):
        rank[u] = dur[u] + max((rank[v] + 550 for v in ss[u]), default=0)
    degree = [len(p) for p in pp]
    ready = [(-rank[u], u) for u in range(n) if not degree[u]]
    heapq.heapify(ready)
    rows, avail, owner = [[] for _ in range(K)], [0.0]*K, {}
    start, finish = [0.0]*n, [0.0]*n
    while ready:
        _, u = heapq.heappop(ready)
        options = []
        for k in range(K):
            begin = max(avail[k] + (100 if rows[k] else 0),
                        max((finish[p] + (1000 if owner[p] != k else 0)
                             for p in pp[u]), default=0))
            options.append((begin + dur[u], sum(owner[p] != k for p in pp[u]), k, begin))
        end, _, k, begin = min(options)
        rows[k].append(u)
        owner[u], start[u], finish[u], avail[k] = k, begin, end, end
        for v in sorted(ss[u]):
            degree[v] -= 1
            if not degree[v]:
                heapq.heappush(ready, (-rank[v], v))
    return rows, start, finish, owner


def _v13_tensor_cost(record, remap=None):
    """Bytes by reading/writing task; forced external outputs stay external.

    Multiple compute producers are ambiguous: their cost is held constant,
    so they cannot manufacture a merge benefit.
    """
    size, producers, consumers, forced = record
    if len(producers) > 1:
        return size * (len(producers) + len(consumers))
    if remap is not None:
        producers = {remap.get(u, u) for u in producers}
        consumers = {remap.get(u, u) for u in consumers}
    reads = len(consumers - producers)
    writes = len(producers) if producers and (forced or consumers - producers) else 0
    return size * (reads + writes)


def _v13_model(g, plan, official, K):
    op, nodes, pred, succ, edge_bytes, prod, cons, tensors = build_views(g)
    rawmap = {int(u): int(c) for u, c in plan["node_to_subgraph"].items()}
    if set(rawmap) != set(nodes):
        raise ValueError("seed mapping must cover every compute op exactly once")
    ids = sorted(set(rawmap.values()))
    ren = {u: i for i, u in enumerate(ids)}
    n = len(ids)
    if n > 512 or len(nodes) > 7000:
        raise ValueError("fast probe limit: <=512 seed tasks and <=7000 compute ops")
    clusters = [[] for _ in ids]
    mapping = {u: ren[c] for u, c in rawmap.items()}
    for u in nodes:
        clusters[mapping[u]].append(u)
    cp, cs, _ = quotient(clusters, nodes, succ, edge_bytes)
    # Include official task dependencies, including dependencies through COPY ops.
    for e in official.get("task_dependencies", []):
        a, b = ren[int(e["source"])], ren[int(e["target"])]
        if a != b:
            cp[b].add(a)
            cs[a].add(b)
    order = _v13_topology(cp, cs)
    rows = [[ren[int(u)] for u in row] for row in plan["core_schedules"]]
    if len(rows) != K:
        raise ValueError("seed core count differs from requested core count")
    measured = {int(u): float(r["local_makespan"])
                for u, r in official["step3_by_task"].items()}
    dur = [max(1.0, measured[u]) for u in ids]
    _, start, finish, owner = _v13_schedule(dur, cp, cs, K, rows)
    descendants = [0]*n
    for u in reversed(order):
        for v in cs[u]:
            descendants[u] |= (1 << v) | descendants[v]
    records, touch = {}, [set() for _ in ids]
    eligible = set(nodes)
    for tid, t in tensors.items():
        size = max(0, int(t.get("size", 0)))
        pp = {mapping[u] for u in prod.get(tid, ()) if u in eligible}
        cc = {mapping[u] for u in cons.get(tid, ()) if u in eligible}
        forced = bool(set(cons.get(tid, ())) - eligible) or not cons.get(tid)
        records[tid] = (size, pp, cc, forced)
        for c in pp | cc:
            touch[c].add(tid)
    op_order = topo(nodes, pred, succ)
    op_pos = {u: i for i, u in enumerate(op_order)}
    compute = [defaultdict(float) for _ in ids]
    for u in nodes:
        compute[mapping[u]][op[u].get("pipe", "other")] += max(0, float(op[u].get("cycles", 0)))
    compute_load = [max(row.values(), default=0.0) for row in compute]
    bottom = [0.0]*n
    for u in reversed(order):
        bottom[u] = dur[u] + max((bottom[v] for v in cs[u]), default=0)
    horizon = max(finish, default=1)
    criticality = [1.0 / (1.0 + max(0, horizon-start[u]-bottom[u]) / max(dur[u], 1))
                   for u in range(n)]
    return dict(op=op, nodes=nodes, op_pred=pred, op_succ=succ, prod=prod, cons=cons,
                tensors=tensors, clusters=clusters, pred=cp, succ=cs, duration=dur,
                start=start, finish=finish, owner=owner, records=records, touch=touch,
                descendants=descendants, criticality=criticality, op_pos=op_pos,
                peak_cache={}, compute_load=compute_load, K=K, baseline_proxy=horizon,
                baseline_bytes=sum(_v13_tensor_cost(t) for t in records.values()))


def _v13_footprint(model, group):
    """Sequential topo liveness proxy, aggregate L1+UB bytes (not a spill simulator).

    Inputs live from first use; internal values from production to last use;
    externally consumed outputs remain live through cluster completion.
    Pool placement, tiling, and actual local op schedule are intentionally unknown.
    """
    key = tuple(sorted(group))
    if key in model["peak_cache"]:
        return model["peak_cache"][key]
    members = {u for c in group for u in model["clusters"][c]}
    ordered = sorted(members, key=model["op_pos"].__getitem__)
    pos = {u: i for i, u in enumerate(ordered)}
    touched = set().union(*(model["touch"][c] for c in group))
    events = [0]*(len(ordered)+2)
    for tid in touched:
        ps = [pos[u] for u in model["prod"].get(tid, ()) if u in pos]
        cs = [pos[u] for u in model["cons"].get(tid, ()) if u in pos]
        points = ps + cs
        if not points:
            continue
        first, last = min(points), max(points)
        if ps and (set(model["cons"].get(tid, ())) - members or not model["cons"].get(tid)):
            last = len(ordered)-1
        size = model["records"][tid][0]
        events[first] += size
        events[last+1] -= size
    live = peak = 0
    for delta in events:
        live += delta
        peak = max(peak, live)
    model["peak_cache"][key] = peak
    return peak


def _v13_compute_floor(model, group):
    members = {u for c in group for u in model["clusters"][c]}
    ordered = sorted(members, key=model["op_pos"].__getitem__)
    finish, pipe = {}, defaultdict(float)
    for u in ordered:
        cost = max(0.0, float(model["op"][u].get("cycles", 0)))
        pipe[model["op"][u].get("pipe", "other")] += cost
        finish[u] = cost + max((finish[p] for p in model["op_pred"][u] if p in finish), default=0)
    return max(1.0, max(finish.values(), default=0), max(pipe.values(), default=0))


def _v13_pairs(model, limit=96):
    """Bound shared-input expansion to neighboring consumers, avoiding O(fanout²)."""
    pairs = set()
    for a, row in enumerate(model["succ"]):
        for b in row:
            pairs.add(tuple(sorted((a, b))))
    for size, pp, cc, _ in model["records"].values():
        if size < 4096:
            continue
        users = sorted(cc, key=lambda u: (model["start"][u], u))
        for i, a in enumerate(users):
            for b in users[i+1:i+5]:
                pairs.add(tuple(sorted((a, b))))
    scored = []
    n = len(model["clusters"])
    work_cap = max(1.0, 1.25 * sum(model["duration"]) / max(1, model["K"]))
    for a, b in sorted(pairs):
        if len(model["clusters"][a]) + len(model["clusters"][b]) > 512:
            continue
        if sum(model["duration"][c] for c in (a, b)) > work_cap:
            continue
        # A non-adjacent ancestor/descendant contraction necessarily creates a cycle.
        related = bool((model["descendants"][a] >> b) & 1 or
                       (model["descendants"][b] >> a) & 1)
        direct = b in model["succ"][a] or a in model["succ"][b]
        if related and not direct:
            continue
        affected = model["touch"][a] | model["touch"][b]
        remap = {b: a}
        saved = sum(_v13_tensor_cost(model["records"][t]) -
                    _v13_tensor_cost(model["records"][t], remap) for t in affected)
        if saved <= 0:
            continue
        overlap = max(0.0, min(model["finish"][a], model["finish"][b]) -
                      max(model["start"][a], model["start"][b]))
        # DDR occupancy is not productive compute parallelism. Penalizing the
        # full I/O-heavy overlap would double-count the cost of shared traffic.
        overlap *= min(1.0, model["compute_load"][a]/model["duration"][a],
                       model["compute_load"][b]/model["duration"][b])
        score = saved / BW - overlap
        scored.append((score, a, b, saved, overlap, direct))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    return scored[:limit], len(scored)


def _v13_contract(model, merges):
    n = len(model["clusters"])
    root = list(range(n))
    for a, b in merges:
        root[b] = a
    keys = sorted(set(root))
    ren = {u: i for i, u in enumerate(keys)}
    assignment = {u: ren[root[u]] for u in range(n)}
    groups = [[] for _ in keys]
    for u in range(n):
        groups[assignment[u]].append(u)
    pp, ss = [set() for _ in groups], [set() for _ in groups]
    for a, row in enumerate(model["succ"]):
        for b in row:
            x, y = assignment[a], assignment[b]
            if x != y:
                pp[y].add(x)
                ss[x].add(y)
    _v13_topology(pp, ss)
    return groups, assignment, pp, ss


def _v14_candidate(model, pairs, policy, deadline, duration_weight=0.55):
    alpha = policy["communication_weight"]
    cap = sum(eval_args()["capacity"].values())
    rows = []
    for _, a, b, saved, overlap, direct in pairs:
        if time.perf_counter() >= deadline:
            break
        floor = _v13_compute_floor(model, (a, b))
        peak_a = _v13_footprint(model, (a,))
        peak_b = _v13_footprint(model, (b,))
        peak = _v13_footprint(model, (a, b))
        excess = max(0, peak-cap) - max(0, max(peak_a, peak_b)-cap)
        spill_penalty = 0.5 * max(0, excess) / BW
        merged_duration = max(floor, model["duration"][a]+model["duration"][b] - duration_weight*saved/BW)
        # All external predecessors must finish before the merged Task starts.
        external = (model["pred"][a] | model["pred"][b]) - {a, b}
        ready = max((model["finish"][u] for u in external), default=0)
        new_end = max(ready, min(model["start"][a], model["start"][b])) + merged_duration
        release = max((max(0, new_end-model["finish"][u]) *
                       max((model["criticality"][v] for v in model["succ"][u] - {a, b}), default=0)
                       for u in (a, b)), default=0)
        # Boundary wait is only a weighted opportunity, not a guaranteed saving.
        wait = (100 if model["owner"][a] == model["owner"][b] else 1000) * 0.25 if direct else 0
        gain = alpha*saved/BW + wait - overlap - policy["release_weight"]*release - spill_penalty
        if gain > 0:
            rows.append(dict(a=a, b=b, saved_bytes=saved, score=gain,
                             parallel_overlap_cycles=overlap, release_penalty_cycles=release,
                             cache_risk_cycles=spill_penalty, duration=merged_duration))
    rows.sort(key=lambda r: (-r["score"], r["a"], r["b"]))
    chosen, used = [], set()
    max_merges = min(8, max(1, len(model["clusters"])//8))
    rejected_cycles = 0
    for row in rows:
        if time.perf_counter() >= deadline or len(chosen) >= max_merges:
            break
        a, b = row["a"], row["b"]
        if a in used or b in used:
            continue
        trial = [(r["a"], r["b"]) for r in chosen] + [(a, b)]
        try:
            _v13_contract(model, trial)
        except ValueError:
            rejected_cycles += 1
            continue
        chosen.append(row)
        used.update((a, b))
    if not chosen:
        return None, dict(status="no_positive_safe_merge", cycle_rejections=rejected_cycles)
    groups, assignment, pred, succ = _v13_contract(model, [(r["a"], r["b"]) for r in chosen])
    costs = {tuple(sorted((r["a"], r["b"]))): r["duration"] for r in chosen}
    durations = [model["duration"][group[0]] if len(group)==1 else costs[tuple(group)] for group in groups]
    schedules, _, finish, _ = _v13_schedule(durations, pred, succ, model["K"])
    mapping = {str(u): assignment[c] for c, cluster in enumerate(model["clusters"]) for u in cluster}
    new_bytes = sum(_v13_tensor_cost(r, assignment) for r in model["records"].values())
    return {"node_to_subgraph": mapping, "core_schedules": schedules}, dict(
        status="generated", merges=chosen, tasks_before=len(model["clusters"]),
        tasks_after=len(groups), estimated_bytes_saved=model["baseline_bytes"]-new_bytes,
        predicted_makespan=max(max(finish, default=0), new_bytes/BW),
        cycle_rejections=rejected_cycles, duration_weight=duration_weight,
        predicted_task_durations={str(i): d for i, d in enumerate(durations)})



def _v14_partition_key(plan):
    groups = defaultdict(list)
    for u, c in plan["node_to_subgraph"].items():
        groups[int(c)].append(int(u))
    return tuple(sorted(tuple(sorted(v)) for v in groups.values()))


def _v14_plan_key(plan):
    groups = defaultdict(list)
    for u, c in plan["node_to_subgraph"].items():
        groups[int(c)].append(int(u))
    groups = {c: tuple(sorted(v)) for c, v in groups.items()}
    return tuple(tuple(groups[int(c)] for c in row) for row in plan["core_schedules"])


def _v14_traffic_parts(model, plan):
    # These are logical tensor/task estimates, not the official added-copy metric.
    assignment = {c: int(plan["node_to_subgraph"][str(cluster[0])])
                  for c, cluster in enumerate(model["clusters"])}
    parts = dict(estimated_external_input_read_bytes_saved=0,
                 estimated_boundary_transfer_bytes_saved=0)
    for record in model["records"].values():
        saved = _v13_tensor_cost(record) - _v13_tensor_cost(record, assignment)
        key = ("estimated_external_input_read_bytes_saved" if not record[1]
               else "estimated_boundary_transfer_bytes_saved")
        parts[key] += saved
    return parts


def _v14_duration_errors(model, plan, metadata, result):
    measured = {int(k): float(v["local_makespan"])
                for k, v in result["step3_by_task"].items()}
    predicted = metadata.get("predicted_task_durations", {})
    errors = []
    for merge in metadata.get("merges", []):
        original_op = model["clusters"][merge["a"]][0]
        task = int(plan["node_to_subgraph"][str(original_op)])
        estimate = predicted.get(str(task))
        actual = measured.get(task)
        if estimate is None or actual is None:
            continue
        errors.append(dict(old_tasks=[merge["a"], merge["b"]], task=task,
                           predicted_local_makespan=estimate, measured_local_makespan=actual,
                           error_cycles=estimate-actual,
                           error_pct=100*(estimate-actual)/max(1.0, actual)))
    return errors


def _v14_refine(g, plan, actual, K):
    """Reuse the first official result. This function makes ZERO evaluator calls."""
    model = _v13_model(g, plan, actual, K)
    rows, _, finish, _ = _v13_schedule(model["duration"], model["pred"], model["succ"], K)
    mapping = {str(u): c for c, nodes in enumerate(model["clusters"]) for u in nodes}
    proposed = dict(node_to_subgraph=mapping, core_schedules=rows)
    return proposed, dict(status="generated", source="official_local_makespan",
                          proxy_before=model["baseline_proxy"],
                          proxy_after=max(finish, default=0),
                          predicted_task_durations={str(i): d for i, d in enumerate(model["duration"])})


def solve_v14(g, K=5, seed_plan=None, single_makespan=None, full_v10=True,
              probe_seconds=3.0, time_limit=None, duration_weight=0.55):
    """Search from scratch by default, then evaluate at most TWO V14 candidates.

    Two different partitions: evaluate both; one unique partition: evaluate it,
    then reassign from its measured local Task durations and evaluate if changed.
    Retain the best official result including the seed, even if a probe fails.
    Time budgets are soft and never interrupt an in-progress official evaluator.
    """
    begun = time.perf_counter()
    if K < 1 or probe_seconds < 0 or not 0 <= duration_weight <= 1:
        raise ValueError("invalid cores/probe_seconds/duration_weight")
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive")
    deadline = begun + time_limit if time_limit is not None else float("inf")
    ran_full_v10 = seed_plan is None and full_v10
    if seed_plan is None:
        if not full_v10:
            raise ValueError("快速模式需要同核数 V10 seed；请传入 --seed-plan 或启用 V10 回退")
        old = solve(g, K)
        seed_plan, single_makespan = old["plan"], old["single"]
    single_seconds = 0.0
    if single_makespan is None:
        t = time.perf_counter()
        single_makespan = evaluate_singlecore(g, **eval_args())["makespan"]
        single_seconds = time.perf_counter()-t
    if not math.isfinite(single_makespan) or single_makespan <= 0:
        raise ValueError("single makespan must be finite and positive")
    t = time.perf_counter()
    baseline = evaluate_scene_a(g, seed_plan, **eval_args())
    baseline_seconds = time.perf_counter()-t
    best, best_plan, selected = baseline, seed_plan, "seed_plan"
    logs, variants = [], []
    info = dict(status="ok", seed_makespan=baseline["makespan"], evaluated=0,
                single_seconds=single_seconds, baseline_seconds=baseline_seconds,
                candidate_generation_seconds=0.0, refinement_generation_seconds=0.0,
                official_candidate_seconds=0.0, policies=[], duration_weight=duration_weight,
                seed_mode="full_v10" if ran_full_v10 else "saved_plan",
                refinement=dict(status="not_attempted"))
    base_added = baseline.get("data_movement_bytes", {}).get("added_copy_bytes")

    def evaluate(name, plan, metadata, model, refined=False):
        nonlocal best, best_plan, selected
        if info["evaluated"] >= 2 or time.perf_counter() >= deadline:
            metadata["status"] = "evaluation_budget_reached"
            return None
        info["evaluated"] += 1
        t = time.perf_counter()
        try:
            actual = evaluate_scene_a(g, plan, **eval_args())
        except Exception as exc:
            elapsed = time.perf_counter()-t
            info["official_candidate_seconds"] += elapsed
            logs.append(dict(name=name, error=repr(exc), seconds=elapsed))
            metadata["status"] = "evaluation_failed"
            return None
        elapsed = time.perf_counter()-t
        info["official_candidate_seconds"] += elapsed
        # Preserve a valid better plan before computing optional diagnostics.
        if actual["makespan"] < best["makespan"]:
            best, best_plan, selected = actual, plan, name
        added = actual.get("data_movement_bytes", {}).get("added_copy_bytes")
        rec = dict(name=name, makespan=actual["makespan"],
                   speedup=single_makespan/actual["makespan"], seconds=elapsed,
                   gain_cycles=baseline["makespan"]-actual["makespan"],
                   gain_pct=100*(baseline["makespan"]-actual["makespan"])/baseline["makespan"],
                   added_copy_bytes=added,
                   actual_added_copy_bytes_saved=base_added-added if base_added is not None and added is not None else None,
                   predicted_makespan=metadata.get("predicted_makespan", metadata.get("proxy_after")))
        try:
            rec.update(_v14_traffic_parts(model, plan))
            rec["estimated_bytes_saved"] = (rec["estimated_external_input_read_bytes_saved"] +
                                            rec["estimated_boundary_transfer_bytes_saved"])
            if not refined:
                rec["merged_task_duration_errors"] = _v14_duration_errors(model, plan, metadata, actual)
        except Exception as exc:
            rec["diagnostic_error"] = repr(exc)
        logs.append(rec)
        metadata["status"] = "evaluated"
        return actual

    t = time.perf_counter()
    try:
        gen_deadline = min(deadline, t + probe_seconds)
        model = _v13_model(g, seed_plan, baseline, K)
        pairs, total_pairs = _v13_pairs(model)
        info.update(eligible_pairs=total_pairs, shortlisted_pairs=len(pairs),
                    seed_tasks=len(model["clusters"]), baseline_proxy=model["baseline_proxy"],
                    baseline_estimated_bytes=model["baseline_bytes"])
        by_partition = {}
        policies = [dict(name="conservative", communication_weight=0.35, release_weight=1.0),
                    dict(name="communication", communication_weight=0.9, release_weight=0.65)]
        for policy in policies:
            if time.perf_counter() >= gen_deadline:
                info["policies"].append(dict(name=policy["name"], status="generation_budget_reached"))
                continue
            plan, meta = _v14_candidate(model, pairs, policy, gen_deadline, duration_weight)
            meta.update(name=policy["name"], communication_weight=policy["communication_weight"])
            if plan is not None:
                meta.update(_v14_traffic_parts(model, plan))
                key = _v14_partition_key(plan)
                if key in by_partition:
                    meta.update(status="duplicate_partition", same_partition_as=by_partition[key])
                else:
                    by_partition[key] = policy["name"]
                    variants.append(("v14_"+policy["name"], plan, meta))
            info["policies"].append(meta)
        info["unique_partitions"] = len(variants)
    except Exception as exc:
        info["status"], info["error"] = "generation_failed_seed_retained", repr(exc)
    info["candidate_generation_seconds"] = time.perf_counter()-t
    try:
        if len(variants) >= 2:
            info["evaluation_route"] = "two_distinct_partitions"
            info["refinement"]["status"] = "budget_reserved_for_distinct_partitions"
            for name, plan, meta in variants[:2]:
                evaluate(name, plan, meta, model)
        elif len(variants) == 1:
            info["evaluation_route"] = "one_partition_then_measured_duration_reassignment"
            name, plan, meta = variants[0]
            actual = evaluate(name, plan, meta, model)
            if actual is None:
                info["refinement"]["status"] = "first_evaluation_unavailable"
            elif time.perf_counter() >= deadline:
                info["refinement"]["status"] = "total_budget_reached"
            else:
                t = time.perf_counter()
                refined, refinement = _v14_refine(g, plan, actual, K)
                info["refinement_generation_seconds"] = time.perf_counter()-t
                info["refinement"] = refinement
                if _v14_plan_key(refined) == _v14_plan_key(plan):
                    refinement["status"] = "unchanged_schedule"
                else:
                    refined_actual = evaluate(name+"_measured_reassign", refined, refinement, model, refined=True)
                    if refined_actual is not None:
                        refinement["gain_over_first_candidate"] = actual["makespan"]-refined_actual["makespan"]
        else:
            info["evaluation_route"] = "no_merge_candidate"
    except Exception as exc:
        info["status"], info["error"] = "refinement_failed_best_retained", repr(exc)
    info["total_seconds"] = time.perf_counter()-begun
    for rec in logs:
        if "makespan" in rec:
            rec["gap_to_best_pct"] = 100*(rec["makespan"]/best["makespan"]-1)
    return dict(single=single_makespan, best_makespan=best["makespan"],
                speedup=single_makespan/best["makespan"], selected=selected,
                plan=best_plan, candidates=logs, v14=info,
                official_evaluations=1+info["evaluated"],
                added_copy_bytes=best.get("data_movement_bytes", {}).get("added_copy_bytes"),
                used_cores=sum(bool(row) for row in best_plan["core_schedules"]))


def _v14_read_single_cache(path, case):
    """Read a prior run_all_cases CSV or singlecore_cache.json; never modify it.

    User-selected cache must describe the same dataset/evaluator settings.
    Validate all matching positive entries agree rather than choosing silently.
    """
    import csv
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            values = [float(row["single"]) for row in csv.DictReader(f)
                      if row.get("case")==case and row.get("single")]
    else:
        with path.open(encoding="utf-8-sig") as f:
            obj = json.load(f)
        values = [float(obj[case])] if case in obj else []
    if not values:
        return None
    if any(not math.isfinite(v) or v <= 0 for v in values) or len(set(values)) != 1:
        raise ValueError("单核缓存存在无效或不一致的值: " + case)
    return values[0]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V14: self-contained search + <=2 cluster/feedback evaluations")
    parser.add_argument("graph")
    parser.add_argument("cores", nargs="?", type=int, default=5)
    parser.add_argument("output_plan", nargs="?")
    parser.add_argument("--seed-plan", help="optional explicit seed; default: search from scratch")
    parser.add_argument("--single", type=float)
    parser.add_argument("--single-cache", help="existing run_all_cases CSV or singlecore_cache.json")
    parser.add_argument("--time-limit", type=float, help="soft budget; in-progress official calls are not interrupted")
    parser.add_argument("--probe-seconds", type=float, default=3.0)
    parser.add_argument("--duration-weight", type=float, default=0.55,
                        help="fixed local-duration savings factor shared by BOTH merge policies")
    parser.add_argument("--full-v10", action="store_true",
                        help="compatibility option; full search without a seed is already the default")
    args = parser.parse_args()
    with open(args.graph, encoding="utf-8") as f:
        graph = json.load(f)
    seed, seed_path = None, None
    paths = []
    if args.seed_plan:
        paths.append(Path(args.seed_plan))
    for path in paths:
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            candidate = json.load(f)
        if len(candidate.get("core_schedules", [])) == args.cores:
            seed, seed_path = candidate, str(path.resolve())
            break
    run_full_v10 = args.full_v10 or seed is None
    single, source = args.single, "--single" if args.single is not None else "official_evaluation"
    cache = args.single_cache
    if single is None and cache:
        try:
            single = _v14_read_single_cache(cache, Path(args.graph).stem)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error("无法读取指定单核缓存: " + str(exc))
        if single is not None:
            source = str(Path(cache).resolve())
    with contextlib.redirect_stdout(sys.stderr):
        _load_official()
        result = solve_v14(graph, args.cores, seed, single, run_full_v10,
                           args.probe_seconds, args.time_limit, args.duration_weight)
    if seed is None:
        source = "official_evaluation"
    result["v14"].update(seed_path=seed_path, single_source=source)
    if args.output_plan:
        target = Path(args.output_plan)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".tmp")
        temp.write_text(json.dumps(result["plan"], ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(temp, target)
        report = target.with_name(target.stem + ".diagnostics.json")
        tmp_report = report.with_name(report.name + ".tmp")
        tmp_report.write_text(json.dumps({k: v for k, v in result.items() if k != "plan"},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_report, report)
    print(json.dumps({k: v for k, v in result.items() if k != "plan"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
