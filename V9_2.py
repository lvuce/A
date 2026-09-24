#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 1 / Scene A V8+ solver (original V8 portfolio + slack-aware cut candidates).

V8+ keeps every original V8 candidate and adds two low-cost slack-aware cut candidates.

Original V8 combines:
1) DDR-aware WCC packing with adaptive active-core count;
2) communication-aware safe chain coarsening;
3) level packing with several granularities;
4) frontier growth for dense single-WCC graphs;
5) core/phase assignment on coarse graphs;
6) evaluator-calibrated exact-duration reassignment;
7) task-level simulated annealing for hard compact task DAGs.

The final winner is always selected by the official Problem-1 evaluator.

Important for Problem 1 / Scene A:
- every subgraph is an independent Task;
- every cross-subgraph tensor goes through DDR even when the two Tasks are on the same core;
- core assignment therefore keeps the original V8 100/1000 Task-wait model;
- the new slack signal is used only while deciding which op dependencies to keep inside a subgraph.
This script expects the official attachment to be unzipped and its code directory
available on PYTHONPATH, or placed at /mnt/data/a/code as in the batch runner.
"""
import os, sys, json, math, random, heapq
from collections import defaultdict, deque

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL_CODE = os.environ.get("HUAWEI_CODE_DIR", "/mnt/data/a/code")
if OFFICIAL_CODE not in sys.path:
    sys.path.insert(0, OFFICIAL_CODE)

from multicore_cut_evaluate_problem_1 import evaluate_scene_a
from singlecore_evaluate import evaluate_singlecore, build_singlecore_plan

BW = 60.0
EXCLUDED = {"COPY_IN", "COPY_OUT"}

# Per-process caches. The batch runner launches one process per case, so these stay small.
_VIEW_CACHE = {}
_TIMING_CACHE = {}


def eval_args():
    return dict(
        bandwidth=60,
        capacity={"L1": 524288, "UB": 131072},
        cross_core_wait=1000,
        same_core_wait=100,
    )


def build_views(g):
    """Build the op-only DAG and tensor-byte view once per case.

    Problem 1 decisions are made only for non-COPY ops.  Tensor sizes are retained
    on op-to-op dependencies because crossing a subgraph boundary causes DDR traffic.
    """
    key = id(g)
    cached = _VIEW_CACHE.get(key)
    if cached is not None and cached[0] is g:
        return cached[1]

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

    views = (op, nodes, pred, succ, edge_bytes, prod, cons, tensor_by)
    _VIEW_CACHE[key] = (g, views)
    return views


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


def static_timing(g):
    """Cheap compute-only DAG timing used only by slack-aware CUT heuristics.

    The timing deliberately ignores the final shared-DDR simulator.  It is not used
    to predict the final makespan.  It only estimates how much delay an op edge can
    tolerate before the compute-only critical path grows.

    Complexity: O(V+E), cached once per case.
    """
    key = id(g)
    cached = _TIMING_CACHE.get(key)
    if cached is not None and cached[0] is g:
        return cached[1]

    op, nodes, pred, succ, edge_bytes, *_ = build_views(g)
    order = topo(nodes, pred, succ)
    dur = {u: max(1.0, float(op[u].get("cycles", 0))) for u in nodes}

    # Earliest finish using only compute-node cycles.
    ef = {}
    for u in order:
        ef[u] = dur[u] + max((ef[p] for p in pred[u]), default=0.0)
    cp_len = max(ef.values(), default=0.0)

    # Bottom level: remaining compute work from u to an exit, inclusive of u.
    bottom = {}
    for u in reversed(order):
        bottom[u] = dur[u] + max((bottom[v] for v in succ[u]), default=0.0)

    latest_start = {u: cp_len - bottom[u] for u in nodes}
    edge_slack = {}
    for u in nodes:
        for v in succ[u]:
            edge_slack[(u, v)] = max(0.0, latest_start[v] - ef[u])

    out = {
        "dur": dur,
        "ef": ef,
        "bottom": bottom,
        "cp_len": cp_len,
        "edge_slack": edge_slack,
    }
    _TIMING_CACHE[key] = (g, out)
    return out


def cut_copy_work(bytes_):
    """Approximate extra DDR service work created by a subgraph cut.

    In Problem 1 every cross-subgraph tensor is materialized through DDR even for
    same-core Tasks.  A boundary is therefore approximated as COPY_OUT + COPY_IN.
    This is only a cut heuristic; the official evaluator still models bandwidth
    sharing and determines the true makespan.
    """
    return 2.0 * float(bytes_) / BW


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


def chain_clusters(g, K, max_ops=256, work_factor=1.5, mode="none", strong_quantile=.5, slack_aware=False):
    op, nodes, pred, succ, edge_bytes, *_ = build_views(g)
    edge_slack = static_timing(g)["edge_slack"] if slack_aware else None
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
        # Original V8 behavior is preserved when slack_aware=False.
        for _ in range(30):
            cp, cs, eb = quotient(clusters, nodes, succ, edge_bytes)
            vals = sorted(x for x in eb.values() if x > 0)
            thr = vals[min(len(vals)-1, max(0, int(strong_quantile * (len(vals)-1))))] if vals else 10**30
            cwork = [sum(max(1, int(op[u].get("cycles", 0))) for u in c) for c in clusters]

            q_exposed = None
            q_copy = None
            if slack_aware:
                # Aggregate exposure at original op-edge granularity.  This is more
                # conservative than applying one minimum slack to the sum of all bytes.
                cid_now = {u: i for i, c in enumerate(clusters) for u in c}
                q_exposed = defaultdict(float)
                q_copy = defaultdict(float)
                for u in nodes:
                    a0 = cid_now[u]
                    for v in succ[u]:
                        b0 = cid_now[v]
                        if a0 == b0:
                            continue
                        copy_work = cut_copy_work(edge_bytes.get((u, v), 0))
                        q_copy[(a0, b0)] += copy_work
                        q_exposed[(a0, b0)] += max(
                            0.0,
                            copy_work - edge_slack.get((u, v), 0.0),
                        )

            cand = []
            for (a, b), bytes_ in eb.items():
                if not (len(cs[a]) == 1 or len(cp[b]) == 1):
                    continue

                comp = max(1.0, min(cwork[a], cwork[b]))

                if not slack_aware:
                    # Exact original V8 criterion.
                    comm = 2.0 * bytes_ / BW
                    ratio = comm / comp
                    if mode == "strong":
                        ok = bytes_ >= thr and ratio >= .08
                    else:  # aggressive
                        ok = bytes_ >= thr or ratio >= .22
                    if ok:
                        cand.append((comm * (1.0 + ratio), a, b))
                    continue

                # Problem-1 slack-aware CUT criterion.  The copy work is incurred by
                # crossing the subgraph boundary regardless of whether the two Tasks
                # later run on the same or on different cores.
                copy_work = q_copy.get((a, b), cut_copy_work(bytes_))
                exposed = q_exposed.get((a, b), 0.0)
                exposure_frac = exposed / max(copy_work, 1e-9)
                ratio = exposed / comp

                # Do NOT let a large byte count bypass slack completely.  Original V8
                # already covers the byte-dominant policy; this extra portfolio branch
                # intentionally focuses on communication that is hard to hide.
                if mode == "strong":
                    ok = bytes_ >= thr and exposure_frac >= .20 and ratio >= .05
                else:  # aggressive slack candidate
                    ok = (bytes_ >= thr and exposure_frac >= .10) or ratio >= .20

                if ok:
                    # A small copy-work term keeps global DDR pressure visible even
                    # when most of the edge delay is statically hideable.
                    priority = exposed * (1.0 + ratio) + 0.10 * copy_work
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


def solve(g, K=5):
    ev=eval_args()
    single=evaluate_singlecore(g,**ev)["makespan"]
    base=build_singlecore_plan(g)
    base["core_schedules"] += [[] for _ in range(K-1)]
    best=[single,"single",base]
    records=[]
    eval_cache={}

    def plan_signature(plan):
        return json.dumps(plan, sort_keys=True, separators=(",", ":"))

    def eval_cached(plan):
        sig=plan_signature(plan)
        if sig not in eval_cache:
            eval_cache[sig]=evaluate_scene_a(g,plan,**ev)
        return eval_cache[sig]

    def test(name, plan):
        nonlocal best
        sig=plan_signature(plan)
        reused=sig in eval_cache
        try:r=eval_cached(plan)
        except Exception as e:
            records.append({"name":name,"error":repr(e)})
            return
        sp=single/r["makespan"]
        rec={"name":name,"makespan":r["makespan"],"speedup":sp,"added_copy_bytes":r["data_movement_bytes"]["added_copy_bytes"]}
        if reused: rec["reused_eval"]=True
        records.append(rec)
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

    # V8+ additions: two extra CUT candidates only.
    # They keep the original V8 HEFT/core-assignment model unchanged.  Therefore
    # the new idea cannot delete an old candidate; the official evaluator simply
    # chooses it when the slack-aware partition is genuinely better.
    for mode,maxops,wf,q,bins in [
        ("aggressive",96,.9,.10,30),
        ("strong",128,.75,.40,12),
    ]:
        try:
            op,nodes,pred,succ,eb,cl=chain_clusters(
                g,K,maxops,wf,mode,q,slack_aware=True
            )
            groups=level_pack(cl,op,nodes,succ,eb,K,bins)
            test(f"slack_level_{mode}_{maxops}_{bins}",heft_plan(g,groups,K))
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

    # V8 task-level SA only on genuinely hard compact task DAGs.
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

    return {
        "single":single,
        "best_makespan":best[0],
        "speedup":single/best[0],
        "selected":best[1],
        "plan":best[2],
        "candidates":records,
    }


def main():
    if len(sys.argv)<2:
        print("Usage: python V8_solver_50case.py <case.json> [num_cores=5] [output_plan.json]",file=sys.stderr)
        raise SystemExit(2)
    graph_path=sys.argv[1]
    K=int(sys.argv[2]) if len(sys.argv)>2 else 5
    out_path=sys.argv[3] if len(sys.argv)>3 else None
    g=json.load(open(graph_path,encoding="utf-8"))
    r=solve(g,K)
    if out_path:
        with open(out_path,"w",encoding="utf-8") as f:
            json.dump(r["plan"],f,ensure_ascii=False,separators=(",",":"))
    print(json.dumps({k:v for k,v in r.items() if k!="plan"},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
