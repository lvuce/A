#!/usr/bin/env python3
"""Problem-1 v4 adaptive portfolio solver.

v4 additions over v3.1:
1) evaluator-calibrated HEFT re-assignment: after partitioning, use official per-task
   local makespan to re-map the fixed subgraphs to cores;
2) safe heavy-edge merge refinement for difficult dense DAGs;
3) adaptive refinement budget so easy near-ideal cases stay fast.

Development solver: official evaluator is used to select among a small deterministic
candidate portfolio and local refinements.
"""
import argparse, json, time, heapq, sys
from collections import defaultdict, deque
from pathlib import Path

# The official evaluator lives in the supplied attachment, not beside this script.
ATTACHMENT_CODE = Path(__file__).resolve().parent / '通用神经网络处理器下的多核调度问题  附件' / 'code'
if not ATTACHMENT_CODE.is_dir():
    raise FileNotFoundError(f'Official evaluator directory is missing: {ATTACHMENT_CODE}')
sys.path.insert(0, str(ATTACHMENT_CODE))

from A题_problem1_heuristic_v3 import (
    generate_safe_packed, core_greedy_phase_plan, wcc_pack_plan3,
    frontier_grow_plan, build_views
)
from multicore_cut_evaluate_problem_1 import evaluate_scene_a, read_scene_a_config
from evaluation_validation import read_evaluation_config
from singlecore_evaluate import build_singlecore_plan, evaluate_singlecore


def _ev_args(config):
    cfg=read_evaluation_config(config); waits=read_scene_a_config(config)
    return dict(bandwidth=cfg['bandwidth'],capacity=cfg['capacity'],
                cross_core_wait=waits['task_cross_core_wait_cycles'],
                same_core_wait=waits['task_same_core_wait_cycles'])


def evaluate_baseline(graph,config):
    ev=_ev_args(config)
    return evaluate_singlecore(graph,bandwidth=ev['bandwidth'],capacity=ev['capacity'])


def _subgraph_edge_bytes(graph, plan):
    _,_,nodes,_,succs,edge_bytes=build_views(graph)
    ns={int(k):int(v) for k,v in plan['node_to_subgraph'].items()}
    eb=defaultdict(int)
    for u in nodes:
        a=ns[u]
        for v in succs[u]:
            b=ns[v]
            if a!=b: eb[(a,b)] += edge_bytes.get((u,v),0)
    return eb


def exact_duration_reassign(graph, plan, result, ncores, alpha=2.0,
                            cross_wait=1000.0, same_wait=100.0):
    """Keep partition fixed; remap subgraphs with official local task makespans.
    alpha weights communication volume in upward-rank tie breaking.
    """
    sgs=sorted(set(int(v) for v in plan['node_to_subgraph'].values()))
    # Our generators use dense sgids. Normalize defensively if needed.
    ren={s:i for i,s in enumerate(sgs)}
    S=len(sgs)
    if any(s!=i for i,s in enumerate(sgs)):
        mapping={k:ren[int(v)] for k,v in plan['node_to_subgraph'].items()}
    else:
        mapping=dict(plan['node_to_subgraph'])
    dur=[1.0]*S
    for k,v in result['step3_by_task'].items():
        kk=ren.get(int(k),int(k))
        if 0<=kk<S: dur[kk]=float(v['local_makespan'])
    preds=[set() for _ in range(S)]; succ=[set() for _ in range(S)]
    for e in result['task_dependencies']:
        a=ren.get(int(e['source']),int(e['source'])); b=ren.get(int(e['target']),int(e['target']))
        if a!=b: succ[a].add(b); preds[b].add(a)
    raw_eb=_subgraph_edge_bytes(graph, {'node_to_subgraph':mapping,'core_schedules':plan['core_schedules']})
    eb=defaultdict(int)
    for (a,b),x in raw_eb.items(): eb[(ren.get(a,a),ren.get(b,b))]+=x
    indeg=[len(preds[i]) for i in range(S)]; q=[i for i in range(S) if indeg[i]==0]; heapq.heapify(q); top=[]
    while q:
        u=heapq.heappop(q); top.append(u)
        for v in succ[u]:
            indeg[v]-=1
            if indeg[v]==0: heapq.heappush(q,v)
    if len(top)!=S: raise RuntimeError('task graph cycle in exact reassignment')
    rank=[0.0]*S
    for u in reversed(top):
        rank[u]=dur[u]+max((cross_wait + alpha*2.0*eb.get((u,v),0)/60.0 + rank[v] for v in succ[u]),default=0.0)
    indeg=[len(preds[i]) for i in range(S)]; ready={i for i in range(S) if indeg[i]==0}
    core_av=[0.0]*ncores; sched=[[] for _ in range(ncores)]; assigned={}; finish={}
    while ready:
        u=max(ready,key=lambda x:(rank[x],dur[x],-x)); ready.remove(u)
        best=None
        for k in range(ncores):
            est=core_av[k]+(same_wait if sched[k] else 0.0); cross_cnt=0; cross_bytes=0
            for p in preds[u]:
                if assigned[p]==k:
                    est=max(est,finish[p])
                else:
                    est=max(est,finish[p]+cross_wait); cross_cnt+=1; cross_bytes += eb.get((p,u),0)
            ft=est+dur[u]
            key=(ft, cross_cnt, core_av[k], k)
            if best is None or key<best[0]: best=(key,k,ft)
        _,k,ft=best; assigned[u]=k; finish[u]=ft; core_av[k]=ft; sched[k].append(u)
        for v in succ[u]:
            indeg[v]-=1
            if indeg[v]==0: ready.add(v)
    return {'node_to_subgraph':mapping,'core_schedules':sched}


def _quotient(graph,plan):
    ns={int(k):int(v) for k,v in plan['node_to_subgraph'].items()}
    sgs=sorted(set(ns.values())); ren={s:i for i,s in enumerate(sgs)}; S=len(sgs)
    groups=[[] for _ in range(S)]
    for u,s in ns.items(): groups[ren[s]].append(u)
    _,_,nodes,_,succs,edge_bytes=build_views(graph)
    pred=[set() for _ in range(S)]; suc=[set() for _ in range(S)]; eb=defaultdict(int)
    for u in nodes:
        a=ren[ns[u]]
        for v in succs[u]:
            b=ren[ns[v]]
            if a!=b: suc[a].add(b); pred[b].add(a); eb[(a,b)]+=edge_bytes.get((u,v),0)
    oldcore={ren[int(s)]:k for k,ls in enumerate(plan['core_schedules']) for s in ls if int(s) in ren}
    return groups,pred,suc,eb,oldcore


def safe_merge_pair(graph,plan,a,b):
    groups,pred,suc,eb,oldcore=_quotient(graph,plan)
    if b not in suc[a] or not (len(suc[a])==1 or len(pred[b])==1): return None
    merged=[]; old2new={}; done=False
    for i,g in enumerate(groups):
        if i in (a,b):
            if not done:
                ni=len(merged); old2new[a]=ni; old2new[b]=ni; merged.append(groups[a]+groups[b]); done=True
        else:
            old2new[i]=len(merged); merged.append(g)
    P=[set() for _ in merged]; Suc=[set() for _ in merged]
    for i in range(len(groups)):
        for j in suc[i]:
            ni,nj=old2new[i],old2new[j]
            if ni!=nj: Suc[ni].add(nj); P[nj].add(ni)
    indeg=[len(x) for x in P]; q=deque(i for i,d in enumerate(indeg) if d==0); top=[]
    while q:
        x=q.popleft(); top.append(x)
        for y in Suc[x]:
            indeg[y]-=1
            if indeg[y]==0:q.append(y)
    if len(top)!=len(merged): return None
    pos={s:i for i,s in enumerate(top)}
    cores=[None]*len(merged)
    for old in range(len(groups)):
        new=old2new[old]
        if cores[new] is None or old==a: cores[new]=oldcore.get(old,0)
    sched=[[] for _ in plan['core_schedules']]
    for s,k in enumerate(cores): sched[k].append(s)
    for k in range(len(sched)): sched[k].sort(key=lambda s:pos[s])
    mapping={str(u):i for i,g in enumerate(merged) for u in g}
    return {'node_to_subgraph':mapping,'core_schedules':sched}


def solve(graph,ncores,config,return_result=False,baseline_result=None):
    ev=_ev_args(config)
    base=build_singlecore_plan(graph); base['core_schedules'] += [[] for _ in range(ncores-1)]
    br=baseline_result if baseline_result is not None else evaluate_scene_a(graph,base,**ev)
    best={'ms':br['makespan'],'name':'single_fallback','plan':base,'res':br,'meta':{'num_subgraphs':1}}
    records=[]
    def test(name,fn):
        nonlocal best
        t=time.time()
        try:
            p,m=fn(); r=evaluate_scene_a(graph,p,**ev)
            rec={'name':name,'makespan':r['makespan'],'added_copy_bytes':r['data_movement_bytes']['added_copy_bytes'],
                 'subgraphs':m.get('num_subgraphs'),'seconds':time.time()-t}; records.append(rec)
            if r['makespan']<best['ms']: best={'ms':r['makespan'],'name':name,'plan':p,'res':r,'meta':m}
        except Exception as e:
            records.append({'name':name,'error':repr(e),'seconds':time.time()-t})
    def finish():
        if best['name']=='single_fallback' and br.get('num_cores')!=ncores:
            best['res']=evaluate_scene_a(graph,base,**ev)
        info={
            'selected':best['name'],'single_makespan':br['makespan'],
            'best_makespan':best['ms'],'speedup':br['makespan']/best['ms'],
            'added_copy_bytes':best['res']['data_movement_bytes']['added_copy_bytes'],
            'subgraphs':best['meta'].get('num_subgraphs'),'candidates':records
        }
        if return_result:
            return best['plan'],info,best['res']
        return best['plan'],info
    opn=sum(1 for o in graph['ops'] if o.get('op') not in {'COPY_IN','COPY_OUT'})
    # Large graphs make each official simulation expensive. Use a bounded portfolio;
    # the same official evaluator still scores and selects every tested plan.
    if opn>10000:
        test('wcc_pack3',lambda:wcc_pack_plan3(graph,ncores))
        if br['makespan']/best['ms'] < .90*ncores:
            test('strong_long8',lambda:generate_safe_packed(
                graph,ncores,mode='strong_safe',max_ops=1024,work_factor=1.5,
                strong_quantile=.5,bins_per_level=max(ncores,8),
                cross_wait=1000,same_wait=100))
            test('aggr_mixed4k',lambda:generate_safe_packed(
                graph,ncores,mode='aggressive_safe',max_ops=160,work_factor=.75,
                strong_quantile=.15,bins_per_level=4*ncores,
                cross_wait=1000,same_wait=100))
        return finish()
    test('wcc_pack3',lambda:wcc_pack_plan3(graph,ncores))
    # Active-core WCC variants: sometimes fewer simultaneous DDR-heavy tasks beat using all cores.
    for _m in range(2,ncores):
        def _wcc_active(m=_m):
            p,meta=wcc_pack_plan3(graph,m); p['core_schedules'] += [[] for _ in range(ncores-m)]; return p,meta
        test(f'wcc_active{_m}',_wcc_active)
    test('strong_b10',lambda:generate_safe_packed(graph,ncores,mode='strong_safe',bins_per_level=2*ncores,work_factor=1.5,strong_quantile=.5,cross_wait=1000,same_wait=100))
    test('aggr_b15',lambda:generate_safe_packed(graph,ncores,mode='aggressive_safe',bins_per_level=3*ncores,work_factor=1.5,strong_quantile=.5,cross_wait=1000,same_wait=100))
    test('aggr_ultrafine20',lambda:generate_safe_packed(graph,ncores,mode='aggressive_safe',max_ops=80,work_factor=.75,strong_quantile=.05,bins_per_level=4*ncores,cross_wait=1000,same_wait=100))
    test('aggr_fine15',lambda:generate_safe_packed(graph,ncores,mode='aggressive_safe',max_ops=96,work_factor=1.0,strong_quantile=.10,bins_per_level=3*ncores,cross_wait=1000,same_wait=100))
    test('aggr_mid12',lambda:generate_safe_packed(graph,ncores,mode='aggressive_safe',max_ops=128,work_factor=1.0,strong_quantile=.25,bins_per_level=max(ncores,12),cross_wait=1000,same_wait=100))
    # v4.2 candidates learned from hard-graph ablation: dense branching, long-narrow, mixed giant-WCC.
    test('aggr_dense6k',lambda:generate_safe_packed(graph,ncores,mode='aggressive_safe',max_ops=96,work_factor=.90,strong_quantile=.0,bins_per_level=6*ncores,cross_wait=1000,same_wait=100))
    test('aggr_mixed4k',lambda:generate_safe_packed(graph,ncores,mode='aggressive_safe',max_ops=160,work_factor=.75,strong_quantile=.15,bins_per_level=4*ncores,cross_wait=1000,same_wait=100))
    test('strong_long8',lambda:generate_safe_packed(graph,ncores,mode='strong_safe',max_ops=1024,work_factor=1.5,strong_quantile=.5,bins_per_level=max(ncores,8),cross_wait=1000,same_wait=100))
    test('strong_compact12',lambda:generate_safe_packed(graph,ncores,mode='strong_safe',max_ops=128,work_factor=.75,strong_quantile=.40,bins_per_level=max(ncores,12),cross_wait=1000,same_wait=100))
    if opn<=10000:
        test('corephase_bal',lambda:core_greedy_phase_plan(graph,ncores,stick=.2,cap=1.10,refine_passes=1))
        test('corephase_sticky',lambda:core_greedy_phase_plan(graph,ncores,stick=3.0,cap=1.5,refine_passes=1))
    if opn<=6000:
        try:
            _,_,nodes,preds,succs,_=build_views(graph);seen=set();nwcc=0
            for u in nodes:
                if u in seen:continue
                nwcc+=1;st=[u];seen.add(u)
                while st:
                    x=st.pop()
                    for y in preds[x]|succs[x]:
                        if y not in seen:seen.add(y);st.append(y)
            if nwcc==1:
                test('frontier6',lambda:frontier_grow_plan(graph,ncores,waves=6,comm_alpha=5.0,rank_alpha=.02,max_ops=1024,min_fill=.35,cross_wait=1000,same_wait=100))
        except Exception: pass

    # v4 stage A: evaluator-calibrated re-assignment of the best fixed partition.
    seed_plan,seed_res=best['plan'],best['res']
    if best['meta'].get('num_subgraphs',1)>1:
        for alpha in (0.1,1.0,2.0,3.0):
            t=time.time()
            try:
                p=exact_duration_reassign(graph,seed_plan,seed_res,ncores,alpha,
                                          ev['cross_core_wait'],ev['same_core_wait'])
                r=evaluate_scene_a(graph,p,**ev)
                records.append({'name':f'exact_reassign_a{alpha:g}','makespan':r['makespan'],
                                'added_copy_bytes':r['data_movement_bytes']['added_copy_bytes'],
                                'subgraphs':len(set(p['node_to_subgraph'].values())),'seconds':time.time()-t})
                if r['makespan']<best['ms']:
                    best={'ms':r['makespan'],'name':f'{best["name"]}+reassign{alpha:g}','plan':p,'res':r,'meta':{'num_subgraphs':len(set(p['node_to_subgraph'].values()))}}
            except Exception as e:
                records.append({'name':f'exact_reassign_a{alpha:g}','error':repr(e),'seconds':time.time()-t})

    # v4 stage B: safe heavy-edge merging only for hard cases; two accepted rounds max.
    single_ms=br['makespan']; speed=single_ms/best['ms']
    if opn<=7000 and speed<3.2 and len(set(best['plan']['node_to_subgraph'].values()))<=500:
        for rnd in range(2):
            groups,pred,suc,eb,oldcore=_quotient(graph,best['plan'])
            cand=[]
            for (a,b),bytes_ in eb.items():
                if not (len(suc[a])==1 or len(pred[b])==1): continue
                same=(oldcore.get(a)==oldcore.get(b))
                score=2.0*bytes_/60.0 + (150.0 if same else -800.0)
                cand.append((score,a,b,bytes_))
            improved=None
            for _,a,b,bytes_ in sorted(cand,reverse=True)[:5]:
                p=safe_merge_pair(graph,best['plan'],a,b)
                if p is None: continue
                try:
                    r=evaluate_scene_a(graph,p,**ev)
                except Exception: continue
                records.append({'name':f'merge_r{rnd}_{a}_{b}','makespan':r['makespan'],
                                'added_copy_bytes':r['data_movement_bytes']['added_copy_bytes'],
                                'subgraphs':len(set(p['node_to_subgraph'].values()))})
                if r['makespan']<best['ms'] and (improved is None or r['makespan']<improved[0]):
                    improved=(r['makespan'],p,r,a,b)
            if improved is None: break
            best={'ms':improved[0],'name':best['name']+f'+merge({improved[3]},{improved[4]})',
                  'plan':improved[1],'res':improved[2],
                  'meta':{'num_subgraphs':len(set(improved[1]['node_to_subgraph'].values()))}}
            # one exact reassignment after each accepted merge
            try:
                p=exact_duration_reassign(graph,best['plan'],best['res'],ncores,2.0,
                                          ev['cross_core_wait'],ev['same_core_wait'])
                r=evaluate_scene_a(graph,p,**ev)
                if r['makespan']<best['ms']:
                    best={'ms':r['makespan'],'name':best['name']+'+reassign2','plan':p,'res':r,
                          'meta':{'num_subgraphs':len(set(p['node_to_subgraph'].values()))}}
            except Exception: pass

    return finish()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('graph');ap.add_argument('-n','--num-cores',type=int,required=True)
    ap.add_argument('--config',required=True);ap.add_argument('-o','--output',required=True);a=ap.parse_args()
    g=json.load(open(a.graph,encoding='utf-8'));p,info=solve(g,a.num_cores,a.config)
    json.dump(p,open(a.output,'w',encoding='utf-8'),ensure_ascii=False,separators=(',',':'))
    print(json.dumps(info,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
