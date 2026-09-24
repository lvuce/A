#!/usr/bin/env python3
import json,sys,math,heapq,time
from collections import defaultdict,deque
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent / '通用神经网络处理器下的多核调度问题  附件' / 'code'
if not CODE_DIR.is_dir():
    raise FileNotFoundError(f'Official evaluator directory is missing: {CODE_DIR}')
sys.path.insert(0,str(CODE_DIR))
from multicore_cut_evaluate_problem_1 import evaluate_scene_a, read_scene_a_config
from singlecore_evaluate import evaluate_singlecore, build_singlecore_plan
from evaluation_validation import read_evaluation_config
BW=60.0
EX={'COPY_IN','COPY_OUT'}

def build_views(g):
    op={o['id']:o for o in g['ops']}; ids=set(op)
    prod=defaultdict(set); cons=defaultdict(set); direct=[]
    for e in g['edges']:
        a,b=e['source'],e['target']; ai=a in ids; bi=b in ids
        if ai and not bi: prod[b].add(a)
        elif not ai and bi: cons[a].add(b)
        elif ai and bi and a!=b: direct.append((a,b))
    nodes=[u for u,o in op.items() if o['op'] not in EX]; elig=set(nodes)
    s0={u:set() for u in ids}
    for a,b in direct:s0[a].add(b)
    for t,ps in prod.items():
        for a in ps:
            for b in cons.get(t,()):
                if a!=b:s0[a].add(b)
    pred={u:set() for u in nodes}; suc={u:set() for u in nodes}
    for src in nodes:
        st=list(s0[src]); seen=set()
        while st:
            v=st.pop()
            if v in elig:
                if v!=src:suc[src].add(v);pred[v].add(src)
            elif v not in seen:
                seen.add(v);st.extend(s0.get(v,()))
    tby={t['id']:t for t in g['tensors']}; eb=defaultdict(int)
    for tid,t in tby.items():
        sz=int(t.get('size',0))
        for a in prod.get(tid,()):
            if a not in elig:continue
            for b in cons.get(tid,()):
                if b in elig and a!=b: eb[(a,b)]+=sz
    for u in nodes:
        for v in suc[u]:eb[(u,v)]+=0
    return op,nodes,pred,suc,eb,prod,cons,tby

def topo(nodes,pred,suc):
    indeg={u:len(pred[u]) for u in nodes};q=[u for u in nodes if indeg[u]==0];heapq.heapify(q);out=[]
    while q:
        u=heapq.heappop(q);out.append(u)
        for v in suc[u]:
            indeg[v]-=1
            if indeg[v]==0:heapq.heappush(q,v)
    return out

def quotient(clusters,nodes,suc,eb):
    cid={u:i for i,c in enumerate(clusters) for u in c}; S=len(clusters)
    cp=[set() for _ in range(S)];cs=[set() for _ in range(S)];e2=defaultdict(int)
    for u in nodes:
        a=cid[u]
        for v in suc[u]:
            b=cid[v]
            if a!=b: cs[a].add(b);cp[b].add(a);e2[(a,b)]+=eb.get((u,v),0)
    return cp,cs,e2

def chain_clusters(g,K,max_ops=256,work_factor=1.5,mode='none',q=.5):
    op,nodes,pred,suc,eb,*_=build_views(g); parent={u:u for u in nodes}; members={u:[u] for u in nodes}; work={u:max(1,op[u].get('cycles',0)) for u in nodes}
    total=sum(work.values()); cap=max(500,total*work_factor/K)
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def merge(a,b):
        a,b=find(a),find(b)
        if a==b:return False
        if len(members[a])+len(members[b])>max_ops or work[a]+work[b]>cap:return False
        parent[b]=a;members[a]+=members[b];work[a]+=work[b];return True
    # exclusive serial chains
    edges=[]
    for u in nodes:
        for v in suc[u]:
            if len(suc[u])==1 and len(pred[v])==1:edges.append((eb.get((u,v),0),u,v))
    for _,u,v in sorted(edges,reverse=True):merge(u,v)
    clusters=list({find(u) for u in nodes}); clusters=[members[r] for r in clusters]
    # optional safe quotient heavy-edge merging where source outdeg1 or target indeg1
    for _ in range(20 if mode!='none' else 0):
        cp,cs,e2=quotient(clusters,nodes,suc,eb)
        vals=sorted(x for x in e2.values() if x>0);thr=vals[int(q*(len(vals)-1))] if vals else 10**30
        cand=[]
        # communication/computation ratio, matching the safer v4 logic
        cwork=[sum(max(1,op[u].get('cycles',0)) for u in c) for c in clusters]
        for (a,b),x in e2.items():
            if not (len(cs[a])==1 or len(cp[b])==1):
                continue
            comm=2.0*x/BW; comp=max(1.0,min(cwork[a],cwork[b])); ratio=comm/comp
            ok = (x>=thr and ratio>=0.08) if mode=='strong' else (x>=thr or ratio>=0.22)
            if ok:
                cand.append((comm*(1.0+ratio),a,b))
        if not cand:break
        used=set(); pairs=[]
        for _,a,b in sorted(cand,reverse=True):
            if a in used or b in used:continue
            wa=sum(max(1,op[u].get('cycles',0)) for u in clusters[a]);wb=sum(max(1,op[u].get('cycles',0)) for u in clusters[b])
            if len(clusters[a])+len(clusters[b])<=max_ops and wa+wb<=cap:
                used|={a,b};pairs.append((a,b))
        if not pairs:break
        mate={};
        for a,b in pairs:mate[a]=b;mate[b]=a
        new=[];seen=set()
        for i,c in enumerate(clusters):
            if i in seen:continue
            if i in mate:
                j=mate[i];new.append(c+clusters[j]);seen|={i,j}
            else:new.append(c);seen.add(i)
        clusters=new
    # topo relabel quotient
    cp,cs,e2=quotient(clusters,nodes,suc,eb); indeg=[len(x) for x in cp];q0=deque(i for i,d in enumerate(indeg) if d==0);order=[]
    while q0:
        a=q0.popleft();order.append(a)
        for b in cs[a]:
            indeg[b]-=1
            if indeg[b]==0:q0.append(b)
    clusters=[clusters[i] for i in order]
    return op,nodes,pred,suc,eb,clusters

def level_pack(clusters,op,nodes,suc,eb,K,bins):
    cp,cs,e2=quotient(clusters,nodes,suc,eb); S=len(clusters); lev=[0]*S
    order=[];ind=[len(x) for x in cp];q=deque(i for i,d in enumerate(ind) if d==0)
    while q:
        a=q.popleft();order.append(a)
        for b in cs[a]: ind[b]-=1; lev[b]=max(lev[b],lev[a]+1); q.append(b) if ind[b]==0 else None
    M=[];V=[]
    for c in clusters:
        M.append(sum(op[u].get('cycles',0) for u in c if op[u].get('pipe')=='PIPE_M'))
        V.append(sum(op[u].get('cycles',0) for u in c if op[u].get('pipe')=='PIPE_V'))
    groups=[]
    for l in range(max(lev,default=-1)+1):
        items=[i for i in range(S) if lev[i]==l]; bcnt=min(len(items),bins)
        if not items:continue
        B=[[] for _ in range(bcnt)];bm=[0]*bcnt;bv=[0]*bcnt
        for i in sorted(items,key=lambda i:max(M[i],V[i]),reverse=True):
            k=min(range(bcnt),key=lambda k:max(bm[k]+M[i],bv[k]+V[i]));B[k].append(i);bm[k]+=M[i];bv[k]+=V[i]
        for b in B:
            if b:groups.append(sum((clusters[i] for i in b),[]))
    return groups

def heft_plan(g,clusters,K):
    op,nodes,pred,suc,eb,*_=build_views(g); cp,cs,e2=quotient(clusters,nodes,suc,eb); S=len(clusters)
    M=[];V=[]
    for c in clusters:
        M.append(sum(op[u].get('cycles',0) for u in c if op[u].get('pipe')=='PIPE_M'))
        V.append(sum(op[u].get('cycles',0) for u in c if op[u].get('pipe')=='PIPE_V'))
    dur=[max(M[i],V[i],.62*(M[i]+V[i])) for i in range(S)];rank=[0]*S
    # topo quotient
    ind=[len(x) for x in cp];q=deque(i for i,d in enumerate(ind) if d==0);to=[]
    while q:
        a=q.popleft();to.append(a)
        for b in cs[a]:ind[b]-=1;q.append(b) if ind[b]==0 else None
    for i in reversed(to):rank[i]=dur[i]+max((550+rank[j] for j in cs[i]),default=0)
    ind=[len(x) for x in cp];ready={i for i,d in enumerate(ind) if d==0};av=[0]*K;sched=[[] for _ in range(K)];ass={};fin={}
    while ready:
        i=max(ready,key=lambda x:(rank[x],dur[x],-x));ready.remove(i);best=None
        for k in range(K):
            est=av[k]+(100 if sched[k] else 0)
            for p in cp[i]:est=max(est,fin[p]+(0 if ass[p]==k else 1000))
            ft=est+dur[i];key=(ft,av[k],k)
            if best is None or key<best[0]:best=(key,k,ft)
        _,k,ft=best;ass[i]=k;fin[i]=ft;av[k]=ft;sched[k].append(i)
        for j in cs[i]:ind[j]-=1;ready.add(j) if ind[j]==0 else None
    return {'node_to_subgraph':{str(u):i for i,c in enumerate(clusters) for u in c},'core_schedules':sched}

def wcc_plan(g,K):
    op,nodes,pred,suc,eb,prod,cons,tby=build_views(g);seen=set();comps=[];comp_of={}
    for u in nodes:
        if u in seen:continue
        st=[u];seen.add(u);cc=[]
        while st:
            x=st.pop();cc.append(x)
            for y in pred[x]|suc[x]:
                if y not in seen:seen.add(y);st.append(y)
        ci=len(comps)
        for x in cc:comp_of[x]=ci
        comps.append(cc)
    # estimate original DDR traffic per component, because all cores share the 60 B/cycle bus
    ddr=[0.0]*len(comps)
    for tid,t in tby.items():
        sz=float(t.get('size',0))
        has_in=any(op[p].get('op')=='COPY_IN' for p in prod.get(tid,()) if p in op)
        has_out=any(op[c].get('op')=='COPY_OUT' for c in cons.get(tid,()) if c in op)
        if has_in:
            for ci in {comp_of[c] for c in cons.get(tid,()) if c in comp_of}: ddr[ci]+=sz
        if has_out:
            for ci in {comp_of[p] for p in prod.get(tid,()) if p in comp_of}: ddr[ci]+=sz
    b=min(K,len(comps));groups=[[] for _ in range(b)];lm=[0.]*b;lv=[0.]*b;ld=[0.]*b
    stats=[]
    for i,c in enumerate(comps):
        m=sum(op[u].get('cycles',0) for u in c if op[u].get('pipe')=='PIPE_M');v=sum(op[u].get('cycles',0) for u in c if op[u].get('pipe')=='PIPE_V');d=ddr[i]/BW
        stats.append((max(m,v,d),m,v,d,c))
    for _,m,v,d,c in sorted(stats,reverse=True,key=lambda x:x[0]):
        k=min(range(b),key=lambda k:(max(lm[k]+m,lv[k]+v,ld[k]+d),lm[k]+lv[k]+ld[k],k));groups[k]+=c;lm[k]+=m;lv[k]+=v;ld[k]+=d
    mapping={str(u):i for i,c in enumerate(groups) for u in c};sched=[[] for _ in range(K)]
    for i in range(len(groups)):sched[i].append(i)
    return {'node_to_subgraph':mapping,'core_schedules':sched},len(comps)

def frontier_plan(g,K,waves=6):
    op,nodes,pred,suc,eb,*_=build_views(g);order=topo(nodes,pred,suc);rank={}
    for u in reversed(order):rank[u]=max(1,op[u].get('cycles',0))+max((rank[v]+2*eb.get((u,v),0)/BW for v in suc[u]),default=0)
    total=sum(op[u].get('cycles',0) for u in nodes);target=max(100,total/(K*waves));left={u:len(pred[u]) for u in nodes};ready={u for u in nodes if left[u]==0};assigned=set();clusters=[]
    while ready:
        seed=max(ready,key=lambda u:rank[u]);ready.remove(seed);cur=[];curset=set();work=0;cand={seed}
        while cand:
            u=max(cand,key=lambda x:(sum(1000+2*eb.get((p,x),0)/BW for p in pred[x] if p in curset)+.02*rank[x],rank[x]));cand.remove(u)
            if u in assigned:continue
            cyc=max(1,op[u].get('cycles',0));aff=sum(eb.get((p,u),0) for p in pred[u] if p in curset)
            if cur and work+cyc>target and work>.35*target and aff<4096:ready.add(u);break
            cur.append(u);curset.add(u);assigned.add(u);work+=cyc
            for v in suc[u]:
                left[v]-=1
                if left[v]==0:ready.add(v)
            for x in list(ready):
                if any(p in curset for p in pred[x]):cand.add(x);ready.discard(x)
        ready.update(x for x in cand if x not in assigned);clusters.append(cur)
    return heft_plan(g,clusters,K)


def clusterphase_plan(g,K,max_ops=128,work_factor=1.0,mode='strong',q=.4,stick=.1,cap=1.3,refine=2):
    """v6: coarsen serial/communication-heavy regions first, then assign coarse clusters
    directly to cores with communication-aware load balancing. Finally collapse equal
    (cross-core-depth, core) clusters into Tasks. This reduces the number of 1000-cycle
    cross-core waits on wide single-WCC DAGs compared with pure level packing.
    """
    op,nodes,pred,suc,eb,clusters=chain_clusters(g,K,max_ops,work_factor,mode,q)
    cp,cs,e2=quotient(clusters,nodes,suc,eb);S=len(clusters)
    # quotient topological order
    indeg=[len(x) for x in cp];hq=[i for i,d in enumerate(indeg) if d==0];heapq.heapify(hq);order=[]
    while hq:
        a=heapq.heappop(hq);order.append(a)
        for b in cs[a]:
            indeg[b]-=1
            if indeg[b]==0:heapq.heappush(hq,b)
    if len(order)!=S: raise RuntimeError('cluster quotient cycle')
    M=[];V=[]
    for c in clusters:
        M.append(sum(op[u].get('cycles',0) for u in c if op[u].get('pipe')=='PIPE_M'))
        V.append(sum(op[u].get('cycles',0) for u in c if op[u].get('pipe')=='PIPE_V'))
    TM=sum(M);TV=sum(V);tM=max(1.0,TM/K);tV=max(1.0,TV/K)
    # critical rank on coarse DAG
    rank=[0.0]*S
    for a in reversed(order):
        local=max(M[a],V[a],.62*(M[a]+V[a]))
        rank[a]=local+max((1000+2*e2.get((a,b),0)/BW+rank[b] for b in cs[a]),default=0.0)
    lm=[0.0]*K;lv=[0.0]*K;ass={};indeg=[len(x) for x in cp];ready={i for i,d in enumerate(indeg) if d==0};seq=[]
    while ready:
        a=max(ready,key=lambda i:(rank[i],max(M[i],V[i]),-i));ready.remove(a);seq.append(a)
        best=None
        for k in range(K):
            aff=sum(1000+2*e2.get((p,a),0)/BW for p in cp[a] if ass.get(p)==k)
            nm=lm[k]+M[a];nv=lv[k]+V[a];ratio=max(nm/tM,nv/tV);over=max(0.0,ratio-cap)
            score=stick*aff-900*ratio-5000*over
            key=(score,-ratio,-k)
            if best is None or key>best[0]:best=(key,k)
        k=best[1];ass[a]=k;lm[k]+=M[a];lv[k]+=V[a]
        for b in cs[a]:
            indeg[b]-=1
            if indeg[b]==0:ready.add(b)
    # small label-propagation refinement on the coarse graph
    for it in range(refine):
        changed=0;walk=seq if it%2==0 else list(reversed(seq))
        for a in walk:
            old=ass[a]
            def affinity(k):
                z=0.0
                for b in cp[a]:
                    if ass[b]==k:z+=1000+2*e2.get((b,a),0)/BW
                for b in cs[a]:
                    if ass[b]==k:z+=1000+2*e2.get((a,b),0)/BW
                return z
            oldaff=affinity(old);choice=(0.0,old)
            for k in range(K):
                if k==old:continue
                nm=lm[k]+M[a];nv=lv[k]+V[a]
                if max(nm/tM,nv/tV)>cap:continue
                before=max(lm[old]/tM,lv[old]/tV)+max(lm[k]/tM,lv[k]/tV)
                after=max((lm[old]-M[a])/tM,(lv[old]-V[a])/tV)+max(nm/tM,nv/tV)
                gain=stick*(affinity(k)-oldaff)-500*(after-before)
                if gain>choice[0]:choice=(gain,k)
            if choice[1]!=old:
                k=choice[1];lm[old]-=M[a];lv[old]-=V[a];lm[k]+=M[a];lv[k]+=V[a];ass[a]=k;changed+=1
        if not changed:break
    # Cross-core-depth phase: every cross-core edge advances one phase, so quotient remains acyclic.
    phase={}
    for a in order:phase[a]=max((phase[p]+(1 if ass[p]!=ass[a] else 0) for p in cp[a]),default=0)
    groups=defaultdict(list)
    for a in order:groups[(phase[a],ass[a])].extend(clusters[a])
    keys=sorted(groups);sid={x:i for i,x in enumerate(keys)};sched=[[] for _ in range(K)]
    for x in keys:sched[x[1]].append(sid[x])
    return {'node_to_subgraph':{str(u):sid[(phase[a],ass[a])] for a in range(S) for u in clusters[a]},'core_schedules':sched}


def v7_active3_strong96(g,K):
    # Deep coarsening with only three active cores. This deliberately trades some raw
    # parallelism for much fewer DDR cuts / 1000-cycle cross-core waits.
    active=min(3,K)
    op,nodes,pred,suc,eb,cl=chain_clusters(g,active,96,3.0,'strong',.25)
    groups=level_pack(cl,op,nodes,suc,eb,active,3)
    p=heft_plan(g,groups,active)
    p['core_schedules'] += [[] for _ in range(K-active)]
    return p

def v7_micro64_b40(g,K):
    # Fine-grained escape candidate for medium-width DAGs; a low work cap exposes
    # more independent work, while level packing still preserves acyclicity.
    op,nodes,pred,suc,eb,cl=chain_clusters(g,K,64,.4,'aggressive',0.0)
    groups=level_pack(cl,op,nodes,suc,eb,K,40)
    return heft_plan(g,groups,K)

def eval_args(config=None):
    if config is None:
        return dict(bandwidth=60,capacity={'L1':524288,'UB':131072},cross_core_wait=1000,same_core_wait=100)
    cfg=read_evaluation_config(config);waits=read_scene_a_config(config)
    return dict(bandwidth=cfg['bandwidth'],capacity=cfg['capacity'],
                cross_core_wait=waits['task_cross_core_wait_cycles'],
                same_core_wait=waits['task_same_core_wait_cycles'])

def solve(g,K=5,baseline_makespan=None,config=None):
    ev=eval_args(config);single=(baseline_makespan if baseline_makespan is not None else evaluate_singlecore(g,**ev)['makespan']);base=build_singlecore_plan(g);base['core_schedules'] += [[] for _ in range(K-1)];best=(single,'single',base); rec=[]
    def test(name,p):
        nonlocal best
        try:r=evaluate_scene_a(g,p,**ev);sp=single/r['makespan'];rec.append((name,sp,r['makespan'],r['data_movement_bytes']['added_copy_bytes']));
        except Exception:return
        if r['makespan']<best[0]:best=(r['makespan'],name,p)
    # WCC all active core counts
    for active in range(2,K+1):
        p,nw=wcc_plan(g,active);p['core_schedules'] += [[] for _ in range(K-active)];test(f'wcc{active}',p)
    # structural candidates
    for mode,maxops,wf,q,bins in [('none',80,.75,.05,20),('aggressive',96,.9,.1,30),('strong',128,.75,.4,12),('aggressive',160,.75,.15,20),('strong',256,1.5,.5,10),('strong',1024,1.5,.5,8)]:
        try:
            op,nodes,pred,suc,eb,cl=chain_clusters(g,K,maxops,wf,mode,q);groups=level_pack(cl,op,nodes,suc,eb,K,bins);p=heft_plan(g,groups,K);test(f'lvl_{mode}_{maxops}_{bins}',p)
        except Exception:pass
    # v7 portfolio additions learned from the 50-case hard set.
    # They are always evaluated, but can only replace the incumbent when the official
    # evaluator reports a lower makespan.
    try:test('v7_active3_strong96',v7_active3_strong96(g,K))
    except Exception:pass
    try:test('v7_micro64_b40',v7_micro64_b40(g,K))
    except Exception:pass
    # v6 coarse-cluster core partitioning: targets wide/short single-WCC DAGs where
    # level packing pays too many 1000-cycle cross-core waits.
    for mo,wf,md,qq,st in [(64,.75,'aggressive',.1,.2),(96,.9,'aggressive',.1,.1),(128,1.0,'strong',.4,.1)]:
        try:test(f'clusterphase_{md}_{mo}_{st}',clusterphase_plan(g,K,mo,wf,md,qq,st,1.3,2))
        except Exception:pass
    # dense small/medium escape
    if sum(1 for o in g['ops'] if o['op'] not in EX)<=7000:
        try:test('frontier6',frontier_plan(g,K,6))
        except Exception:pass
        for st,cp in [(0.05,1.05),(0.1,1.1),(0.2,1.15),(0.5,1.2),(1.0,1.3),(3.0,1.5)]:
            try:test(f'corephase_{st}_{cp}',corephase_plan(g,K,st,cp))
            except Exception:pass
        if sum(1 for o in g['ops'] if o['op'] not in EX)<=2500:
            for sd,cs in [(1,.1),(1,1.0),(2,1.0),(3,3.0)]:
                try:test(f'kl_{sd}_{cs}',kl_phase_plan(g,K,sd,cs))
                except Exception:pass
    # evaluator-calibrated core reassignment for the best fixed partition
    if len(set(best[2]['node_to_subgraph'].values()))>1:
        seed=best
        try:
            r0=evaluate_scene_a(g,seed[2],**ev)
            for a in [0.1,1.0,2.0,3.0]:
                q=exact_reassign(g,seed[2],r0,K,a);test(f'reassign_{a}',q)
        except Exception:pass
    return {'single':single,'best_makespan':best[0],'speedup':single/best[0],'selected':best[1],'plan':best[2],'candidates':rec}

def corephase_plan(g,K,stick=.3,cap=1.2):
    op,nodes,pred,suc,eb,*_=build_views(g);order=topo(nodes,pred,suc)
    TM=sum(op[u].get('cycles',0) for u in nodes if op[u].get('pipe')=='PIPE_M');TV=sum(op[u].get('cycles',0) for u in nodes if op[u].get('pipe')=='PIPE_V')
    tM=max(1,TM/K);tV=max(1,TV/K);lm=[0.]*K;lv=[0.]*K;ass={}
    for u in order:
        cyc=float(op[u].get('cycles',0));pipe=op[u].get('pipe');best=None
        for k in range(K):
            affinity=sum(1000+2*eb.get((p,u),0)/BW for p in pred[u] if ass.get(p)==k)
            nm=lm[k]+(cyc if pipe=='PIPE_M' else 0);nv=lv[k]+(cyc if pipe=='PIPE_V' else 0);ratio=max(nm/tM,nv/tV);over=max(0,ratio-cap)
            score=stick*affinity-900*ratio-5000*over;key=(score,-ratio,-k)
            if best is None or key>best[0]:best=(key,k)
        k=best[1];ass[u]=k
        if pipe=='PIPE_M':lm[k]+=cyc
        elif pipe=='PIPE_V':lv[k]+=cyc
    # phase increments on cross-core backward deps; this guarantees acyclic task quotient
    phase={}
    for u in order:
        phase[u]=max((phase[p]+(1 if ass[p]!=ass[u] else 0) for p in pred[u]),default=0)
    groups=defaultdict(list)
    for u in order:groups[(phase[u],ass[u])].append(u)
    keys=sorted(groups);sid={k:i for i,k in enumerate(keys)};sched=[[] for _ in range(K)]
    for k in keys:sched[k[1]].append(sid[k])
    return {'node_to_subgraph':{str(u):sid[(phase[u],ass[u])] for u in nodes},'core_schedules':sched}

def exact_reassign(g,plan,res,K,alpha=1.0):
    sgs=sorted(set(int(v) for v in plan['node_to_subgraph'].values())); ren={s:i for i,s in enumerate(sgs)};S=len(sgs)
    mapping={str(k):ren[int(v)] for k,v in plan['node_to_subgraph'].items()}
    dur=[1.0]*S
    for k,v in res['step3_by_task'].items():
        kk=ren.get(int(k));
        if kk is not None:dur[kk]=float(v['local_makespan'])
    pred=[set() for _ in range(S)];suc=[set() for _ in range(S)]
    for e in res['task_dependencies']:
        a=ren[int(e['source'])];b=ren[int(e['target'])]
        if a!=b:suc[a].add(b);pred[b].add(a)
    op,nodes,opred,osuc,eb,*_=build_views(g);nsg={int(k):int(v) for k,v in mapping.items()};e2=defaultdict(int)
    for u in nodes:
        a=nsg[u]
        for v in osuc[u]:
            b=nsg[v]
            if a!=b:e2[(a,b)]+=eb.get((u,v),0)
    ind=[len(x) for x in pred];q=[i for i,d in enumerate(ind) if d==0];heapq.heapify(q);to=[]
    while q:
        a=heapq.heappop(q);to.append(a)
        for b in suc[a]:ind[b]-=1;heapq.heappush(q,b) if ind[b]==0 else None
    rank=[0.]*S
    for a in reversed(to):rank[a]=dur[a]+max((1000+alpha*2*e2.get((a,b),0)/60+rank[b] for b in suc[a]),default=0)
    ind=[len(x) for x in pred];ready={i for i,d in enumerate(ind) if d==0};av=[0.]*K;sched=[[] for _ in range(K)];ass={};fin={}
    while ready:
        a=max(ready,key=lambda x:(rank[x],dur[x],-x));ready.remove(a);best=None
        for k in range(K):
            est=av[k]+(100 if sched[k] else 0);cross=0
            for p in pred[a]:
                if ass[p]==k:est=max(est,fin[p])
                else:est=max(est,fin[p]+1000);cross+=1
            ft=est+dur[a];key=(ft,cross,av[k],k)
            if best is None or key<best[0]:best=(key,k,ft)
        _,k,ft=best;ass[a]=k;fin[a]=ft;av[k]=ft;sched[k].append(a)
        for b in suc[a]:ind[b]-=1;ready.add(b) if ind[b]==0 else None
    return {'node_to_subgraph':mapping,'core_schedules':sched}

def kl_phase_plan(g,K,seed=1,comm_scale=1.0):
    import networkx as nx
    op,nodes,pred,suc,eb,*_=build_views(g)
    UG=nx.Graph();UG.add_nodes_from(nodes)
    for u in nodes:
        for v in suc[u]:
            w=1000.0+comm_scale*2.0*eb.get((u,v),0)/BW
            if UG.has_edge(u,v):UG[u][v]['weight']+=w
            else:UG.add_edge(u,v,weight=w)
    parts=[set(nodes)]
    def work(S):
        m=sum(op[u].get('cycles',0) for u in S if op[u].get('pipe')=='PIPE_M');v=sum(op[u].get('cycles',0) for u in S if op[u].get('pipe')=='PIPE_V')
        return max(m,v,.62*(m+v))
    while len(parts)<K:
        i=max(range(len(parts)),key=lambda i:work(parts[i]));S=parts.pop(i)
        if len(S)<2:parts.append(S);break
        try:a,b=nx.community.kernighan_lin_bisection(UG.subgraph(S),weight='weight',seed=seed,max_iter=30)
        except Exception:
            L=list(S);a=set(L[:len(L)//2]);b=set(L[len(L)//2:])
        parts += [set(a),set(b)]
    ass={u:k for k,S in enumerate(parts) for u in S};order=topo(nodes,pred,suc);phase={}
    for u in order:phase[u]=max((phase[p]+(1 if ass[p]!=ass[u] else 0) for p in pred[u]),default=0)
    groups=defaultdict(list)
    for u in order:groups[(phase[u],ass[u])].append(u)
    keys=sorted(groups);sid={k:i for i,k in enumerate(keys)};sched=[[] for _ in range(K)]
    for k in keys:sched[k[1]].append(sid[k])
    return {'node_to_subgraph':{str(u):sid[(phase[u],ass[u])] for u in nodes},'core_schedules':sched}

if __name__=='__main__':
    g=json.load(open(sys.argv[1]));r=solve(g,int(sys.argv[2]) if len(sys.argv)>2 else 5);print(json.dumps({k:v for k,v in r.items() if k!='plan'},ensure_ascii=False,indent=2))
