#!/usr/bin/env python3
import json, math
from collections import defaultdict, deque

EXCLUDED={'COPY_IN','COPY_OUT'}
BW=60.0

class DSU:
    def __init__(self, nodes):
        self.p={x:x for x in nodes}; self.sz={x:1 for x in nodes}
    def find(self,x):
        p=self.p[x]
        if p!=x:
            self.p[x]=self.find(p)
        return self.p[x]
    def union(self,a,b):
        a=self.find(a); b=self.find(b)
        if a==b:return a
        if self.sz[a]<self.sz[b]: a,b=b,a
        self.p[b]=a; self.sz[a]+=self.sz[b]
        return a

def build_views(g):
    op_by={o['id']:o for o in g['ops']}
    op_ids=set(op_by)
    tensor_by={t['id']:t for t in g['tensors']}
    producers=defaultdict(set); consumers=defaultdict(set); direct=[]
    for e in g['edges']:
        a,b=e['source'],e['target']
        ao=a in op_ids; bo=b in op_ids
        if ao and not bo: producers[b].add(a)
        elif (not ao) and bo: consumers[a].add(b)
        elif ao and bo and a!=b: direct.append((a,b))
    nodes=[i for i,o in op_by.items() if o['op'] not in EXCLUDED]
    elig=set(nodes)
    # Full op graph then contract COPY nodes.
    preds0={i:set() for i in op_ids}; succs0={i:set() for i in op_ids}
    for a,b in direct: succs0[a].add(b); preds0[b].add(a)
    for tid,ps in producers.items():
        for a in ps:
            for b in consumers.get(tid,()):
                if a!=b: succs0[a].add(b); preds0[b].add(a)
    preds={i:set() for i in nodes}; succs={i:set() for i in nodes}
    for src in nodes:
        stack=list(succs0[src]); seen=set()
        while stack:
            v=stack.pop()
            if v in elig:
                if v!=src: succs[src].add(v); preds[v].add(src)
            elif v not in seen:
                seen.add(v); stack.extend(succs0.get(v,()))
    edge_bytes=defaultdict(int)
    # only tensors directly connecting eligible producer/consumer; COPY hops typically DDR boundaries and don't matter as internal affinity
    for tid,t in tensor_by.items():
        ps=[p for p in producers.get(tid,()) if p in elig]
        cs=[c for c in consumers.get(tid,()) if c in elig]
        s=int(t.get('size',0))
        for p in ps:
            for c in cs:
                if p!=c: edge_bytes[(p,c)] += s
    # Preserve dependency even if no direct internal tensor by assigning 0-byte edge.
    for u in nodes:
        for v in succs[u]: edge_bytes[(u,v)] += 0
    return op_by,tensor_by,nodes,preds,succs,edge_bytes

def topo(nodes,preds,succs):
    indeg={u:len(preds[u]) for u in nodes}
    q=deque(sorted(u for u in nodes if indeg[u]==0)); out=[]
    while q:
        u=q.popleft(); out.append(u)
        for v in sorted(succs[u]):
            indeg[v]-=1
            if indeg[v]==0:q.append(v)
    if len(out)!=len(nodes): raise RuntimeError('cycle')
    return out

def cluster_graph(g, num_cores, mode='chain', max_ops=256, work_factor=3.0, strong_quantile=0.75):
    op_by,tensor_by,nodes,preds,succs,edge_bytes=build_views(g)
    dsu=DSU(nodes)
    work={u:max(1,int(op_by[u].get('cycles',0))) for u in nodes}
    M={u:(work[u] if op_by[u].get('pipe')=='PIPE_M' else 0) for u in nodes}
    V={u:(work[u] if op_by[u].get('pipe')=='PIPE_V' else 0) for u in nodes}
    nops={u:1 for u in nodes}
    total_work=sum(work.values())
    # A generous cap: prevent a single task from consuming too much of ideal per-core work.
    work_cap=max(1000.0, work_factor*total_work/max(1,num_cores))
    pos_bytes=[b for b in edge_bytes.values() if b>0]
    if pos_bytes:
        sb=sorted(pos_bytes); qidx=min(len(sb)-1,max(0,int(strong_quantile*(len(sb)-1))))
        byte_thr=sb[qidx]
    else: byte_thr=10**30

    def root_stats(r):
        r=dsu.find(r); return nops[r],work[r],M[r],V[r]
    def can_merge(a,b):
        ra,rb=dsu.find(a),dsu.find(b)
        if ra==rb:return False
        return nops[ra]+nops[rb] <= max_ops and work[ra]+work[rb] <= work_cap
    def merge(a,b):
        ra,rb=dsu.find(a),dsu.find(b)
        if ra==rb:return
        na=nops[ra]+nops[rb]; ww=work[ra]+work[rb]; mm=M[ra]+M[rb]; vv=V[ra]+V[rb]
        r=dsu.union(ra,rb)
        nops[r]=na; work[r]=ww; M[r]=mm; V[r]=vv

    # Stage 1: merge strict serial-chain edges first. This preserves branch-level parallelism.
    candidates=[]
    for (u,v),b in edge_bytes.items():
        if v not in succs[u]: continue
        exclusive=(len(succs[u])==1 and len(preds[v])==1)
        if exclusive:
            score=1e18 + b
            candidates.append((score,u,v,b,True))
    for _,u,v,b,_ in sorted(candidates,reverse=True):
        if can_merge(u,v): merge(u,v)

    if mode!='chain':
        # Stage 2: merge only expensive cut edges. Ratio favors communication-heavy, low-compute pairs.
        cand=[]
        for (u,v),b in edge_bytes.items():
            if b<=0: continue
            comm=2.0*b/BW
            comp=max(1.0,min(work[u],work[v]))
            ratio=comm/comp
            if mode=='strong':
                ok=(b>=byte_thr and ratio>=0.10)
            elif mode=='aggressive':
                ok=(b>=byte_thr or ratio>=0.35)
            elif mode=='very_aggressive':
                ok=(b>=min(byte_thr,4096) or ratio>=0.15)
            else: ok=False
            if ok:
                # exclusive already merged; prefer huge bytes and high ratio
                cand.append((comm*(1.0+ratio),u,v,b))
        for _,u,v,b in sorted(cand,reverse=True):
            if can_merge(u,v): merge(u,v)

    # Collect clusters.
    groups=defaultdict(list)
    for u in nodes: groups[dsu.find(u)].append(u)
    clusters=list(groups.values())
    # quotient dependencies
    cid={u:i for i,ch in enumerate(clusters) for u in ch}
    cp=[set() for _ in clusters]; cs=[set() for _ in clusters]; eb=defaultdict(int)
    for u in nodes:
        a=cid[u]
        for v in succs[u]:
            b=cid[v]
            if a!=b:
                cs[a].add(b); cp[b].add(a); eb[(a,b)]+=edge_bytes.get((u,v),0)
    # Edge contraction guarantees DAG. Topological relabel for stable evaluator schedules.
    indeg=[len(x) for x in cp]; q=deque(sorted(i for i,x in enumerate(indeg) if x==0)); ctop=[]
    while q:
        a=q.popleft(); ctop.append(a)
        for b in sorted(cs[a]):
            indeg[b]-=1
            if indeg[b]==0:q.append(b)
    if len(ctop)!=len(clusters): raise RuntimeError('quotient cycle')
    old2new={old:new for new,old in enumerate(ctop)}
    clusters2=[clusters[old] for old in ctop]
    cp2=[set() for _ in clusters2]; cs2=[set() for _ in clusters2]; eb2=defaultdict(int)
    for (a,b),x in eb.items():
        na,nb=old2new[a],old2new[b]; cs2[na].add(nb); cp2[nb].add(na); eb2[(na,nb)]+=x
    return op_by,nodes,preds,succs,edge_bytes,clusters2,cp2,cs2,eb2

def task_weights(clusters,op_by):
    M=[];V=[];O=[]
    for ch in clusters:
        m=v=o=0.0
        for u in ch:
            c=float(op_by[u].get('cycles',0)); p=op_by[u].get('pipe')
            if p=='PIPE_M':m+=c
            elif p=='PIPE_V':v+=c
            else:o+=c
        M.append(m);V.append(v);O.append(o)
    return M,V,O

def assign_heft(clusters, cp, cs, eb, op_by, num_cores, local_durations=None, cross_wait=1000.0, same_wait=100.0):
    S=len(clusters); M,V,O=task_weights(clusters,op_by)
    if local_durations is None:
        dur=[max(M[i],V[i],0.62*(M[i]+V[i]))+O[i] for i in range(S)]
    else: dur=list(map(float,local_durations))
    # upward ranks; average core relation wait acts as priority only
    rank=[0.0]*S
    for i in reversed(range(S)):
        if cs[i]:
            rank[i]=dur[i]+max(550.0+rank[j] for j in cs[i])
        else: rank[i]=dur[i]
    # Ready-list HEFT: only schedule a task after all predecessors assigned.
    indeg=[len(cp[i]) for i in range(S)]
    ready={i for i in range(S) if indeg[i]==0}
    core_av=[0.0]*num_cores; sched=[[] for _ in range(num_cores)]
    coreM=[0.0]*num_cores; coreV=[0.0]*num_cores
    assigned={}; finish={}
    while ready:
        s=max(ready,key=lambda i:(rank[i],dur[i],-i)); ready.remove(s)
        best=None
        for k in range(num_cores):
            est=core_av[k] + (same_wait if sched[k] else 0.0)
            cross_cnt=0
            for p in cp[s]:
                if assigned[p]==k:
                    est=max(est,finish[p])
                else:
                    est=max(est,finish[p]+cross_wait); cross_cnt+=1
            ft=est+dur[s]
            # Soft balance of M/V work, but finish time dominates.
            bal=max(coreM[k]+M[s],coreV[k]+V[s])
            key=(ft,0.002*bal,cross_cnt,k)
            if best is None or key<best[0]: best=(key,k,ft)
        _,k,ft=best
        assigned[s]=k; finish[s]=ft; core_av[k]=ft; coreM[k]+=M[s]; coreV[k]+=V[s]; sched[k].append(s)
        for t in cs[s]:
            indeg[t]-=1
            if indeg[t]==0: ready.add(t)
    # Need each core schedule to respect task deps; global assignment ready-list + per-core append guarantees it.
    mapping={str(u):i for i,ch in enumerate(clusters) for u in ch}
    return {'node_to_subgraph':mapping,'core_schedules':sched}

def generate(g,num_cores,mode='chain',max_ops=256,work_factor=3.0,strong_quantile=0.75,
             local_durations=None,cross_wait=1000.0,same_wait=100.0):
    op_by,nodes,preds,succs,edge_bytes,clusters,cp,cs,eb=cluster_graph(
        g,num_cores,mode,max_ops,work_factor,strong_quantile)
    plan=assign_heft(clusters,cp,cs,eb,op_by,num_cores,local_durations,cross_wait,same_wait)
    return plan,{'num_subgraphs':len(clusters),'clusters':clusters,'cp':cp,'cs':cs,'eb':eb}

def pack_same_levels(clusters,cp,cs,eb,op_by,num_cores,bins_per_level=None):
    """Merge mutually incomparable quotient nodes that share the same longest-path level.
    This reduces Task overhead while keeping stage parallelism. Same-level nodes cannot reach one another.
    """
    S=len(clusters)
    level=[0]*S
    for i in range(S):
        if cp[i]: level[i]=1+max(level[p] for p in cp[i])
    M,V,O=task_weights(clusters,op_by)
    groups=[]
    old_to_new={}
    for lv in range(max(level,default=-1)+1):
        items=[i for i in range(S) if level[i]==lv]
        if not items: continue
        bcount=min(len(items), bins_per_level or num_cores)
        bins=[[] for _ in range(bcount)]; bm=[0.0]*bcount; bv=[0.0]*bcount; bo=[0.0]*bcount
        # LPT on dominant pipe work; pack to balance both M and V.
        items.sort(key=lambda i:max(M[i],V[i],0.65*(M[i]+V[i]))+O[i],reverse=True)
        for i in items:
            best=min(range(bcount),key=lambda b:(max(bm[b]+M[i],bv[b]+V[i])+bo[b]+O[i], bm[b]+bv[b], b))
            bins[best].append(i); bm[best]+=M[i]; bv[best]+=V[i]; bo[best]+=O[i]
        for b in bins:
            if not b: continue
            ni=len(groups); merged=[]
            for old in b:
                old_to_new[old]=ni; merged.extend(clusters[old])
            groups.append(merged)
    cp2=[set() for _ in groups]; cs2=[set() for _ in groups]; eb2=defaultdict(int)
    for (a,b),x in eb.items():
        na,nb=old_to_new[a],old_to_new[b]
        if na!=nb:
            cs2[na].add(nb); cp2[nb].add(na); eb2[(na,nb)]+=x
    # Topologically relabel because level-group creation is already increasing levels, but keep robust.
    indeg=[len(x) for x in cp2]; q=deque(i for i,x in enumerate(indeg) if x==0); order=[]
    while q:
        i=q.popleft(); order.append(i)
        for j in cs2[i]:
            indeg[j]-=1
            if indeg[j]==0:q.append(j)
    if len(order)!=len(groups): raise RuntimeError('cycle after level pack')
    rem={old:new for new,old in enumerate(order)}
    g3=[groups[old] for old in order]; cp3=[set() for _ in g3]; cs3=[set() for _ in g3]; eb3=defaultdict(int)
    for (a,b),x in eb2.items():
        na,nb=rem[a],rem[b]; cs3[na].add(nb); cp3[nb].add(na); eb3[(na,nb)]+=x
    return g3,cp3,cs3,eb3

def generate_packed(g,num_cores,mode='chain',max_ops=256,work_factor=3.0,strong_quantile=.75,bins_per_level=None,
                    cross_wait=1000.0,same_wait=100.0):
    op_by,nodes,preds,succs,edge_bytes,clusters,cp,cs,eb=cluster_graph(g,num_cores,mode,max_ops,work_factor,strong_quantile)
    clusters,cp,cs,eb=pack_same_levels(clusters,cp,cs,eb,op_by,num_cores,bins_per_level)
    plan=assign_heft(clusters,cp,cs,eb,op_by,num_cores,None,cross_wait,same_wait)
    return plan,{'num_subgraphs':len(clusters),'clusters':clusters,'cp':cp,'cs':cs,'eb':eb}


def _quotient_from_clusters(clusters,nodes,succs,edge_bytes):
    cid={u:i for i,ch in enumerate(clusters) for u in ch}
    cp=[set() for _ in clusters]; cs=[set() for _ in clusters]; eb=defaultdict(int)
    for u in nodes:
        a=cid[u]
        for v in succs[u]:
            b=cid[v]
            if a!=b:
                cs[a].add(b); cp[b].add(a); eb[(a,b)]+=edge_bytes.get((u,v),0)
    return cp,cs,eb

def safe_coarsen_after_chain(g,num_cores,mode='strong_safe',max_ops=256,work_factor=3.0,strong_quantile=.75):
    # First create chain-only clusters, then iteratively merge quotient edges where source has one successor
    # or target has one predecessor. Such edges cannot have an alternate path and their contraction preserves DAG.
    op_by,tensor_by,nodes,preds,succs,edge_bytes=build_views(g)
    # get chain clusters by original function in chain mode
    _,_,_,_,_,clusters,cp,cs,eb=cluster_graph(g,num_cores,'chain',max_ops,work_factor,strong_quantile)
    total_work=sum(max(1,int(op_by[u].get('cycles',0))) for u in nodes)
    work_cap=max(1000.0,work_factor*total_work/max(1,num_cores))
    pos=[x for x in eb.values() if x>0]
    if pos:
        ss=sorted(pos); thr=ss[min(len(ss)-1,max(0,int(strong_quantile*(len(ss)-1))))]
    else: thr=10**30
    for _round in range(100):
        M,V,O=task_weights(clusters,op_by)
        sizes=[len(ch) for ch in clusters]
        works=[M[i]+V[i]+O[i] for i in range(len(clusters))]
        cand=[]
        for (a,b),bytes_ in eb.items():
            if not (len(cs[a])==1 or len(cp[b])==1):
                continue
            if sizes[a]+sizes[b]>max_ops or works[a]+works[b]>work_cap:
                continue
            comm=2.0*bytes_/BW
            comp=max(1.0,min(works[a],works[b]))
            ratio=comm/comp
            if mode=='strong_safe': ok=(bytes_>=thr and ratio>=0.08)
            elif mode=='aggressive_safe': ok=(bytes_>=thr or ratio>=0.22)
            elif mode=='very_safe': ok=(bytes_>=min(thr,4096) or ratio>=0.10)
            else: ok=False
            if ok: cand.append((comm*(1+ratio),a,b))
        if not cand: break
        # choose a maximal disjoint merge set to keep each round simple
        used=set(); chosen=[]
        for _,a,b in sorted(cand,reverse=True):
            if a in used or b in used: continue
            used.add(a);used.add(b);chosen.append((a,b))
        if not chosen: break
        mate={}
        for a,b in chosen: mate[a]=b;mate[b]=a
        new=[]; old2new={}; seen=set()
        for i in range(len(clusters)):
            if i in seen: continue
            ni=len(new)
            if i in mate:
                j=mate[i]; seen.add(i);seen.add(j); merged=clusters[i]+clusters[j]
                old2new[i]=ni;old2new[j]=ni;new.append(merged)
            else:
                seen.add(i);old2new[i]=ni;new.append(clusters[i])
        clusters=new
        cp,cs,eb=_quotient_from_clusters(clusters,nodes,succs,edge_bytes)
        # validate DAG
        indeg=[len(x) for x in cp]; q=deque(i for i,d in enumerate(indeg) if d==0); cnt=0
        while q:
            a=q.popleft();cnt+=1
            for b in cs[a]:
                indeg[b]-=1
                if indeg[b]==0:q.append(b)
        if cnt!=len(clusters): raise RuntimeError('safe coarsen unexpectedly created cycle')
    # topo relabel
    indeg=[len(x) for x in cp]; q=deque(sorted(i for i,d in enumerate(indeg) if d==0)); order=[]
    while q:
        a=q.popleft();order.append(a)
        for b in sorted(cs[a]):
            indeg[b]-=1
            if indeg[b]==0:q.append(b)
    rem={old:new for new,old in enumerate(order)}
    c2=[clusters[i] for i in order]; cp2=[set() for _ in c2];cs2=[set() for _ in c2];eb2=defaultdict(int)
    for (a,b),x in eb.items():
        na,nb=rem[a],rem[b];cs2[na].add(nb);cp2[nb].add(na);eb2[(na,nb)]+=x
    return op_by,nodes,preds,succs,edge_bytes,c2,cp2,cs2,eb2

def generate_safe_packed(g,num_cores,mode='strong_safe',max_ops=256,work_factor=3.0,strong_quantile=.75,bins_per_level=None,
                         cross_wait=1000.0,same_wait=100.0):
    op_by,nodes,preds,succs,edge_bytes,clusters,cp,cs,eb=safe_coarsen_after_chain(g,num_cores,mode,max_ops,work_factor,strong_quantile)
    clusters,cp,cs,eb=pack_same_levels(clusters,cp,cs,eb,op_by,num_cores,bins_per_level)
    plan=assign_heft(clusters,cp,cs,eb,op_by,num_cores,None,cross_wait,same_wait)
    return plan,{'num_subgraphs':len(clusters),'clusters':clusters,'cp':cp,'cs':cs,'eb':eb}

def core_greedy_phase_plan(g,num_cores,stick=1.0,cap=1.20,refine_passes=2):
    op_by,tensor_by,nodes,preds,succs,edge_bytes=build_views(g)
    order=topo(nodes,preds,succs)
    totalM=sum(op_by[u].get('cycles',0) for u in nodes if op_by[u].get('pipe')=='PIPE_M')
    totalV=sum(op_by[u].get('cycles',0) for u in nodes if op_by[u].get('pipe')=='PIPE_V')
    tgtM=max(1,totalM/num_cores); tgtV=max(1,totalV/num_cores)
    loadM=[0.0]*num_cores; loadV=[0.0]*num_cores; assign={}
    for u in order:
        cyc=float(op_by[u].get('cycles',0)); pipe=op_by[u].get('pipe')
        rewards=[0.0]*num_cores
        for p in preds[u]:
            if p in assign:
                k=assign[p]
                rewards[k]+=1000.0+2.0*edge_bytes.get((p,u),0)/BW
        candidates=[]
        for k in range(num_cores):
            nm=loadM[k]+(cyc if pipe=='PIPE_M' else 0)
            nv=loadV[k]+(cyc if pipe=='PIPE_V' else 0)
            ratio=max(nm/tgtM,nv/tgtV)
            # Allow temporary overload only if predecessor affinity strongly favors this core.
            overload=max(0.0,ratio-cap)
            balance=max(nm/tgtM,nv/tgtV)
            score=stick*rewards[k]-900.0*balance-5000.0*overload
            candidates.append((score,-balance,-k,k))
        k=max(candidates)[3]
        assign[u]=k
        if pipe=='PIPE_M':loadM[k]+=cyc
        elif pipe=='PIPE_V':loadV[k]+=cyc

    # Local label refinement: reduce weighted cut while maintaining per-pipe balance envelope.
    for _ in range(refine_passes):
        changed=0
        for u in order:
            old=assign[u]; cyc=float(op_by[u].get('cycles',0)); pipe=op_by[u].get('pipe')
            neigh=preds[u]|succs[u]
            def affinity(k):
                val=0.0
                for v in neigh:
                    if assign.get(v)==k:
                        b=edge_bytes.get((u,v),edge_bytes.get((v,u),0))
                        val += 1000.0 + 2.0*b/BW
                return val
            oldaff=affinity(old)
            best=(0.0,old)
            for k in range(num_cores):
                if k==old:continue
                nm_old=loadM[old]-(cyc if pipe=='PIPE_M' else 0); nv_old=loadV[old]-(cyc if pipe=='PIPE_V' else 0)
                nm_new=loadM[k]+(cyc if pipe=='PIPE_M' else 0); nv_new=loadV[k]+(cyc if pipe=='PIPE_V' else 0)
                if max(nm_new/tgtM,nv_new/tgtV)>cap: continue
                before=max(loadM[old]/tgtM,loadV[old]/tgtV)+max(loadM[k]/tgtM,loadV[k]/tgtV)
                after=max(nm_old/tgtM,nv_old/tgtV)+max(nm_new/tgtM,nv_new/tgtV)
                gain=stick*(affinity(k)-oldaff)-500.0*(after-before)
                if gain>best[0]+1e-9:best=(gain,k)
            if best[1]!=old:
                k=best[1]
                if pipe=='PIPE_M':loadM[old]-=cyc;loadM[k]+=cyc
                elif pipe=='PIPE_V':loadV[old]-=cyc;loadV[k]+=cyc
                assign[u]=k;changed+=1
        if not changed:break

    # Cross-core-depth phase: cross-core edges strictly increase phase; same-core edges may stay in phase.
    phase={}
    for u in order:
        ph=0
        for p in preds[u]:
            ph=max(ph,phase[p]+(1 if assign[p]!=assign[u] else 0))
        phase[u]=ph
    # one task for each (phase, core); this guarantees an acyclic task graph and minimizes same-core boundaries for fixed assignment
    groups=defaultdict(list)
    for u in order:groups[(phase[u],assign[u])].append(u)
    keys=sorted(groups)
    sgid={key:i for i,key in enumerate(keys)}
    mapping={str(u):sgid[(phase[u],assign[u])] for u in nodes}
    core_sched=[[] for _ in range(num_cores)]
    for key in keys:
        ph,k=key; core_sched[k].append(sgid[key])
    return {'node_to_subgraph':mapping,'core_schedules':core_sched},{'num_subgraphs':len(keys),'max_phase':max(phase.values(),default=0),'loadM':loadM,'loadV':loadV}

def wcc_pack_plan(g,num_cores,bins=None):
    op_by,tensor_by,nodes,preds,succs,edge_bytes=build_views(g)
    seen=set(); comps=[]
    for u in nodes:
        if u in seen: continue
        st=[u];seen.add(u);cc=[]
        while st:
            x=st.pop();cc.append(x)
            for y in preds[x]|succs[x]:
                if y not in seen:seen.add(y);st.append(y)
        comps.append(cc)
    # disconnected components can be merged arbitrarily without changing dependencies.
    bcount=min(len(comps), bins or num_cores)
    BM=[0.0]*bcount;BV=[0.0]*bcount; groups=[[] for _ in range(bcount)]
    def w(cc):
        m=sum(op_by[u].get('cycles',0) for u in cc if op_by[u].get('pipe')=='PIPE_M')
        v=sum(op_by[u].get('cycles',0) for u in cc if op_by[u].get('pipe')=='PIPE_V')
        return m,v,max(m,v,0.65*(m+v))
    items=[(w(cc)[2],i,w(cc)[0],w(cc)[1]) for i,cc in enumerate(comps)]
    for _,i,m,v in sorted(items,reverse=True):
        k=min(range(bcount),key=lambda z:(max(BM[z]+m,BV[z]+v),BM[z]+BV[z],z))
        groups[k].extend(comps[i]);BM[k]+=m;BV[k]+=v
    groups=[x for x in groups if x]
    mapping={str(u):i for i,ch in enumerate(groups) for u in ch}
    # No dependencies across WCCs, but if a WCC is whole within one group there are no cross-task deps at all.
    # Since each WCC is indivisible here, groups remain mutually independent.
    core_sched=[[] for _ in range(num_cores)]
    # one group/task per active core by construction when bins<=num_cores
    for i in range(len(groups)): core_sched[i%num_cores].append(i)
    return {'node_to_subgraph':mapping,'core_schedules':core_sched},{'num_subgraphs':len(groups),'num_wcc':len(comps),'loadM':BM,'loadV':BV}

def wcc_pack_plan3(g,num_cores):
    op_by,tensor_by,nodes,preds,succs,edge_bytes=build_views(g)
    # eligible weak components
    seen=set(); comps=[]; comp_of={}
    for u in nodes:
        if u in seen: continue
        st=[u];seen.add(u);cc=[]
        while st:
            x=st.pop();cc.append(x)
            for y in preds[x]|succs[x]:
                if y not in seen:seen.add(y);st.append(y)
        ci=len(comps)
        for x in cc:comp_of[x]=ci
        comps.append(cc)
    # Map original tensors and copy ops to per-component DDR traffic.
    opids=set(op_by)
    prod=defaultdict(set);cons=defaultdict(set)
    for e in g['edges']:
        a,b=e['source'],e['target']
        if a in opids and b not in opids:prod[b].add(a)
        elif a not in opids and b in opids:cons[a].add(b)
    ddr=[0.0]*len(comps)
    for tid,t in tensor_by.items():
        s=float(t.get('size',0))
        # Any original COPY_IN -> tensor -> eligible consumer
        has_in=any(op_by[p].get('op')=='COPY_IN' for p in prod.get(tid,()) if p in op_by)
        has_out=any(op_by[c].get('op')=='COPY_OUT' for c in cons.get(tid,()) if c in op_by)
        if has_in:
            cs={comp_of[c] for c in cons.get(tid,()) if c in comp_of}
            for ci in cs: ddr[ci]+=s
        if has_out:
            ps={comp_of[p] for p in prod.get(tid,()) if p in comp_of}
            for ci in ps: ddr[ci]+=s
    stats=[]
    for i,cc in enumerate(comps):
        m=sum(op_by[u].get('cycles',0) for u in cc if op_by[u].get('pipe')=='PIPE_M')
        v=sum(op_by[u].get('cycles',0) for u in cc if op_by[u].get('pipe')=='PIPE_V')
        d=ddr[i]/BW
        stats.append((m,v,d,max(m,v,d)))
    bcount=min(num_cores,len(comps)); groups=[[] for _ in range(bcount)];BM=[0.]*bcount;BV=[0.]*bcount;BD=[0.]*bcount
    for _,i in sorted(((stats[i][3],i) for i in range(len(comps))), reverse=True):
        m,v,d,_=stats[i]
        k=min(range(bcount),key=lambda z:(max(BM[z]+m,BV[z]+v,BD[z]+d),BM[z]+BV[z]+BD[z],z))
        groups[k].extend(comps[i]);BM[k]+=m;BV[k]+=v;BD[k]+=d
    groups=[x for x in groups if x]
    mapping={str(u):i for i,ch in enumerate(groups) for u in ch}
    sched=[[] for _ in range(num_cores)]
    for i in range(len(groups)):sched[i].append(i)
    return {'node_to_subgraph':mapping,'core_schedules':sched},{'num_subgraphs':len(groups),'num_wcc':len(comps),'loadM':BM,'loadV':BV,'loadDDR':BD}

def kl_core_phase_plan(g,num_cores,seed=1,comm_scale=1.0):
    import networkx as nx
    op_by,tensor_by,nodes,preds,succs,edge_bytes=build_views(g)
    UG=nx.Graph(); UG.add_nodes_from(nodes)
    for u in nodes:
        for v in succs[u]:
            b=edge_bytes.get((u,v),0)
            w=1000.0+comm_scale*2.0*b/BW
            if UG.has_edge(u,v): UG[u][v]['weight']+=w
            else: UG.add_edge(u,v,weight=w)
    parts=[set(nodes)]
    # Recursively split the heaviest-work part until k parts.
    def part_work(S):
        m=sum(op_by[u].get('cycles',0) for u in S if op_by[u].get('pipe')=='PIPE_M')
        v=sum(op_by[u].get('cycles',0) for u in S if op_by[u].get('pipe')=='PIPE_V')
        return max(m,v,0.65*(m+v))
    while len(parts)<num_cores:
        idx=max(range(len(parts)),key=lambda i:part_work(parts[i]))
        S=parts.pop(idx)
        if len(S)<2:
            parts.append(S);break
        sub=UG.subgraph(S)
        try:
            a,b=nx.community.kernighan_lin_bisection(sub,weight='weight',seed=seed,max_iter=20)
        except Exception:
            ls=list(S);mid=len(ls)//2;a=set(ls[:mid]);b=set(ls[mid:])
        parts.extend([set(a),set(b)])
    # Map parts to cores and use cross-core-depth phase decomposition.
    assign={u:k for k,S in enumerate(parts) for u in S}
    order=topo(nodes,preds,succs);phase={}
    for u in order:
        phase[u]=max([phase[p]+(assign[p]!=assign[u]) for p in preds[u]],default=0)
    groups=defaultdict(list)
    for u in order:groups[(phase[u],assign[u])].append(u)
    keys=sorted(groups);sid={k:i for i,k in enumerate(keys)}
    mapping={str(u):sid[(phase[u],assign[u])] for u in nodes}
    sched=[[] for _ in range(num_cores)]
    for k in keys:sched[k[1]].append(sid[k])
    return {'node_to_subgraph':mapping,'core_schedules':sched},{'num_subgraphs':len(keys),'max_phase':max(phase.values(),default=0),'part_sizes':[len(x) for x in parts]}

# ---------------- v3: frontier-growing partition ----------------
def frontier_grow_plan(g,num_cores,waves=6,comm_alpha=5.0,rank_alpha=.02,max_ops=1024,min_fill=.35,
                       cross_wait=1000.0,same_wait=100.0):
    """Build branch-coherent tasks from the DAG ready frontier.

    Unlike level packing, a task can grow through several topological levels as long as
    newly ready successors have strong affinity to the current task. This preserves
    pipeline structure on dense single-WCC graphs while keeping the task DAG acyclic.
    """
    import heapq
    op_by,tensor_by,nodes,preds,succs,edge_bytes=build_views(g)
    order=topo(nodes,preds,succs)
    rank={}
    for u in reversed(order):
        base=max(1.0,float(op_by[u].get('cycles',0)))
        rank[u]=base+max((rank[v]+comm_alpha*(2.0*edge_bytes.get((u,v),0)/BW) for v in succs[u]),default=0.0)
    totalM=sum(op_by[u].get('cycles',0) for u in nodes if op_by[u].get('pipe')=='PIPE_M')
    totalV=sum(op_by[u].get('cycles',0) for u in nodes if op_by[u].get('pipe')=='PIPE_V')
    target=max(totalM,totalV,0.65*(totalM+totalV))/max(1,num_cores*waves)
    target=max(100.0,target)
    assigned=set(); predleft={u:len(preds[u]) for u in nodes}
    ready=set(u for u in nodes if predleft[u]==0); clusters=[]
    while ready:
        seed=max(ready,key=lambda u:(rank[u],op_by[u].get('cycles',0),-u));ready.remove(seed)
        cur=[];curset=set();M=V=O=0.0;cand={seed}
        while cand and len(cur)<max_ops:
            def score(u):
                aff=sum(cross_wait+2.0*edge_bytes.get((p,u),0)/BW for p in preds[u] if p in curset)
                return (comm_alpha*aff+rank_alpha*rank[u],rank[u],-u)
            u=max(cand,key=score);cand.remove(u)
            if u in assigned or u in curset:continue
            cyc=float(op_by[u].get('cycles',0));pipe=op_by[u].get('pipe')
            nm=M+(cyc if pipe=='PIPE_M' else 0);nv=V+(cyc if pipe=='PIPE_V' else 0);no=O+(cyc if pipe not in ('PIPE_M','PIPE_V') else 0)
            oldload=max(M,V,0.65*(M+V))+O;newload=max(nm,nv,0.65*(nm+nv))+no
            affbytes=sum(edge_bytes.get((p,u),0) for p in preds[u] if p in curset)
            if cur and newload>target and oldload>=min_fill*target and affbytes<4096:
                ready.add(u);break
            cur.append(u);curset.add(u);assigned.add(u);M,V,O=nm,nv,no
            for v in succs[u]:
                predleft[v]-=1
                if predleft[v]==0:ready.add(v)
            for x in list(ready):
                if any(p in curset for p in preds[x]):
                    cand.add(x);ready.discard(x)
        ready.update(x for x in cand if x not in assigned)
        if not cur:raise RuntimeError('empty frontier cluster')
        clusters.append(cur)
    if len(assigned)!=len(nodes):raise RuntimeError('frontier grow left nodes unassigned')
    cp,cs,eb=_quotient_from_clusters(clusters,nodes,succs,edge_bytes)
    indeg=[len(cp[i]) for i in range(len(clusters))];q=deque(sorted(i for i,d in enumerate(indeg) if d==0));ordc=[]
    while q:
        a=q.popleft();ordc.append(a)
        for b in sorted(cs[a]):
            indeg[b]-=1
            if indeg[b]==0:q.append(b)
    if len(ordc)!=len(clusters):raise RuntimeError('frontier quotient cycle')
    rem={old:new for new,old in enumerate(ordc)};c2=[clusters[i] for i in ordc];cp2=[set() for _ in c2];cs2=[set() for _ in c2];eb2=defaultdict(int)
    for (a,b),x in eb.items():
        na,nb=rem[a],rem[b];cp2[nb].add(na);cs2[na].add(nb);eb2[(na,nb)]+=x
    plan=assign_heft(c2,cp2,cs2,eb2,op_by,num_cores,None,cross_wait,same_wait)
    return plan,{'num_subgraphs':len(c2),'clusters':c2,'cp':cp2,'cs':cs2,'eb':eb2,'target':target}
