#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, math, statistics
from pathlib import Path

def num(r,k,d=None):
    v=str(r.get(k,'')).strip()
    if not v: return d
    try: return float(v)
    except: return d

def finite(xs): return [x for x in xs if x is not None and math.isfinite(x)]
def mean(xs):
    xs=finite(xs); return statistics.fmean(xs) if xs else float('nan')
def pct(xs,q):
    xs=sorted(finite(xs))
    if not xs: return float('nan')
    p=(len(xs)-1)*q; lo=int(math.floor(p)); hi=int(math.ceil(p))
    if lo==hi: return xs[lo]
    a=p-lo; return xs[lo]*(1-a)+xs[hi]*a

def fmt(x,n=3):
    try:
        if not math.isfinite(float(x)): return 'nan'
        return f'{float(x):.{n}f}'
    except: return str(x)

def load(p):
    with open(p,'r',encoding='utf-8',newline='') as f: return list(csv.DictReader(f))

def first_brake(rows,event,thr=0.1):
    for r in rows:
        t=num(r,'t_s')
        if t is not None and t>=event and (num(r,'brake',0) or 0)>thr: return r
    return None

def summarize(rows,route_len,event):
    dev=[num(r,'route_deviation_m') for r in rows]
    prog=[num(r,'ego_route_progress_m') for r in rows]
    gap=[num(r,'route_bumper_gap_m') for r in rows]
    ttc=[num(r,'route_ttc_s') for r in rows]
    spd=[num(r,'ego_speed_mps') for r in rows]
    st=[num(r,'steer') for r in rows]
    th=[num(r,'throttle') for r in rows]
    br=[num(r,'brake') for r in rows]
    dst=[num(r,'destination_distance_m') for r in rows]
    fb=first_brake(rows,event)
    fprog=finite(prog); fdst=finite(dst); fbr=finite(br)
    return {
      'frames':len(rows),'last_frame':int(num(rows[-1],'scenario_frame',-1)) if rows else -1,
      'route_completion_pct':100*max(fprog)/route_len if fprog else float('nan'),
      'route_dev_mean_m':mean(dev),'route_dev_p95_m':pct(dev,.95),'route_dev_max_m':max(finite(dev)) if finite(dev) else float('nan'),
      'destination_final_m':fdst[-1] if fdst else float('nan'),'destination_min_m':min(fdst) if fdst else float('nan'),
      'speed_mean_mps':mean(spd),'abs_steer_mean':mean([abs(x) for x in finite(st)]),'throttle_mean':mean(th),'brake_mean':mean(br),
      'brake_fraction_pct':100*sum(x>.1 for x in fbr)/len(fbr) if fbr else float('nan'),
      'min_bumper_gap_m':min(finite(gap)) if finite(gap) else float('nan'),
      'min_finite_ttc_s':min(finite(ttc)) if finite(ttc) else float('nan'),
      'virtual_collision_frames':sum((num(r,'route_virtual_collision',0) or 0)>0.5 for r in rows),
      'first_brake_frame_after_event':int(num(fb,'scenario_frame',-1)) if fb else -1,
      'reaction_delay_s':num(fb,'t_s')-event if fb else float('nan'),
    }

def aligned(c,h):
    C={int(num(r,'scenario_frame')):r for r in c}; H={int(num(r,'scenario_frame')):r for r in h}
    return [(k,C[k],H[k]) for k in sorted(C.keys() & H.keys())]

def absdiff(P,k):
    out=[]
    for _,a,b in P:
        x=num(a,k); y=num(b,k)
        if x is not None and y is not None and math.isfinite(x) and math.isfinite(y): out.append(abs(x-y))
    return out

def pair(c,h):
    P=aligned(c,h); xy=[]
    for _,a,b in P:
        ax,ay,bx,by=num(a,'ego_x'),num(a,'ego_y'),num(b,'ego_x'),num(b,'ego_y')
        if None not in (ax,ay,bx,by): xy.append(math.hypot(ax-bx,ay-by))
    return {
      'common_frames':len(P),
      'ego_xy_error_mean_m':mean(xy),'ego_xy_error_p95_m':pct(xy,.95),'ego_xy_error_max_m':max(finite(xy)) if finite(xy) else float('nan'),
      'speed_abs_error_mean_mps':mean(absdiff(P,'ego_speed_mps')),
      'route_progress_abs_error_mean_m':mean(absdiff(P,'ego_route_progress_m')),
      'route_deviation_abs_error_mean_m':mean(absdiff(P,'route_deviation_m')),
      'steer_abs_error_mean':mean(absdiff(P,'steer')),
      'throttle_abs_error_mean':mean(absdiff(P,'throttle')),
      'brake_abs_error_mean':mean(absdiff(P,'brake')),
      'bumper_gap_abs_error_mean_m':mean(absdiff(P,'route_bumper_gap_m')),
      'destination_abs_error_mean_m':mean(absdiff(P,'destination_distance_m')),
    }

def first_dev(rows,t):
    for r in rows:
        d=num(r,'route_deviation_m')
        if d is not None and d>t: return r
    return None

def cilpp_diag(c,h):
    print('\n'+'='*100+'\nCIL++ ROUTE / DESTINATION DIAGNOSTIC\n'+'='*100)
    for name,rows in [('CARLA',c),('HE',h)]:
        print(f'\n[{name}]')
        for t in (1,2,4,8):
            r=first_dev(rows,t)
            if r:
                print(f"first dev>{t}m: f={int(num(r,'scenario_frame',-1))} t={fmt(num(r,'t_s'))} dev={fmt(num(r,'route_deviation_m'))} dest={fmt(num(r,'destination_distance_m'))} route_idx={int(num(r,'route_index',-1))} progress={fmt(num(r,'ego_route_progress_m'))} yaw={fmt(num(r,'ego_yaw'))} cmd={r.get('command_name','')}({r.get('command_value','')}) target=({fmt(num(r,'target_x'))},{fmt(num(r,'target_y'))})")
            else: print(f'first dev>{t}m: never')
        print('command transitions:')
        prev=None; n=0
        for r in rows:
            cur=(r.get('command_name',''),r.get('command_value',''))
            if cur!=prev:
                print(f" f={int(num(r,'scenario_frame',-1)):3d} t={fmt(num(r,'t_s'))} route_idx={int(num(r,'route_index',-1))} dev={fmt(num(r,'route_deviation_m'))} dest={fmt(num(r,'destination_distance_m'))} cmd={cur[0]}({cur[1]}) target=({fmt(num(r,'target_x'))},{fmt(num(r,'target_y'))})")
                prev=cur; n+=1
                if n>=20: break
        print('last 10 frames:')
        for r in rows[-10:]:
            print(f" f={int(num(r,'scenario_frame',-1)):3d} dev={fmt(num(r,'route_deviation_m'))} dest={fmt(num(r,'destination_distance_m'))} route_idx={int(num(r,'route_index',-1))} progress={fmt(num(r,'ego_route_progress_m'))} yaw={fmt(num(r,'ego_yaw'))} steer={fmt(num(r,'steer'))} cmd={r.get('command_name','')} target=({fmt(num(r,'target_x'))},{fmt(num(r,'target_y'))})")

def writecsv(path,rows):
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default=r'driving_models\outputs\physical_anchor_long_v1\signalized_lead_follow_001')
    ap.add_argument('--scenario',default='signalized_lead_follow_001')
    ap.add_argument('--route-length-m',type=float,default=204.45)
    ap.add_argument('--event-start-s',type=float,default=23.76)
    a=ap.parse_args(); root=Path(a.root)
    condout=[]; pairout=[]; D={}
    print('='*100+'\nCARLA vs HE: MODEL-BY-MODEL\n'+'='*100)
    for m in ('tcp','neat','cilpp','aimmt'):
        D[m]={}
        for c in ('carla','he'):
            p=root/m/f'{a.scenario}_{m}_{c}.csv'; R=load(p); D[m][c]=R; s=summarize(R,a.route_length_m,a.event_start_s)
            condout.append({'model':m,'condition':c,**s})
            print(f"{m.upper():6s} {c.upper():5s} frames={s['frames']:3d} route={fmt(s['route_completion_pct'],1)}% dev={fmt(s['route_dev_mean_m'])}/{fmt(s['route_dev_p95_m'])}/{fmt(s['route_dev_max_m'])}m min_gap={fmt(s['min_bumper_gap_m'])}m min_TTC={fmt(s['min_finite_ttc_s'])}s coll={s['virtual_collision_frames']} brake_f={s['first_brake_frame_after_event']} delay={fmt(s['reaction_delay_s'])}s")
        p=pair(D[m]['carla'],D[m]['he']); pairout.append({'model':m,**p,'carla_frames':len(D[m]['carla']),'he_frames':len(D[m]['he']),'frame_delta_he_minus_carla':len(D[m]['he'])-len(D[m]['carla'])})
        print(f"{m.upper():6s} PAIR  common={p['common_frames']} XY_MAE={fmt(p['ego_xy_error_mean_m'])}m XY_P95={fmt(p['ego_xy_error_p95_m'])}m speed_MAE={fmt(p['speed_abs_error_mean_mps'])}m/s progress_MAE={fmt(p['route_progress_abs_error_mean_m'])}m steer_MAE={fmt(p['steer_abs_error_mean'])} gap_MAE={fmt(p['bumper_gap_abs_error_mean_m'])}m\n")
    c1=root/'carla_vs_he_condition_summary_v1.csv'; c2=root/'carla_vs_he_pair_summary_v1.csv'; writecsv(c1,condout); writecsv(c2,pairout)
    print('saved:',c1); print('saved:',c2)
    cilpp_diag(D['cilpp']['carla'],D['cilpp']['he'])
if __name__=='__main__': main()
