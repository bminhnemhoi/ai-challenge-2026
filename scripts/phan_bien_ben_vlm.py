# -*- coding: utf-8 -*-
"""Kiem chung do BEN cua cau hinh DA CHOT (siglipB, alpha=0,5, w=1,0) tren CA 66 cau
hai canh, voi ho hat giong DOC LAP (666000). Cau hinh da co dinh tu truoc — day khong
phai mot lan chon nua, ma la doc lai cung mot cau hinh tren toan bo nhom bi tac dong."""
import argparse, json, sys
import numpy as np
from collections import defaultdict
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.experiment_phu_quet_luoi import cac_lan_boc, cham_nhanh, ma_tran_dong
from scripts.make_submission import DEFAULT_N_FLAT, allocate_rows
from scripts.experiment_cap_thoi_gian import _plan
from src.core.submission import MAX_ROWS
import scripts.do_vlm_noi_video_moi as L

a = argparse.Namespace(data=str(pathlib.Path(__file__).resolve().parents[1]/'data'), moi=str(pathlib.Path(__file__).resolve().parents[1]/'data'/'ground_truth_moi.json'),
                       cache=str(pathlib.Path(__file__).resolve().parents[1]/'data'/'cache_bo_do_moi'), canh_b=100, allocator='coverage',
                       videos=3, frames=12, windows='6,10,20')
sach, nhan, bat, cands, kf, ten_khung, hang_of, kho = L.nap(a)
windows=[6,10,20]
rows_nen=[allocate_rows(c,'coverage',DEFAULT_N_FLAT,_plan())[:MAX_ROWS] for c in cands]
khung_of={i: L.chon_khung_de_cham(cands[i], 3, 12) for i in bat}

def tin_hieu_siglip():
    ra_all={}
    for i in bat:
        s=kho.lay(nhan[i]['canh_B_vi'], nhan[i]['canh_B_en'])
        tv=defaultdict(list)
        for c in khung_of[i]: tv[c.video_id].append(int(c.frame_idx))
        ra={}
        for v,fs in tv.items():
            gia=np.array([float(s[hang_of[(v,f)]]) for f in fs])
            lo,hi=float(gia.min()),float(gia.max())
            ch=(gia-lo)/(hi-lo) if hi>lo else np.zeros_like(gia)
            for f,x in zip(fs,ch): ra[(v,f)]=(float(x),'siglip')
        ra_all[i]=ra
    return ra_all
sig=tin_hieu_siglip()
alpha,w=0.5,1.0
rows=list(rows_nen)
for i in bat:
    sc=sig[i]; tv=defaultdict(list)
    for (v,f) in sc: tv[v].append(f)
    dB={k:float(v[0]) for k,v in sc.items()}
    loc={}
    for v,fs in tv.items():
        sub=L.suy_ra_loc({f:dB[(v,f)] for f in fs},{v:fs},alpha)
        for f,x in sub.items(): loc[(v,f)]=x
    key=L.khoa_theo_chi_so(cands[i],loc,w)
    if len(key)<2: continue
    rows[i]=allocate_rows(L.hoan_vi_diem(cands[i],key),'coverage',DEFAULT_N_FLAT,_plan())[:MAX_ROWS]
ngoai=[i for i in range(len(sach)) if i not in bat]
for i in ngoai: assert rows[i]==rows_nen[i]
print(f'bat bien: {len(ngoai)} cau MOT canh ra dong giong het nen (assert OK)')

gt=[sach[i] for i in bat]
ho=cac_lan_boc(666000,6,48,gt,kf)
mn=ma_tran_dong([rows_nen[i] for i in bat],gt); mc=ma_tran_dong([rows[i] for i in bat],gt)
def per(m):
    r=np.zeros(len(gt))
    for dr in ho:
        for q in range(len(gt)): r[q]+=cham_nhanh([m[q]],[dr[q]],windows)
    return r/len(ho)
dn,dc=per(mn),per(mc)
h=dc-dn
print(f'\nCA 66 cau HAI canh, hat giong DOC LAP 666000:')
print(f'  nen {dn.mean():.4f} -> chot {dc.mean():.4f} = {100*(dc.mean()/dn.mean()-1):+.1f}%  '
      f'(chenh tuyet doi {h.mean():+.4f})')
print(f'  {int((h>1e-12).sum())} cau TOT | {int((h<-1e-12).sum())} cau XAU | '
      f'{int((abs(h)<=1e-12).sum())} khong doi')
rng=np.random.default_rng(2026)
lay=rng.integers(0,len(h),size=(4000,len(h)))
dd=dc[lay].mean(1)-dn[lay].mean(1)
print(f'  bootstrap theo CAU: KTC 95% [{np.percentile(dd,2.5):+.4f}, {np.percentile(dd,97.5):+.4f}]'
      f'  P(<=0)={float((dd<=0).mean()):.1%}')
o=np.argsort(-h)
print(f'\n  DO BEN — bo k cau dong gop nhieu nhat:')
for k in (0,1,2,3,5,8):
    keep=np.array(sorted(o[k:]))
    print(f'    bo {k:>2} cau: n={len(keep):>2}  {dn[keep].mean():.4f} -> {dc[keep].mean():.4f} '
          f'= {100*(dc[keep].mean()/dn[keep].mean()-1):+6.1f}%')
print(f'\n  5 cau dong gop lon nhat: {np.round(h[o[:5]],3)}  = '
      f'{100*h[o[:5]].sum()/max(h.sum(),1e-12):.0f}% tong muc tang')
