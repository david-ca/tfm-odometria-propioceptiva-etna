"""
plot_velocity_hist.py -- Histograma del avance por intervalo (proxy de velocidad)
en Run A y Run B, para ilustrar que sus perfiles de velocidad difieren. Respalda
el hallazgo de la cobertura de velocidades. Genera figura_velocidades.png.
Uso:  python plot_velocity_hist.py
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(r"C:\Users\Usuario\Desktop\TFM\extracted")
DT = 1.0; SEG = 2.0; MOVE_EPS = 0.05

def load(p):
    with open(p) as f:
        r = csv.reader(f); next(r)
        return np.array([[float(x) for x in row] for row in r])

def advances(run):
    gt = load(BASE/run/"ground_truth.csv"); gt_t = gt[:,0]
    def pos(t): return np.array([np.interp(t,gt_t,gt[:,1]), np.interp(t,gt_t,gt[:,2])])
    segs,start=[],0
    for i in range(1,len(gt)):
        if gt_t[i]-gt_t[i-1]>SEG: segs.append((start,i)); start=i
    segs.append((start,len(gt)))
    adv=[]
    for (s,e) in segs:
        t0,t1=gt_t[s],gt_t[e-1]
        if t1-t0<2*DT: continue
        t=t0+DT
        while t+DT<=t1:
            d_prev=pos(t)-pos(t-DT); d_now=pos(t+DT)-pos(t)
            if np.hypot(*d_prev)>=MOVE_EPS:
                h=np.arctan2(d_prev[1],d_prev[0]); c,sr=np.cos(h),np.sin(h)
                adv.append((c*d_now[0]+sr*d_now[1])*100)  # cm/s
            t+=DT
    return np.array(adv)

a = advances("Run_A"); b = advances("Run_B")
fig, ax = plt.subplots(figsize=(8,5))
bins = np.linspace(0, 40, 41)
ax.hist(b, bins=bins, alpha=0.6, label=f"Recorrido B (media {b.mean():.1f} cm/s)", color="tab:green", density=True)
ax.hist(a, bins=bins, alpha=0.6, label=f"Recorrido A (media {a.mean():.1f} cm/s)", color="tab:orange", density=True)
ax.axvline(b.mean(), color="tab:green", ls="--", lw=1.5)
ax.axvline(a.mean(), color="tab:orange", ls="--", lw=1.5)
ax.set_xlabel("Avance por intervalo (cm/s)")
ax.set_ylabel("Densidad")
ax.set_title("Distribución de la velocidad de avance en cada recorrido")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figura_velocidades.png", dpi=150, bbox_inches="tight")
print(f"Guardado: figura_velocidades.png  |  A: media={a.mean():.1f} p10={np.percentile(a,10):.1f} p90={np.percentile(a,90):.1f}  |  B: media={b.mean():.1f} p10={np.percentile(b,10):.1f} p90={np.percentile(b,90):.1f}")