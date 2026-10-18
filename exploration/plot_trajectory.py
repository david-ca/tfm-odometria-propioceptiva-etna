"""
plot_trajectory.py -- Figura de trayectoria: reconstruccion del modelo (IMU-only,
Hold-Out) sobre Run A, superpuesta a la trayectoria real (DGPS). Dos paneles:
rumbo ideal (GPS) y rumbo propioceptivo (Madgwick). Guarda figura_trayectoria.png.

Reutiliza la logica de reconstruct.py. Uso:  python plot_trajectory.py
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(r"C:\Users\Usuario\Desktop\TFM\extracted")
MODELS = Path(r"C:\Users\Usuario\Desktop\TFM\models")
DT = 1.0

def load(p):
    with open(p) as f:
        r = csv.reader(f); next(r)
        return np.array([[float(x) for x in row] for row in r])

def umeyama_2d(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    S = (dst-md).T @ (src-ms) / len(src)
    U,D,Vt = np.linalg.svd(S); R = U@Vt
    if np.linalg.det(R)<0: U[:,-1]*=-1; R=U@Vt
    s = D.sum()/((src-ms)**2).sum()*len(src)
    return (s*(R@src.T)).T + (md - s*(R@ms))

pred_data = np.load(MODELS / "test_pred_gru_imu.npz")
pred, true, seg = pred_data["pred"], pred_data["true"], pred_data["seg_id"]
ds = np.load(BASE/"Run_A"/"dataset_imu_Run_A.npz", allow_pickle=True)
t_start = ds["t_start"]; assert np.allclose(ds["y"], true, atol=1e-5)

gt = load(BASE/"Run_A"/"ground_truth.csv"); gt_t = gt[:,0]
ori = load(BASE/"Run_A"/"orientation_full.csv"); ori_t = ori[:,0]/1e9; yaw = ori[:,7]

def pos(t): return np.array([np.interp(t,gt_t,gt[:,1]), np.interp(t,gt_t,gt[:,2]), np.interp(t,gt_t,gt[:,3])])
def gps_head(t):
    d = pos(t+DT)-pos(t); return np.arctan2(d[1],d[0])
def madg_head(t): return np.deg2rad(np.interp(t,ori_t,yaw))

def reconstruct(mode):
    est_all, gt_all = [], []
    for s in np.unique(seg):
        idx = np.where(seg==s)[0]; idx = idx[np.argsort(t_start[idx])]
        p0 = pos(t_start[idx[0]]); p = p0.copy(); est=[p.copy()]; g=[p0.copy()]
        for k in idx:
            t = t_start[k]; h = gps_head(t) if mode=="gps" else madg_head(t)
            c,sr = np.cos(h),np.sin(h); df,dl,du = pred[k]
            p = p + np.array([c*df-sr*dl, sr*df+c*dl, du])
            est.append(p.copy()); g.append(pos(t+DT))
        est_all.append(np.array(est)); gt_all.append(np.array(g))
    return est_all, gt_all

fig, axes = plt.subplots(1, 2, figsize=(13,6))
for ax, (mode, titulo, ate) in zip(axes, [("gps","Rumbo ideal (GPS)  ATE=21,3 m","gps"),("prop","Rumbo propioceptivo  ATE=51,0 m","prop")]):
    est_all, gt_all = reconstruct(mode)
    # alinear cada tramo con Umeyama (como el ATE) y trazar
    for i,(est,g) in enumerate(zip(est_all,gt_all)):
        e2 = umeyama_2d(est[:,:2], g[:,:2])
        lg = 'Real (DGPS)' if i==0 else None
        le = 'Estimada (modelo)' if i==0 else None
        ax.plot(g[:,0], g[:,1], 'g-', lw=2, label=lg)
        ax.plot(e2[:,0], e2[:,1], 'r--', lw=1.3, label=le)
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.axis('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.suptitle('Reconstrucción de la trayectoria en Run A (modelo solo-IMU) frente a la referencia', fontsize=12)
fig.tight_layout()
fig.savefig('figura_trayectoria.png', dpi=150, bbox_inches='tight')
print("Guardado: figura_trayectoria.png")