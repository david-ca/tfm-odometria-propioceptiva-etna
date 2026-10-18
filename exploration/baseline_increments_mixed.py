"""
baseline_increments_mixed.py -- Error de INCREMENTO del baseline sobre EXACTAMENTE
el mismo conjunto de test que la particion mixta (la porcion de tramos de Run B
reservada a test), para comparar manzana con manzana con el modelo en esa particion.

Lee que tramos de B son de test desde dataset_<cfg>_mixed_test.npz (seg_id con
prefijo +2000), y calcula el MAE de incremento del baseline (odometria de ruedas)
solo sobre esos tramos de Run B, en el mismo marco relativo que el target.

Uso:  python baseline_increments_mixed.py
"""
from pathlib import Path
import csv
import numpy as np

BASE = Path(r"C:\Users\Usuario\Desktop\TFM\extracted")
RUN = "Run_B"     # el test mixto es una porcion de Run B
DT = 1.0; SEG = 2.0; MOVE_EPS = 0.05


def load(p):
    with open(p) as f:
        r = csv.reader(f); next(r)
        return np.array([[float(x) for x in row] for row in r])


# tramos de B que estan en el test mixto (seg_id - 2000)
d = np.load(BASE / "Run_B" / "dataset_imu_mixed_test.npz", allow_pickle=True) \
    if (BASE / "Run_B" / "dataset_imu_mixed_test.npz").exists() \
    else np.load(BASE / "dataset_imu_mixed_test.npz", allow_pickle=True)
test_segs = set(int(s) - 2000 for s in np.unique(d["seg_id"]))
print(f"Tramos de Run B en el test mixto: {sorted(test_segs)}  ({len(test_segs)} tramos)\n")

gt = load(BASE / RUN / "ground_truth.csv")
wo = load(BASE / RUN / "wheel_odometry.csv")
gt_t = gt[:, 0]; wo_t = wo[:, 0] / 1e9


def pos(ti):
    return np.array([np.interp(ti, gt_t, gt[:, 1]), np.interp(ti, gt_t, gt[:, 2])])


def wheel_disp(t0, t1):
    m = (wo_t >= t0) & (wo_t <= t1)
    if m.sum() < 2:
        return None
    tt = wo_t[m]; vx = wo[m, 1]; vy = wo[m, 2]
    dts = np.diff(tt)
    return (float(np.sum((vx[:-1]+vx[1:])/2*dts)), float(np.sum((vy[:-1]+vy[1:])/2*dts)))


# segmentacion identica a build_windows (numeracion 1..N sobre tramos usables)
segs, start = [], 0
for i in range(1, len(gt)):
    if gt_t[i]-gt_t[i-1] > SEG:
        segs.append((start, i)); start = i
segs.append((start, len(gt)))

err_fwd, err_lat, tgt_lat, bas_lat = [], [], [], []
seg_id = 0
n = 0
for (s, e) in segs:
    t0, t1 = gt_t[s], gt_t[e-1]
    if t1 - t0 < 2*DT:
        continue
    seg_id += 1
    if seg_id not in test_segs:      # solo los tramos del test mixto
        continue
    t = t0 + DT
    while t + DT <= t1:
        d_prev = pos(t) - pos(t-DT); d_now = pos(t+DT) - pos(t)
        if np.hypot(*d_prev) >= MOVE_EPS:
            heading = np.arctan2(d_prev[1], d_prev[0])
            c, sr = np.cos(heading), np.sin(heading)
            tf = c*d_now[0] + sr*d_now[1]; tl = -sr*d_now[0] + c*d_now[1]
            wd = wheel_disp(t, t+DT)
            if wd is not None:
                err_fwd.append(abs(wd[0]-tf)); err_lat.append(abs(wd[1]-tl))
                tgt_lat.append(tl); bas_lat.append(wd[1]); n += 1
        t += DT

err_fwd = np.array(err_fwd); err_lat = np.array(err_lat)
print(f"=== Baseline sobre el TEST MIXTO (porcion de Run B, {n} intervalos) ===\n")
print(f"  d_fwd:  MAE={err_fwd.mean()*100:.2f} cm   RMSE={np.sqrt((err_fwd**2).mean())*100:.2f} cm")
print(f"  d_lat:  MAE={err_lat.mean()*100:.2f} cm   RMSE={np.sqrt((err_lat**2).mean())*100:.2f} cm")
print(f"          target lat medio={np.abs(tgt_lat).mean()*100:.2f} cm  "
      f"baseline lat medio={np.abs(bas_lat).mean()*100:.2f} cm")
print(f"\n=== Comparacion en la PARTICION MIXTA (mismo test) ===")
print(f"  modelo IMU-only  d_fwd MAE: 1.89 cm  |  d_lat MAE: ~1.67 cm")
print(f"  modelo IMU+wheel d_fwd MAE: 1.64 cm  |  d_lat MAE: ~1.67 cm")
print(f"  baseline         d_fwd MAE: {err_fwd.mean()*100:.2f} cm  |  d_lat MAE: {err_lat.mean()*100:.2f} cm")