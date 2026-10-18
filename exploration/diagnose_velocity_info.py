"""
diagnose_velocity_info.py -- Investiga si la senal de entrada contiene la
informacion para inferir el AVANCE (d_fwd), y si esa relacion es consistente
entre Run B (train) y Run A (test). Responde a: el fallo de generalizacion del
avance, es del MODELO (la info esta y no la usa) o de la INFORMACION (la senal
no distingue las velocidades entre runs)?

Trabaja sobre los .npz del Bloque 5 (X normalizado NO sirve para magnitudes
fisicas; por eso re-lee los CSV crudos para lin_x y para la IMU). Compara:

  1) lin_x (velocidad de rueda) vs d_fwd real, en cada run:
     - correlacion y pendiente. Si lin_x predice d_fwd IGUAL en A y B,
       el WHEEL tiene la info -> el modelo deberia poder generalizar con wheel.
  2) "actividad" de la IMU (magnitud de |acc - g| y |gyro|) vs velocidad media,
     en cada run: si la IMU no cambia entre runs pese a velocidades distintas,
     la IMU sola NO distingue la velocidad (coherente: mide aceleracion, no
     velocidad; en crucero la aceleracion ~0 en ambos runs).

No reentrena. Uso:  python diagnose_velocity_info.py
"""
import csv
from pathlib import Path
import numpy as np

BASE = Path(r"C:\Users\Usuario\Desktop\TFM\extracted")
DT = 1.0; SEG = 2.0; MOVE_EPS = 0.05


def load(p):
    with open(p) as f:
        r = csv.reader(f); next(r)
        return np.array([[float(x) for x in row] for row in r])


def analyze(run):
    gt = load(BASE / run / "ground_truth.csv")
    imu = load(BASE / run / "imu.csv")
    wo = load(BASE / run / "wheel_odometry.csv")
    gt_t = gt[:, 0]
    imu_t = imu[:, 0] / 1e9
    wo_t = wo[:, 0] / 1e9

    def pos(ti):
        return np.array([np.interp(ti, gt_t, gt[:, 1]),
                         np.interp(ti, gt_t, gt[:, 2]),
                         np.interp(ti, gt_t, gt[:, 3])])

    # segmentacion
    segs, start = [], 0
    for i in range(1, len(gt)):
        if gt_t[i] - gt_t[i - 1] > SEG:
            segs.append((start, i)); start = i
    segs.append((start, len(gt)))

    dfwd_list, linx_list, accmag_list, gyromag_list = [], [], [], []
    for (s, e) in segs:
        t0, t1 = gt_t[s], gt_t[e - 1]
        if t1 - t0 < 2 * DT:
            continue
        t = t0 + DT
        while t + DT <= t1:
            d_prev = pos(t) - pos(t - DT)
            d_now = pos(t + DT) - pos(t)
            if np.hypot(d_prev[0], d_prev[1]) >= MOVE_EPS:
                heading = np.arctan2(d_prev[1], d_prev[0])
                c, sr = np.cos(heading), np.sin(heading)
                d_fwd = c * d_now[0] + sr * d_now[1]
                # lin_x medio del wheel en [t, t+DT]
                m = (wo_t >= t) & (wo_t < t + DT)
                linx = wo[m, 1].mean() if m.sum() > 0 else np.nan
                # actividad IMU en [t, t+DT]: |acc|-g y |gyro|
                mi = (imu_t >= t) & (imu_t < t + DT)
                if mi.sum() > 0:
                    acc = imu[mi, 4:7]
                    gyro = imu[mi, 1:4]
                    accmag = np.abs(np.linalg.norm(acc, axis=1) - 9.81).mean()
                    gyromag = np.linalg.norm(gyro, axis=1).mean()
                else:
                    accmag = gyromag = np.nan
                dfwd_list.append(d_fwd); linx_list.append(linx)
                accmag_list.append(accmag); gyromag_list.append(gyromag)
            t += DT

    d = np.array(dfwd_list); lx = np.array(linx_list)
    am = np.array(accmag_list); gm = np.array(gyromag_list)
    ok = ~np.isnan(lx) & ~np.isnan(am)
    d, lx, am, gm = d[ok], lx[ok], am[ok], gm[ok]

    # relacion lin_x -> d_fwd (ajuste lineal)
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, d, rcond=None)[0]
    corr = np.corrcoef(lx, d)[0, 1]

    print(f"=== {run} ({len(d)} intervalos con movimiento) ===")
    print(f"  d_fwd real: media={d.mean()*100:.1f}cm")
    print(f"  lin_x (vel. rueda): media={lx.mean()*100:.1f}cm/s")
    print(f"  relacion lin_x -> d_fwd: corr={corr:.3f}  pendiente={slope:.3f}  "
          f"interc={intercept*100:.1f}cm")
    print(f"  actividad IMU: |acc|-g media={am.mean():.3f} m/s2   "
          f"|gyro| media={gm.mean():.4f} rad/s")
    return {"slope": slope, "intercept": intercept, "corr": corr,
            "dfwd": d.mean(), "linx": lx.mean(), "accmag": am.mean(), "gyromag": gm.mean()}


print("Analizando relacion senal->velocidad en ambos runs...\n")
rb = analyze("Run_B")
print()
ra = analyze("Run_A")

print("\n=== INTERPRETACION ===")
print(f"lin_x -> d_fwd:  pendiente B={rb['slope']:.3f}  A={ra['slope']:.3f}  "
      f"(si son parecidas, el WHEEL da la velocidad de forma consistente)")
print(f"velocidad real:  B={rb['dfwd']*100:.1f}cm  A={ra['dfwd']*100:.1f}cm  "
      f"(A mas rapido)")
print(f"actividad IMU:   acc B={rb['accmag']:.3f} A={ra['accmag']:.3f}  |  "
      f"gyro B={rb['gyromag']:.4f} A={ra['gyromag']:.4f}")
print("Si la actividad IMU es similar entre runs pese a velocidades distintas,")
print("la IMU sola NO codifica la velocidad absoluta (mide aceleracion). En ese")
print("caso la velocidad vive en el WHEEL (lin_x), y la clave es que el modelo lo use.")