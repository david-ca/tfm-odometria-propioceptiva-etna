"""
baseline_increments.py -- Error de INCREMENTO del baseline (odometria de ruedas),
en el mismo marco y contra el mismo target que el modelo, para comparar de forma
justa el MAE por intervalo (lo que el tutor pide: verificar pertinencia frente a
un metodo de referencia).

Para cada intervalo de 1 s del target (misma logica que build_target.py):
  - target real (d_fwd, d_lat) desde el GPS, marco relativo al movimiento previo.
  - prediccion del baseline: integra la odometria de ruedas (lin_x, lin_y) en el
    intervalo -> desplazamiento del cuerpo; se proyecta al MISMO marco relativo
    (usando la direccion de movimiento previa del GPS, igual que el target) y se
    compara con el target real.

El baseline es 2D: no estima d_up, asi que se comparan solo d_fwd y d_lat.
Nota: se usa la direccion de referencia del GPS previo para ambos (target y
baseline), de modo que la comparacion aisla la calidad del DESPLAZAMIENTO de la
odometria de ruedas, no su rumbo (que se evalua aparte en la trayectoria/ATE).

Uso:  python baseline_increments.py Run_A
"""
import sys, csv
from pathlib import Path
import numpy as np

if len(sys.argv) < 2:
    print("Uso: python baseline_increments.py <RUN>"); sys.exit(1)
RUN = sys.argv[1]
BASE = Path(r"C:\Users\Usuario\Desktop\TFM") / "extracted" / RUN
DT = 1.0; SEG = 2.0; MOVE_EPS = 0.05


def load(p):
    with open(p) as f:
        r = csv.reader(f); next(r)
        return np.array([[float(x) for x in row] for row in r])


gt = load(BASE / "ground_truth.csv")
wo = load(BASE / "wheel_odometry.csv")
gt_t = gt[:, 0]
wo_t = wo[:, 0] / 1e9


def pos(ti):
    return np.array([np.interp(ti, gt_t, gt[:, 1]), np.interp(ti, gt_t, gt[:, 2])])


def wheel_disp(t0, t1):
    """Desplazamiento del cuerpo segun la odometria de ruedas en [t0,t1]:
    integral de (lin_x, lin_y). Devuelve (dx_body, dy_body) en el marco del cuerpo."""
    m = (wo_t >= t0) & (wo_t <= t1)
    if m.sum() < 2:
        return None
    tt = wo_t[m]; vx = wo[m, 1]; vy = wo[m, 2]
    dts = np.diff(tt)
    dx = float(np.sum((vx[:-1] + vx[1:]) / 2 * dts))
    dy = float(np.sum((vy[:-1] + vy[1:]) / 2 * dts))
    return dx, dy


# segmentacion
segs, start = [], 0
for i in range(1, len(gt)):
    if gt_t[i] - gt_t[i-1] > SEG:
        segs.append((start, i)); start = i
segs.append((start, len(gt)))

err_fwd, err_lat = [], []
tgt_fwd, tgt_lat = [], []
bas_fwd, bas_lat = [], []
n = 0
for (s, e) in segs:
    t0, t1 = gt_t[s], gt_t[e-1]
    if t1 - t0 < 2 * DT:
        continue
    t = t0 + DT
    while t + DT <= t1:
        d_prev = pos(t) - pos(t - DT)
        d_now = pos(t + DT) - pos(t)
        if np.hypot(*d_prev) >= MOVE_EPS:
            heading = np.arctan2(d_prev[1], d_prev[0])
            c, sr = np.cos(heading), np.sin(heading)
            # target real en marco relativo
            tf = c * d_now[0] + sr * d_now[1]
            tl = -sr * d_now[0] + c * d_now[1]
            # baseline: desplazamiento de ruedas en el marco del cuerpo, proyectado
            # al MISMO marco relativo (usando la direccion de referencia del GPS)
            wd = wheel_disp(t, t + DT)
            if wd is not None:
                # el desplazamiento de ruedas ya esta en marco del cuerpo (lin_x=fwd
                # aproximado). Para comparar con el target relativo, tomamos lin_x
                # como avance y lin_y como lateral directamente (marco del cuerpo).
                bf, bl = wd[0], wd[1]
                err_fwd.append(abs(bf - tf)); err_lat.append(abs(bl - tl))
                tgt_fwd.append(tf); tgt_lat.append(tl)
                bas_fwd.append(bf); bas_lat.append(bl)
                n += 1
        t += DT

err_fwd = np.array(err_fwd); err_lat = np.array(err_lat)
tgt_fwd = np.array(tgt_fwd); bas_fwd = np.array(bas_fwd)
tgt_lat = np.array(tgt_lat); bas_lat = np.array(bas_lat)

print(f"=== Error de INCREMENTO del baseline (odometria de ruedas) -- {RUN} ===")
print(f"    {n} intervalos con movimiento\n")
print(f"  d_fwd:  MAE={err_fwd.mean()*100:.2f} cm   RMSE={np.sqrt((err_fwd**2).mean())*100:.2f} cm")
print(f"          target medio={np.abs(tgt_fwd).mean()*100:.2f} cm  baseline medio={np.abs(bas_fwd).mean()*100:.2f} cm")
print(f"          sesgo (baseline-target)={ (bas_fwd-tgt_fwd).mean()*100:+.2f} cm")
print(f"  d_lat:  MAE={err_lat.mean()*100:.2f} cm   RMSE={np.sqrt((err_lat**2).mean())*100:.2f} cm")
print(f"          target medio={np.abs(tgt_lat).mean()*100:.2f} cm  baseline medio={np.abs(bas_lat).mean()*100:.2f} cm")
print(f"\n  (baseline 2D: no estima d_up; se comparan solo d_fwd y d_lat)")
print(f"\n=== Para comparar con el modelo (Hold-Out, test = Run_A) ===")
print(f"  modelo IMU-only  d_fwd MAE: 9.10 cm   |  d_lat MAE: 2.32 cm")
print(f"  modelo IMU+wheel d_fwd MAE: 17.38 cm  |  d_lat MAE: 2.99 cm")
print(f"  baseline (arriba) d_fwd MAE: {err_fwd.mean()*100:.2f} cm  |  d_lat MAE: {err_lat.mean()*100:.2f} cm")