"""
diagnose_runB_wheel.py -- Por que lin_x (velocidad de rueda) apenas correlaciona
con d_fwd (avance real) en Run B (corr 0.38) mientras en Run A es casi perfecta
(corr 0.90)? Distingue entre:
  (a) deslizamiento genuino  -> nube dispersa, lin_x alto con avance variable
  (b) problema de escala     -> relacion lineal pero pendiente/offset raros
  (c) artefactos de datos    -> valores anomalos de lin_x rompiendo la correlacion

Genera para cada run: percentiles de lin_x y d_fwd, el ratio d_fwd/lin_x
(deberia ser ~1 si la rueda mide bien el avance en 1s), y una tabla por franjas
de lin_x mostrando el avance medio en cada franja (revela si la relacion es
monotona o esta rota). Sin dependencias graficas (texto).

Uso:  python diagnose_runB_wheel.py
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


def collect(run):
    gt = load(BASE / run / "ground_truth.csv")
    wo = load(BASE / run / "wheel_odometry.csv")
    gt_t = gt[:, 0]; wo_t = wo[:, 0] / 1e9

    def pos(ti):
        return np.array([np.interp(ti, gt_t, gt[:, 1]),
                         np.interp(ti, gt_t, gt[:, 2]),
                         np.interp(ti, gt_t, gt[:, 3])])

    segs, start = [], 0
    for i in range(1, len(gt)):
        if gt_t[i] - gt_t[i - 1] > SEG:
            segs.append((start, i)); start = i
    segs.append((start, len(gt)))

    D, LX = [], []
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
                m = (wo_t >= t) & (wo_t < t + DT)
                if m.sum() > 0:
                    tt = wo_t[m]; vx = wo[m, 1]
                    # excluir intervalos que contengan filas todo-cero del wheel
                    # (sensor en reposo/ausente): contaminarian la integral y
                    # se confundirian con deslizamiento real
                    wrows = wo[m, 1:]
                    has_zero = np.any(np.all(wrows == 0.0, axis=1))
                    if has_zero:
                        t += DT
                        continue
                    if len(tt) >= 2:
                        # integral trapezoidal manual (compatible con NumPy 2.x,
                        # donde np.trapz fue eliminado en favor de np.trapezoid)
                        dts = np.diff(tt)
                        wheel_adv = float(np.sum((vx[:-1] + vx[1:]) / 2 * dts))
                    else:
                        wheel_adv = vx.mean() * DT
                    D.append(d_fwd); LX.append(wheel_adv)
            t += DT
    return np.array(D), np.array(LX)


def report(run):
    d, wadv = collect(run)
    print(f"=== {run} ({len(d)} intervalos) ===")
    print(f"  d_fwd (avance GPS):      p10={np.percentile(d,10)*100:.1f}  "
          f"p50={np.percentile(d,50)*100:.1f}  p90={np.percentile(d,90)*100:.1f}  cm")
    print(f"  wheel_adv (avance rueda):p10={np.percentile(wadv,10)*100:.1f}  "
          f"p50={np.percentile(wadv,50)*100:.1f}  p90={np.percentile(wadv,90)*100:.1f}  cm")
    # ratio avance-GPS / avance-rueda (deberia ser ~1 si la rueda mide bien)
    valid = wadv > 0.02
    ratio = d[valid] / wadv[valid]
    print(f"  ratio d_fwd/wheel_adv:   p10={np.percentile(ratio,10):.2f}  "
          f"p50={np.percentile(ratio,50):.2f}  p90={np.percentile(ratio,90):.2f}  "
          f"(1.0 = rueda mide bien; >1 rueda subestima; <1 patina)")
    corr = np.corrcoef(wadv, d)[0, 1]
    print(f"  corr(wheel_adv, d_fwd):  {corr:.3f}")
    # tabla por franjas de avance-rueda
    print(f"  avance GPS medio por franja de avance-rueda:")
    edges = np.percentile(wadv, [0, 20, 40, 60, 80, 100])
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        mask = (wadv >= lo) & (wadv <= hi)
        if mask.sum() > 0:
            print(f"    rueda [{lo*100:5.1f},{hi*100:5.1f}]cm  (n={mask.sum():4d}): "
                  f"avance GPS medio = {d[mask].mean()*100:5.1f}cm")
    print()


print("Comparacion detallada de la relacion rueda->avance en ambos runs...\n")
report("Run_B")
report("Run_A")
print("=== LECTURA ===")
print("- Si en Run_B el ratio d_fwd/wheel_adv es ~1 y la tabla por franjas es")
print("  monotona -> la rueda mide bien, la baja corr venia del ruido; no hay problema.")
print("- Si el ratio se desvia mucho de 1 o la tabla NO es monotona -> hay")
print("  deslizamiento o desajuste real entre rueda y avance en Run_B.")