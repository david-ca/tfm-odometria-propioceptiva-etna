"""
baseline_mixed_paired.py -- Linea base (odometria de ruedas) evaluada EXACTAMENTE
sobre las mismas ventanas de prueba de la particion mixta que los modelos.

Para cada ventana de prueba [t_start, t_start + 1 s]:
  - prediccion de la linea base: integral trapezoidal de (lin_x, lin_y) de la
    odometria de ruedas en el intervalo -> (avance, lateral) en el marco del
    cuerpo (misma funcion wheel_disp que baseline_increments.py);
  - componente vertical: 0 (la odometria es 2D);
  - referencia: el objetivo y de la propia ventana (mismo que ven los modelos).
Al usar las mismas ventanas, la comparacion modelo-linea base es PAREADA
(permite un test estadistico ventana a ventana).

Requiere los .npz mixtos con t_start (build_dataset_mixed.py v2).
Salida: results/baseline_mixed_{cfg}.npz y resumen por pantalla y JSON.

Uso:  python baseline_mixed_paired.py
"""
import csv, json
from pathlib import Path
import numpy as np

BASE = Path(r"C:\Users\Usuario\Desktop\TFM\extracted")
OUT = Path(r"C:\Users\Usuario\Desktop\TFM\results")
DT = 1.0


def load(p):
    with open(p) as f:
        r = csv.reader(f); next(r)
        return np.array([[float(x) for x in row] for row in r])


wo = load(BASE / "Run_B" / "wheel_odometry.csv")
wo_t = wo[:, 0] / 1e9


def wheel_disp(t0, t1):
    m = (wo_t >= t0) & (wo_t <= t1)
    if m.sum() < 2:
        return None
    tt = wo_t[m]; vx = wo[m, 1]; vy = wo[m, 2]
    dts = np.diff(tt)
    return (float(np.sum((vx[:-1] + vx[1:]) / 2 * dts)),
            float(np.sum((vy[:-1] + vy[1:]) / 2 * dts)))


OUT.mkdir(parents=True, exist_ok=True)
summary = {}
for cfg in ["imu", "imuwheel"]:
    d = np.load(BASE / f"dataset_{cfg}_mixed_test.npz", allow_pickle=True)
    y, ts, seg = d["y"], d["t_start"], d["seg_id"]
    pred = np.full_like(y, np.nan)
    for i, t in enumerate(ts):
        wd = wheel_disp(t, t + DT)
        if wd is not None:
            pred[i] = [wd[0], wd[1], 0.0]
    valid = ~np.isnan(pred[:, 0])
    e = pred[valid] - y[valid]
    rec = {"n_windows": int(len(y)), "n_valid": int(valid.sum()),
           "mae_cm": (np.abs(e).mean(0) * 100).tolist(),
           "rmse_cm": (np.sqrt((e ** 2).mean(0)) * 100).tolist(),
           "bias_cm": (e.mean(0) * 100).tolist(),
           "mean_abs_true_cm": (np.abs(y[valid]).mean(0) * 100).tolist(),
           "note": "componente vertical: la linea base predice 0 (2D)"}
    summary[cfg] = rec
    np.savez_compressed(OUT / f"baseline_mixed_{cfg}.npz", pred=pred, true=y,
                        seg_id=seg, t_start=ts, valid=valid)
    print(f"[{cfg}] ventanas {rec['n_valid']}/{rec['n_windows']}  "
          f"MAE fwd/lat/up = {rec['mae_cm'][0]:.2f}/{rec['mae_cm'][1]:.2f}/{rec['mae_cm'][2]:.2f} cm  "
          f"sesgo fwd {rec['bias_cm'][0]:+.2f} cm")
(OUT / "baseline_mixed_summary.json").write_text(json.dumps(summary, indent=2))
