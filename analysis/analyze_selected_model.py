"""
analyze_selected_model.py -- Estadisticos de las componentes vertical y lateral del modelo
seleccionado (GRU, IMU + odometria de ruedas) sobre el conjunto de PRUEBA de la particion mixta.

Calcula (apartados 5.2.5.4 y 5.2.5.5 de la memoria):
  vertical : correlacion de Pearson y R^2 entre la prediccion y el valor real (media y desv. de 5 semillas)
  lateral  : correlacion de Pearson y R^2 del modelo; correlacion de la odometria de ruedas;
             desviacion tipica y valor absoluto medio del lateral real; avance medio y su desviacion;
             % de intervalos con |lateral| < UMBRAL_RUIDO_CM;
             autocorrelacion de retardo 1 del lateral real dentro de tramos continuos de >= 10 s
             (media ponderada por la longitud del tramo y mediana).

Entradas : results/final/gru_imuwheel_seed*_test.npz y results/baseline_mixed_imuwheel.npz
Salida   : results/analisis_modelo_seleccionado.json
Uso      : python analyze_selected_model.py
"""
import json
from pathlib import Path
import numpy as np

RES = Path(r"C:\Users\Usuario\Desktop\TFM\results")
UMBRAL_RUIDO_CM = 0.76      # umbral de ruido del DGPS empleado en el analisis previo (analyze_slip.py)
MIN_RUN = 10                # s: longitud minima de tramo continuo para la autocorrelacion


def runs(ts, seg):
    out = []
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]; idx = idx[np.argsort(ts[idx])]; cur = [idx[0]]
        for p, q in zip(idx[:-1], idx[1:]):
            if abs(ts[q] - ts[p] - 1) < 1e-3: cur.append(q)
            else: out.append(cur); cur = [q]
        out.append(cur)
    return out


def r2(p, t):
    return 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()


files = sorted((RES / "final").glob("gru_imuwheel_seed*_test.npz"))
lat_c, lat_r2, up_c, up_r2 = [], [], [], []
for f in files:
    m = np.load(f); p, t = m["pred"], m["true"]
    lat_c.append(np.corrcoef(p[:, 1], t[:, 1])[0, 1]); lat_r2.append(r2(p[:, 1], t[:, 1]))
    up_c.append(np.corrcoef(p[:, 2], t[:, 2])[0, 1]); up_r2.append(r2(p[:, 2], t[:, 2]))

t = m["true"] * 100                       # el valor real es el mismo en todas las semillas
ac, w = [], []
for r in runs(m["t_start"], m["seg_id"]):
    if len(r) >= MIN_RUN:
        x = t[r, 1]; ac.append(np.corrcoef(x[:-1], x[1:])[0, 1]); w.append(len(r))
b = np.load(RES / "baseline_mixed_imuwheel.npz")

out = {
    "n_semillas": len(files), "n_ventanas": int(len(t)),
    "vertical": {"pearson_media": float(np.mean(up_c)), "pearson_sd": float(np.std(up_c)),
                 "r2_media": float(np.mean(up_r2)), "r2_sd": float(np.std(up_r2))},
    "lateral": {"pearson_modelo_media": float(np.mean(lat_c)), "pearson_modelo_sd": float(np.std(lat_c)),
                "r2_modelo_media": float(np.mean(lat_r2)), "r2_modelo_sd": float(np.std(lat_r2)),
                "pearson_odometria": float(np.corrcoef(b["pred"][:, 1], b["true"][:, 1])[0, 1]),
                "real_sd_cm": float(t[:, 1].std()), "real_abs_medio_cm": float(np.abs(t[:, 1]).mean()),
                "pct_bajo_umbral_ruido": float((np.abs(t[:, 1]) < UMBRAL_RUIDO_CM).mean() * 100),
                "umbral_ruido_cm": UMBRAL_RUIDO_CM,
                "autocorr_lag1_media_ponderada": float(np.average(ac, weights=w)),
                "autocorr_lag1_mediana": float(np.median(ac)), "autocorr_n_tramos": len(ac),
                "autocorr_min_tramo_s": MIN_RUN},
    "avance": {"real_medio_cm": float(t[:, 0].mean()), "real_sd_cm": float(t[:, 0].std())},
}
(RES / "analisis_modelo_seleccionado.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
