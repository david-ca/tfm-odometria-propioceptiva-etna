"""
reconstruct_mixed.py -- Reconstruccion de trayectoria sobre los tramos de PRUEBA de
la particion mixta (Run B), con el modelo SELECCIONADO (GRU, IMU + odometria de
ruedas) y comparacion equitativa con la odometria de ruedas (sin calibrar y
calibrada), usando EXACTAMENTE el mismo rumbo para todos los metodos.

Por que asi (respuesta a la nota [140] del tutor): no se vuelve al recorrido A ni
al modelo de la particion entre recorridos. Se reconstruyen los tramos de prueba
de B, que son continuos a trozos. Cada tramo continuo (ventanas consecutivas de
1 s, sin huecos) se reconstruye por separado, anclado en su posicion DGPS
inicial.

Rumbo (dos escenarios, los mismos para todos los metodos):
  ideal : direccion de movimiento del DGPS en el intervalo previo (la misma
          referencia con la que se construye el objetivo; con los incrementos
          reales reconstruye la trayectoria exacta -> se comprueba al inicio).
  imu   : rumbo inicial del tramo tomado del DGPS (se supone conocido al
          anclar) + variacion de la guinada del filtro de Madgwick desde el
          inicio del tramo. Es el escenario realista: solo propiocepcion
          despues del anclaje.

Metodos (todos en 2D para la comparacion; el modelo tambien da la vertical):
  modelo       : incrementos (avance, lateral) predichos por la GRU (5 semillas)
  ruedas       : integral de (lin_x, lin_y) de la odometria en cada intervalo
  ruedas_cal   : ruedas con el avance escalado por k (results/baseline_calibrada.json)

Metricas por tramo (tramos de al menos MIN_LEN s):
  deriva_final_pct : error de posicion horizontal al final / distancia recorrida
  rmse_m           : RMSE de la posicion horizontal a lo largo del tramo
Se resumen como mediana y media sobre tramos, y RMSE global (todos los puntos).

Salida: results/trayectoria_mixta.json y results/fig_trayectoria_B.png
Uso:  python reconstruct_mixed.py
"""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(r"C:\Users\Usuario\Desktop\TFM\extracted")
RES = Path(r"C:\Users\Usuario\Desktop\TFM\results")
CFG = "imuwheel"; ARCH = "gru"
DT = 1.0
MIN_LEN = 30          # s: tramos mas cortos no se usan para las metricas


def load(p):
    with open(p) as f:
        r = csv.reader(f); next(r)
        return np.array([[float(x) for x in row] for row in r])


gt = load(BASE / "Run_B" / "ground_truth.csv")
ori = load(BASE / "Run_B" / "orientation_full.csv")
wo = load(BASE / "Run_B" / "wheel_odometry.csv")
ori_t = ori[:, 0] / 1e9; yaw = np.unwrap(np.deg2rad(ori[:, 7]))
wo_t = wo[:, 0] / 1e9
k = json.loads((RES / "baseline_calibrada.json").read_text())["k"]


def gpos(t):
    return np.array([np.interp(t, gt[:, 0], gt[:, c]) for c in (1, 2, 3)])


def yaw_at(t):
    return np.interp(t, ori_t, yaw)


def wheel(t0):
    m = (wo_t >= t0) & (wo_t <= t0 + DT)
    if m.sum() < 2:
        return 0.0, 0.0
    tt, dts = wo_t[m], np.diff(wo_t[m])
    return (float(np.sum((wo[m, 1][:-1] + wo[m, 1][1:]) / 2 * dts)),
            float(np.sum((wo[m, 2][:-1] + wo[m, 2][1:]) / 2 * dts)))


def runs(ts, seg):
    out = []
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]; idx = idx[np.argsort(ts[idx])]; cur = [idx[0]]
        for p, q in zip(idx[:-1], idx[1:]):
            if abs(ts[q] - ts[p] - 1) < 1e-3: cur.append(q)
            else: out.append(cur); cur = [q]
        out.append(cur)
    return out


def heading_ideal(t, mode):
    d = (gpos(t) - gpos(t - DT)) if mode == "prev" else (gpos(t + DT) - gpos(t))
    return np.arctan2(d[1], d[0])


def integrate(inc, ts, hmode, ideal_conv):
    """inc: (n,2) incrementos (avance, lateral) en orden temporal. Devuelve (n+1,2)."""
    p = gpos(ts[0])[:2].copy(); out = [p.copy()]
    h0 = heading_ideal(ts[0], ideal_conv); y0 = yaw_at(ts[0])
    for (df, dl), t in zip(inc, ts):
        h = heading_ideal(t, ideal_conv) if hmode == "ideal" else h0 + (yaw_at(t) - y0)
        c, s = np.cos(h), np.sin(h)
        p = p + np.array([c * df - s * dl, s * df + c * dl]); out.append(p.copy())
    return np.array(out)


def ref_path(ts):
    return np.array([gpos(ts[0])[:2]] + [gpos(t + DT)[:2] for t in ts])


def metrics(est, ref):
    e = np.linalg.norm(est - ref, axis=1)
    dist = np.sum(np.linalg.norm(np.diff(ref, axis=0), axis=1))
    return e[-1] / dist * 100, np.sqrt((e ** 2).mean()), e, dist


# ------------------------------------------------------------------ datos
seeds = sorted(int(p.stem.split("seed")[1].split("_")[0]) for p in (RES / "final").glob(f"{ARCH}_{CFG}_seed*_test.npz"))
M = {s: np.load(RES / "final" / f"{ARCH}_{CFG}_seed{s}_test.npz") for s in seeds}
ts_all, seg_all, y_all = M[seeds[0]]["t_start"], M[seeds[0]]["seg_id"], M[seeds[0]]["true"]
R = [np.array(r) for r in runs(ts_all, seg_all) if len(r) >= MIN_LEN]
W = np.array([wheel(t) for t in ts_all])
print(f"Tramos continuos de prueba con >= {MIN_LEN} s: {len(R)} "
      f"({sum(len(r) for r in R)} s de {len(ts_all)} s de prueba)")

# --- comprobacion del convenio de rumbo ideal con los incrementos REALES ---
conv = None
for c in ("prev", "next"):
    errs = [metrics(integrate(y_all[r, :2], ts_all[r], "ideal", c), ref_path(ts_all[r]))[1] for r in R]
    print(f"  comprobacion convenio '{c}': RMSE con incrementos reales = {np.mean(errs)*100:.2f} cm")
    if conv is None or np.mean(errs) < best:
        conv, best = c, np.mean(errs)
print(f"  -> convenio de rumbo ideal: '{conv}' (debe dar un RMSE casi nulo)")

methods = {"ruedas": W, "ruedas_cal": np.column_stack([k * W[:, 0], W[:, 1]])}
for s in seeds:
    methods[f"modelo_s{s}"] = M[s]["pred"][:, :2]

out = {"min_len_s": MIN_LEN, "n_tramos": len(R), "segundos": int(sum(len(r) for r in R)),
       "convenio_rumbo_ideal": conv, "k": k, "resultados": {}}
per_run = {}
for hm in ["ideal", "imu"]:
    for name, inc in methods.items():
        d, r_, allerr = [], [], []
        for r in R:
            est = integrate(inc[r], ts_all[r], hm, conv); ref = ref_path(ts_all[r])
            dp, rm, e, dist = metrics(est, ref); d.append(dp); r_.append(rm); allerr.append(e)
            per_run[(hm, name, int(r[0]))] = (est, ref)
        allerr = np.concatenate(allerr)
        out["resultados"][f"{hm}|{name}"] = {"deriva_final_pct_mediana": float(np.median(d)),
                                             "deriva_final_pct_media": float(np.mean(d)),
                                             "rmse_m_mediana": float(np.median(r_)),
                                             "rmse_global_m": float(np.sqrt((allerr ** 2).mean()))}

# resumen con el modelo promediado sobre semillas
print(f"\n{'rumbo':6s} {'metodo':12s} {'deriva final (mediana / media)':>32s} {'RMSE global':>12s}")
summary = {}
for hm in ["ideal", "imu"]:
    for name in ["ruedas", "ruedas_cal", "modelo"]:
        keys = [f"{hm}|{n}" for n in methods if (n == name or (name == "modelo" and n.startswith("modelo_")))]
        vals = {m: np.mean([out["resultados"][kk][m] for kk in keys]) for m in out["resultados"][keys[0]]}
        sd = np.std([out["resultados"][kk]["rmse_global_m"] for kk in keys]) if name == "modelo" else 0
        summary[f"{hm}|{name}"] = {**{m: float(v) for m, v in vals.items()}, "rmse_global_sd_m": float(sd)}
        print(f"{hm:6s} {name:12s} {vals['deriva_final_pct_mediana']:13.2f} % / {vals['deriva_final_pct_media']:6.2f} %"
              f" {vals['rmse_global_m']:10.3f} m" + (f" (+/- {sd:.3f})" if sd else ""))
out["resumen"] = summary
(RES / "trayectoria_mixta.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

# ------------------------------------------------------------------ figura
vals = {s: json.loads((RES / "final" / f"{ARCH}_{CFG}_seed{s}.json").read_text())["best_val_mae"] for s in seeds}
s_med = sorted(seeds, key=lambda s: vals[s])[len(seeds) // 2]
longest = sorted(R, key=len, reverse=True)[:2]
COL = {"ref": "#2b2b2b", f"modelo_s{s_med}": "#eb6834", "ruedas": "#8a8a85", "ruedas_cal": "#2a78d6"}
LAB = {"ref": "Referencia (DGPS)", f"modelo_s{s_med}": "GRU · IMU + odometría",
       "ruedas": "Odometría de ruedas", "ruedas_cal": "Odometría calibrada"}
fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.6))
for i, r in enumerate(longest):
    for j, hm in enumerate(["ideal", "imu"]):
        ax = axs[i, j]
        est, ref = per_run[(hm, "ruedas", int(r[0]))]
        ax.plot(ref[:, 0], ref[:, 1], color=COL["ref"], lw=2.2, label=LAB["ref"])
        for name in ["ruedas", "ruedas_cal", f"modelo_s{s_med}"]:
            est, _ = per_run[(hm, name, int(r[0]))]
            ax.plot(est[:, 0], est[:, 1], color=COL[name], lw=1.6, label=LAB[name])
        ax.plot(*ref[0], "o", color=COL["ref"], ms=5)
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(color="#e4e2dc", lw=0.6)
        ax.set_title(f"Tramo de {len(r)} s · rumbo {'de referencia' if hm == 'ideal' else 'de la IMU'}",
                     fontsize=9, loc="left")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for i, lab in enumerate(["a)", "b)"]):
    axs[i, 0].text(-0.22, 1.08, lab, transform=axs[i, 0].transAxes,
                    fontsize=11, fontweight="bold", va="bottom")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.01))
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(RES / "fig_trayectoria_B.png", dpi=220, bbox_inches="tight")
print(f"\nGuardado: trayectoria_mixta.json y fig_trayectoria_B.png (figura con la semilla {s_med})")