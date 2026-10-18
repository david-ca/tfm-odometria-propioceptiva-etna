"""
calibrated_baseline.py -- Linea base de odometria de ruedas CALIBRADA, para una
comparacion exigente con el modelo.

Motivo: la odometria de ruedas presenta un sesgo sistematico en el avance
(subestima ~1,2-1,4 cm por intervalo en la particion mixta). Un sesgo constante
se acumula linealmente y explica gran parte de su error de avance acumulado.
Antes de atribuir la ventaja del modelo a algo mas que corregir ese sesgo, se
compara tambien con una odometria a la que se le corrige la escala con los
MISMOS datos de entrenamiento que ve el modelo:

    avance_cal = k * avance_ruedas      (k por minimos cuadrados, sin ordenada)
    variante afin: avance_cal = a * avance_ruedas + b

k (y a, b) se ajustan SOLO con las ventanas de ENTRENAMIENTO de la particion
mixta (Run A completo + 75 % de B) y se aplican a las ventanas de prueba.

Salida: results/baseline_calibrada_{cfg}.npz y results/baseline_calibrada.json,
con MAE/sesgo del avance por intervalo y error del avance acumulado por horizonte
(5-60 s) para: linea base sin calibrar, calibrada (escala), calibrada (afin) y el
modelo seleccionado (GRU IMU+ruedas, 5 semillas).

Uso:  python calibrated_baseline.py
"""
import csv, json
from pathlib import Path
import numpy as np

BASE = Path(r"C:\Users\Usuario\Desktop\TFM\extracted")
RES = Path(r"C:\Users\Usuario\Desktop\TFM\results")
DT = 1.0
H = [5, 10, 20, 30, 45, 60]
SEL = ("gru", "imuwheel")


def load(p):
    with open(p) as f:
        r = csv.reader(f); next(r)
        return np.array([[float(x) for x in row] for row in r])


WO = {}
for run in ["Run_A", "Run_B"]:
    w = load(BASE / run / "wheel_odometry.csv")
    WO[run] = (w[:, 0] / 1e9, w[:, 1])          # tiempo (s), lin_x


def wheel_fwd(run, t0):
    t, vx = WO[run]
    m = (t >= t0) & (t <= t0 + DT)
    if m.sum() < 2:
        return np.nan
    tt, v = t[m], vx[m]
    return float(np.sum((v[:-1] + v[1:]) / 2 * np.diff(tt)))


def runs(ts, seg):
    out = []
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]; idx = idx[np.argsort(ts[idx])]; cur = [idx[0]]
        for p, q in zip(idx[:-1], idx[1:]):
            if abs(ts[q] - ts[p] - 1) < 1e-3: cur.append(q)
            else: out.append(cur); cur = [q]
        out.append(cur)
    return out


def cum_err(pred, true, R, h):
    e = []
    for r in R:
        r = np.array(r)
        for k in range(len(r) - h + 1):
            w = r[k:k + h]; e.append(abs(pred[w].sum() - true[w].sum()))
    return float(np.mean(e) * 100) if e else float("nan")


cfg = SEL[1]
tr = np.load(BASE / f"dataset_{cfg}_mixed_train.npz", allow_pickle=True)
te = np.load(BASE / f"dataset_{cfg}_mixed_test.npz", allow_pickle=True)

# --- ajuste en ENTRENAMIENTO ---
run_of = lambda s: "Run_A" if s < 2000 else "Run_B"
wtr = np.array([wheel_fwd(run_of(s), t) for s, t in zip(tr["seg_id"], tr["t_start"])])
ytr = tr["y"][:, 0]
ok = ~np.isnan(wtr)
k = float(np.sum(wtr[ok] * ytr[ok]) / np.sum(wtr[ok] ** 2))
a, b = np.polyfit(wtr[ok], ytr[ok], 1)
print(f"Ajuste en entrenamiento ({ok.sum()} ventanas): escala k = {k:.4f} | afin a = {a:.4f}, b = {b*100:+.3f} cm")
print(f"  sesgo de la odometria en entrenamiento: {(wtr[ok]-ytr[ok]).mean()*100:+.2f} cm")

# --- prueba ---
wte = np.array([wheel_fwd("Run_B", t) for t in te["t_start"]])
yte = te["y"][:, 0]; R = runs(te["t_start"], te["seg_id"])
cands = {"sin_calibrar": wte, "calibrada_escala": k * wte, "calibrada_afin": a * wte + b}
out = {"k": k, "a": float(a), "b_cm": float(b * 100), "n_train": int(ok.sum()), "metodos": {}}
for name, p in cands.items():
    e = p - yte
    out["metodos"][name] = {"mae_fwd_cm": float(np.nanmean(np.abs(e)) * 100),
                            "sesgo_fwd_cm": float(np.nanmean(e) * 100),
                            "acumulado_cm": {h: cum_err(p, yte, R, h) for h in H}}
mods = []
for s in range(5):
    f = RES / "final" / f"{SEL[0]}_{cfg}_seed{s}_test.npz"
    if f.exists():
        m = np.load(f); assert np.allclose(m["true"][:, 0], yte)
        mods.append({"mae": np.abs(m["pred"][:, 0] - yte).mean() * 100,
                     "sesgo": (m["pred"][:, 0] - yte).mean() * 100,
                     "acc": [cum_err(m["pred"][:, 0], yte, R, h) for h in H]})
out["metodos"]["modelo_GRU_imuwheel"] = {
    "mae_fwd_cm": float(np.mean([x["mae"] for x in mods])),
    "sesgo_fwd_cm": float(np.mean([x["sesgo"] for x in mods])),
    "acumulado_cm": {h: float(np.mean([x["acc"][i] for x in mods])) for i, h in enumerate(H)},
    "acumulado_sd_cm": {h: float(np.std([x["acc"][i] for x in mods])) for i, h in enumerate(H)}}

print(f"\n{'metodo':22s} {'MAE':>6s} {'sesgo':>7s}   " + "  ".join(f"{h:>4d}s" for h in H))
for name, r in out["metodos"].items():
    print(f"{name:22s} {r['mae_fwd_cm']:6.2f} {r['sesgo_fwd_cm']:+7.2f}   " +
          "  ".join(f"{r['acumulado_cm'][h]:5.1f}" for h in H))
print("(MAE y sesgo por intervalo en cm; columnas = error medio del avance acumulado en cm)")
(RES / "baseline_calibrada.json").write_text(json.dumps(out, indent=2))
np.savez_compressed(RES / f"baseline_calibrada_{cfg}.npz", wheel=wte, k=k, a=a, b=b,
                    true=yte, seg_id=te["seg_id"], t_start=te["t_start"])
