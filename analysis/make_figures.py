"""
make_figures.py -- Figuras y tablas de resultados de la memoria a partir de results/.
Uso:  python make_figures.py <ruta_results> <ruta_salida>
Genera: curvas de entrenamiento (6 + rejilla), comparacion de modelos, error del
avance acumulado por horizonte, error por rango de velocidad, y tablas CSV.
"""
import sys, json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RES = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
ARCH = ["gru", "tcn", "transformer"]; ARCH_N = {"gru": "GRU", "tcn": "TCN", "transformer": "Transformer"}
CFG = ["imu", "imuwheel"]; CFG_N = {"imu": "Solo IMU", "imuwheel": "IMU + odometría de ruedas"}
C_TRAIN, C_VAL = "#2a78d6", "#eb6834"
C_CFG = {"imu": "#2a78d6", "imuwheel": "#eb6834"}
C_BASE = "#8a8a85"; INK = "#2b2b2b"; GRID = "#e4e2dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#9a9890", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "savefig.dpi": 220, "savefig.bbox": "tight"})

def rec(a, c, s): return json.loads((RES / "final" / f"{a}_{c}_seed{s}.json").read_text())
def seeds(a, c): return sorted(int(p.stem.split("seed")[1]) for p in (RES / "final").glob(f"{a}_{c}_seed*.json"))
def median_val_seed(a, c):
    S = seeds(a, c); v = [rec(a, c, s)["best_val_mae"] for s in S]
    return S[int(np.argsort(v)[len(v) // 2])]

# ---------------- 1. curvas de entrenamiento ----------------
def plot_curve(ax, a, c, legend=True):
    ymax = 0
    for s in seeds(a, c):
        h = rec(a, c, s)["history"]; ep = [r["epoch"] for r in h]
        ax.plot(ep, [r["val_mse"] * 1e4 for r in h], color=C_VAL, lw=0.8, alpha=0.25)
    s = median_val_seed(a, c); r = rec(a, c, s); h = r["history"]; ep = [x["epoch"] for x in h]
    tr = [x["train_mse"] * 1e4 for x in h]; va = [x["val_mse"] * 1e4 for x in h]
    ax.plot(ep, tr, color=C_TRAIN, lw=2, label="Entrenamiento (MSE)")
    ax.plot(ep, va, color=C_VAL, lw=2, label="Validación (MSE)")
    ax.axvline(r["best_epoch"], color=INK, lw=1, ls="--")
    ax.text(r["best_epoch"], 0.97, f" época {r['best_epoch']}", transform=ax.get_xaxis_transform(),
            va="top", fontsize=8, color=INK)
    ax.set_xlabel("Época"); ax.set_ylabel("MSE (cm²)")
    ax.set_title(f"{ARCH_N[a]} · {CFG_N[c]}", fontsize=9.5, color=INK, loc="left")
    if legend: ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1, 0.88))
    return max(tr[1:] + va)

ylim = 0
for a in ARCH:
    for c in CFG:
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        ylim = max(ylim, plot_curve(ax, a, c)); plt.close(fig)
for a in ARCH:
    for c in CFG:
        fig, ax = plt.subplots(figsize=(5.2, 3.0)); plot_curve(ax, a, c); ax.set_ylim(0, ylim * 1.05)
        fig.savefig(OUT / f"fig_entrenamiento_{a}_{c}.png"); plt.close(fig)
fig, axs = plt.subplots(3, 2, figsize=(7.2, 8.2), sharey=True)
for i, a in enumerate(ARCH):
    for j, c in enumerate(CFG):
        plot_curve(axs[i, j], a, c, legend=False); axs[i, j].set_ylim(0, ylim * 1.05)
h_, l_ = axs[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 1.01))
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / "fig_entrenamiento_rejilla.png"); plt.close(fig)

# ---------------- 2. comparacion de modelos (prueba) ----------------
base = {c: json.loads((RES / "baseline_mixed_summary.json").read_text())[c] for c in CFG}
comp = ["Avance", "Deslizamiento lateral", "Vertical"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9))
groups = ARCH + ["base"]; gname = ["GRU", "TCN", "Transf.", "Línea\nbase"]
for k, ax in enumerate(axs):
    for j, c in enumerate(CFG):
        xs, ms, sd = [], [], []
        for g, a in enumerate(groups):
            if a == "base":
                ms.append(base[c]["mae_cm"][k]); sd.append(0)
            else:
                v = [rec(a, c, s)["test"]["mae"][k] * 100 for s in seeds(a, c)]
                ms.append(np.mean(v)); sd.append(np.std(v))
            xs.append(g + (j - 0.5) * 0.38)
        bars = ax.bar(xs, ms, width=0.36, color=C_CFG[c], yerr=sd,
                      error_kw={"lw": 0.8, "capsize": 2, "ecolor": INK}, edgecolor="white", linewidth=1)
        bars[-1].set_facecolor("white"); bars[-1].set_edgecolor(C_CFG[c]); bars[-1].set_hatch("////")
    ax.set_xticks(range(4)); ax.set_xticklabels(gname, fontsize=8)
    ax.set_title(comp[k], fontsize=9.5, loc="left", color=INK); ax.grid(axis="x", visible=False)
    if k == 0: ax.set_ylabel("MAE por intervalo (cm)")
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_CFG[c]) for c in CFG]
           + [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, hatch="////")],
           labels=[CFG_N[c] for c in CFG] + ["Línea base (mismas ventanas)"], fontsize=8,
           loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT / "fig_comparacion_modelos.png"); plt.close(fig)

# ---------------- 3. analisis del modelo seleccionado ----------------
SEL_A, SEL_C = "gru", "imuwheel"
b = np.load(RES / f"baseline_mixed_{SEL_C}.npz")

def runs(ts, seg):
    out = []
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]; idx = idx[np.argsort(ts[idx])]; cur = [idx[0]]
        for p, q in zip(idx[:-1], idx[1:]):
            if abs(ts[q] - ts[p] - 1) < 1e-3: cur.append(q)
            else: out.append(cur); cur = [q]
        out.append(cur)
    return out

H = [5, 10, 20, 30, 45, 60]; acc = {h: [] for h in H}; accb = {}; wil = []
for s in seeds(SEL_A, SEL_C):
    m = np.load(RES / "final" / f"{SEL_A}_{SEL_C}_seed{s}_test.npz")
    assert np.allclose(m["true"], b["true"])
    em = np.abs(m["pred"][:, 0] - m["true"][:, 0]); eb = np.abs(b["pred"][:, 0] - b["true"][:, 0])
    w = stats.wilcoxon(em, eb)
    wil.append({"seed": s, "mae_model_cm": float(em.mean() * 100), "mae_base_cm": float(eb.mean() * 100),
                "pct_windows_model_better": float((em < eb).mean() * 100), "wilcoxon_stat": float(w.statistic),
                "wilcoxon_p": float(w.pvalue)})
    R = runs(m["t_start"], m["seg_id"])
    for h in H:
        e1, e2, tt = [], [], []
        for r in R:
            r = np.array(r)
            for k0 in range(len(r) - h + 1):
                w_ = r[k0:k0 + h]; t = m["true"][w_, 0].sum()
                e1.append(abs(m["pred"][w_, 0].sum() - t)); e2.append(abs(b["pred"][w_, 0].sum() - t)); tt.append(t)
        e1, e2, tt = map(np.array, (e1, e2, tt))
        acc[h].append((e1.mean() * 100, (e1 / tt).mean() * 100, len(e1))); accb[h] = (e2.mean() * 100, (e2 / tt).mean() * 100)
json.dump(wil, open(OUT / "tabla_wilcoxon.json", "w", encoding="utf-8"), indent=2)
rows = ["horizonte_s,n_ventanas,modelo_cm,modelo_sd,base_cm,modelo_rel_pct,base_rel_pct"]
for h in H:
    v = np.array(acc[h]); rows.append(f"{h},{int(v[0,2])},{v[:,0].mean():.2f},{v[:,0].std():.2f},{accb[h][0]:.2f},{v[:,1].mean():.2f},{accb[h][1]:.2f}")
(OUT / "tabla_avance_acumulado.csv").write_text("\n".join(rows), encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(5.2, 3.0))
mv = [np.mean([x[0] for x in acc[h]]) for h in H]; ms_ = [np.std([x[0] for x in acc[h]]) for h in H]
ax.plot(H, [accb[h][0] for h in H], color=C_BASE, lw=2, marker="o", ms=5, label="Odometría de ruedas (línea base)")
calp = RES / "baseline_calibrada.json"
if calp.exists():
    cal = json.loads(calp.read_text())["metodos"]["calibrada_escala"]["acumulado_cm"]
    ax.plot(H, [cal[str(h)] for h in H], color=C_BASE, lw=2, ls="--", marker="s", ms=5,
            label="Odometría de ruedas calibrada (escala)")
ax.errorbar(H, mv, yerr=ms_, color=C_CFG["imuwheel"], lw=2, marker="o", ms=5, capsize=2, label="GRU · IMU + odometría")
ax.set_xlabel("Horizonte de acumulación (s)"); ax.set_ylabel("Error del avance acumulado (cm)")
ax.legend(fontsize=8, loc="upper left"); fig.savefig(OUT / "fig_avance_acumulado.png"); plt.close(fig)

s_med = median_val_seed(SEL_A, SEL_C); m = np.load(RES / "final" / f"{SEL_A}_{SEL_C}_seed{s_med}_test.npz")
tf = m["true"][:, 0] * 100; edges = [-5, 12, 15, 18, 30]; lab = ["< 12", "12–15", "15–18", "≥ 18"]
rows = ["rango_cm,n,mae_modelo,mae_base,sesgo_modelo,sesgo_base"]; mm, bb, nn = [], [], []
for lo, hi, l in zip(edges[:-1], edges[1:], lab):
    k = (tf >= lo) & (tf < hi)
    e1 = m["pred"][k, 0] - m["true"][k, 0]; e2 = b["pred"][k, 0] - b["true"][k, 0]
    mm.append(np.abs(e1).mean() * 100); bb.append(np.abs(e2).mean() * 100); nn.append(k.sum())
    rows.append(f"{l},{k.sum()},{mm[-1]:.2f},{bb[-1]:.2f},{e1.mean()*100:+.2f},{e2.mean()*100:+.2f}")
(OUT / "tabla_error_por_velocidad.csv").write_text("\n".join(rows), encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(5.2, 3.0)); x = np.arange(len(lab))
ax.bar(x - 0.19, mm, 0.36, color=C_CFG["imuwheel"], label="GRU · IMU + odometría", edgecolor="white")
ax.bar(x + 0.19, bb, 0.36, color=C_BASE, label="Odometría de ruedas (línea base)", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(n = {n})" for l, n in zip(lab, nn)])
ax.set_xlabel("Velocidad real (cm/s)"); ax.set_ylabel("MAE del avance (cm)")
ax.grid(axis="x", visible=False); ax.legend(fontsize=8)
fig.savefig(OUT / "fig_error_por_velocidad.png"); plt.close(fig)

# ---------------- 4. tablas ----------------
rows = ["arquitectura,entrada,hiperparametros,parametros,mae_avance,sd,mae_lateral,sd,mae_vertical,sd,mae_global,sd,"
        "rmse_avance,rmse_lateral,rmse_vertical,sesgo_avance,epoca_sel,epocas,t_entren_s,t_epoca_s,inf_lote_ms,inf_unit_ms"]
for c in CFG:
    for a in ARCH:
        R = [rec(a, c, s) for s in seeds(a, c)]
        f = lambda g: (np.mean([g(r) for r in R]), np.std([g(r) for r in R]))
        cells = [ARCH_N[a], CFG_N[c], R[0]["hp"] and json.dumps({k: v for k, v in R[0]["hp"].items()}).replace(",", ";"), str(R[0]["n_params"])]
        for k in range(3):
            mu, sd = f(lambda r: r["test"]["mae"][k] * 100); cells += [f"{mu:.2f}", f"{sd:.2f}"]
        mu, sd = f(lambda r: r["test"]["mae_g"] * 100); cells += [f"{mu:.2f}", f"{sd:.2f}"]
        for k in range(3): cells.append(f"{f(lambda r: r['test']['rmse'][k]*100)[0]:.2f}")
        cells.append(f"{f(lambda r: r['test']['bias'][0]*100)[0]:+.2f}")
        for g in [lambda r: r["best_epoch"], lambda r: r["epochs_run"], lambda r: r["train_time_s"],
                  lambda r: r["time_per_epoch_s"], lambda r: r["infer_ms_per_window_batched"],
                  lambda r: r["infer_ms_per_window_single"]]:
            cells.append(f"{f(g)[0]:.3f}")
        rows.append(",".join(cells))
(OUT / "tabla_modelos_finales.csv").write_text("\n".join(rows), encoding="utf-8-sig")
import csv as _csv
hp = list(_csv.DictReader(open(RES / "hpsearch.csv")))
sel = json.loads((RES / "selected_hparams.json").read_text())
agg = {}
for r in hp:
    agg.setdefault((r["arch"], r["cfg"], r["key"]), []).append(r)
rows = ["arquitectura,entrada,combinacion,fase,parametros,val_mae_cm,sd,epoca_sel_media,epocas_max,seg_por_ejecucion,elegida"]
for (a, c, k), rs in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1], np.mean([float(x["val_mae_cm"]) for x in kv[1]]))):
    v = np.array([float(x["val_mae_cm"]) for x in rs])
    rows.append(f"{ARCH_N[a]},{CFG_N[c]},{k.replace('|', ' ')},{rs[0]['phase']},{rs[0]['n_params']},{v.mean():.3f},{v.std(ddof=0):.3f},"
                f"{np.mean([int(x['best_epoch']) for x in rs]):.1f},{max(int(x['epochs_run']) for x in rs)},"
                f"{np.mean([float(x['secs']) for x in rs]):.1f},{'SI' if sel[f'{a}_{c}']['key'] == k else ''}")
(OUT / "tabla_busqueda_hiperparametros.csv").write_text("\n".join(rows), encoding="utf-8-sig")
print("Figuras y tablas en", OUT)