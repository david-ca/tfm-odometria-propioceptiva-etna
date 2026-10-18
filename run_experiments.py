"""
run_experiments.py -- Experimentos finales de la memoria (SOLO particion MIXTA).

Responde a las correcciones del tutor (08/10/2026):
  - 6 modelos: 3 arquitecturas (GRU, TCN, Transformer) x 2 entradas (imu, imuwheel).
  - Busqueda de hiperparametros para LAS TRES arquitecturas (no solo la GRU),
    guiada por el error de VALIDACION (el test no se usa para elegir nada).
  - Para cada uno de los 6 modelos finales se guarda: historial por epoca
    (MSE de entrenamiento, MSE y MAE de validacion por componente), epoca
    seleccionada por parada temprana, tiempo de entrenamiento, tiempo de
    inferencia, numero de parametros y predicciones sobre el conjunto de PRUEBA.
  - Resumen de la particion (ventanas y duracion de train / val / test).

Etapas (--stage):
  smoke  : prueba rapida (1 epoca, 1 semilla, 1 combinacion por arquitectura y
           entrada) para comprobar que todo funciona y estimar tiempos.
  split  : resumen de la particion -> results/split_info.json
  search : busqueda de hiperparametros -> results/hpsearch.csv y
           results/selected_hparams.json
  final  : entrena los 6 modelos con los hiperparametros elegidos y N semillas
           -> results/final/*.json, *.npz y results/summary_final.csv
  all    : split + search + final

Busqueda de hiperparametros (por arquitectura y entrada):
  Fase A (estructura, lr = 1e-3): rejilla 3x3 de los dos hiperparametros
    estructurales de cada arquitectura:
      GRU         : hidden {32, 64, 128}   x capas {1, 2, 3}
      TCN         : canales {32, 64, 128}  x niveles {3, 4, 5}
      Transformer : d_model {32, 64, 128}  x capas {1, 2, 3}
  Fase B (tasa de aprendizaje): lr {3e-4, 3e-3} sobre la mejor estructura de A.
  Fijos: dropout 0.2, batch 128, Adam, MSE, max 60 epocas, paciencia 10.
         TCN: kernel 3. Transformer: 4 cabezas, dim_ff = 2*d_model.
  Seleccion (regla de una desviacion): entre las combinaciones cuyo val_MAE
  medio no supera (mejor media + desv. de la mejor), se elige la de MENOS
  parametros. Si todas empatan dentro del ruido de semilla, gana la mas ligera.

Resumible: cada ejecucion se guarda nada mas terminar; si se interrumpe y se
relanza, salta lo ya hecho.

Uso:
  python run_experiments.py --stage smoke
  python run_experiments.py --stage all
  (opcional) --search_seeds 2 --final_seeds 5 --cfgs imu imuwheel
"""
import argparse, csv, json, time, itertools
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from dataset import (make_loaders_mixed, load_test_mixed, _load_mixed,
                     split_by_segments, VAL_FRAC, SEED)
from train import GRURegressor, TCNRegressor, TransformerRegressor, metrics, evaluate

ARCHS = ["gru", "tcn", "transformer"]
STRUCT = {   # (nombre1, valores1, nombre2, valores2)
    "gru":         ("hidden",   [32, 64, 128], "layers", [1, 2, 3]),
    "tcn":         ("channels", [32, 64, 128], "levels", [3, 4, 5]),
    "transformer": ("d_model",  [32, 64, 128], "layers", [1, 2, 3]),
}
FIXED = {"dropout": 0.2, "batch": 128, "lr": 1e-3, "kernel": 3, "nhead": 4}
LR_SWEEP = [3e-4, 3e-3]
DT = 1.0


# ----------------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------------
def build(arch, n_ch, hp):
    if arch == "gru":
        return GRURegressor(n_ch, hidden=hp["hidden"], layers=hp["layers"],
                            dropout=hp["dropout"])
    if arch == "tcn":
        return TCNRegressor(n_ch, channels=hp["channels"], levels=hp["levels"],
                            kernel=hp["kernel"], dropout=hp["dropout"])
    if arch == "transformer":
        return TransformerRegressor(n_ch, d_model=hp["d_model"], nhead=hp["nhead"],
                                    layers=hp["layers"], dim_ff=2 * hp["d_model"],
                                    dropout=hp["dropout"])
    raise ValueError(arch)


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def hp_key(arch, hp):
    a, _, b, _ = STRUCT[arch]
    return f"{a}={hp[a]}|{b}={hp[b]}|lr={hp['lr']:.0e}"


# ----------------------------------------------------------------------------
# Entrenamiento (mismo protocolo que train.py: Adam, MSE, early stopping sobre
# el MAE global de validacion)
# ----------------------------------------------------------------------------
_LOADERS = {}


def loaders(base, cfg, batch):
    k = (cfg, batch)
    if k not in _LOADERS:
        _LOADERS[k] = make_loaders_mixed(base, cfg, batch)
    return _LOADERS[k]


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def train_run(arch, cfg, hp, seed, args, device):
    torch.manual_seed(seed); np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    tr, va, info = loaders(args.base, cfg, hp["batch"])
    model = build(arch, info["n_channels"], hp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    loss_fn = nn.MSELoss()

    hist = []
    best_val, best_ep, best_state, bad = float("inf"), -1, None, 0
    sync(device); t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            tl += loss.item() * len(xb); n += len(xb)
        P, T = evaluate(model, va, device)
        mae, rmse, mae_g, rmse_g = metrics(P, T)
        hist.append({"epoch": ep, "train_mse": tl / n,
                     "val_mse": float(((P - T) ** 2).mean()),
                     "val_mae": float(mae_g), "val_rmse": float(rmse_g),
                     "val_mae_fwd": float(mae[0]), "val_mae_lat": float(mae[1]),
                     "val_mae_up": float(mae[2])})
        if mae_g < best_val:
            best_val, best_ep, bad = mae_g, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    sync(device); train_s = time.time() - t0
    model.load_state_dict(best_state)
    return model, {"history": hist, "best_epoch": best_ep, "epochs_run": len(hist),
                   "best_val_mae": float(best_val), "train_time_s": train_s,
                   "time_per_epoch_s": train_s / len(hist), "n_params": n_params(model)}


def time_inference(model, X, device, n_single=300):
    """ms por ventana: (a) en lote de 256 sobre todo el test, (b) ventana a ventana."""
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(3):                                   # calentamiento
            model(Xt[:256].to(device))
        sync(device); t0 = time.time()
        for i in range(0, len(Xt), 256):
            model(Xt[i:i + 256].to(device))
        sync(device); batched = (time.time() - t0) / len(Xt) * 1e3
        m = min(n_single, len(Xt))
        sync(device); t0 = time.time()
        for i in range(m):
            model(Xt[i:i + 1].to(device))
        sync(device); single = (time.time() - t0) / m * 1e3
    return batched, single


# ----------------------------------------------------------------------------
# Etapa: resumen de la particion
# ----------------------------------------------------------------------------
def seg_duration(ts, seg):
    """Duracion cubierta (s) = suma por tramo de (ultimo inicio - primer inicio + DT)."""
    return float(sum(ts[seg == s].max() - ts[seg == s].min() + DT for s in np.unique(seg)))


def stage_split(args):
    out = {}
    for cfg in args.cfgs:
        Xtr, ytr, segtr, _ = _load_mixed(args.base, cfg, "train")
        dtr = np.load(Path(args.base) / f"dataset_{cfg}_mixed_train.npz", allow_pickle=True)
        dte = np.load(Path(args.base) / f"dataset_{cfg}_mixed_test.npz", allow_pickle=True)
        tr_mask, va_mask, val_segs = split_by_segments(segtr, VAL_FRAC, SEED)
        rec = {"n_windows": {"train": int(tr_mask.sum()), "val": int(va_mask.sum()),
                             "test": int(len(dte["y"]))},
               "n_segments": {"train": int(len(np.unique(segtr[tr_mask]))),
                              "val": int(len(val_segs)),
                              "test": int(len(np.unique(dte["seg_id"])))}}
        if "t_start" in dtr.files and "t_start" in dte.files:
            ts_tr, ts_te = dtr["t_start"], dte["t_start"]
            d = {"train": seg_duration(ts_tr[tr_mask], segtr[tr_mask]),
                 "val": seg_duration(ts_tr[va_mask], segtr[va_mask]),
                 "test": seg_duration(ts_te, dte["seg_id"])}
            tot = sum(d.values())
            rec["duration_s"] = d
            rec["duration_pct"] = {k: 100 * v / tot for k, v in d.items()}
            # origen de cada conjunto (A = ids 1000+, B = ids 2000+)
            rec["train_from_A_s"] = seg_duration(ts_tr[tr_mask & (segtr < 2000)],
                                                 segtr[tr_mask & (segtr < 2000)])
            rec["val_from_A_s"] = seg_duration(ts_tr[va_mask & (segtr < 2000)],
                                               segtr[va_mask & (segtr < 2000)]) \
                if np.any(va_mask & (segtr < 2000)) else 0.0
        else:
            rec["duration_s"] = "falta t_start: ejecutar build_dataset_mixed.py v2"
        out[cfg] = rec
        print(f"[{cfg}] {json.dumps(rec, indent=1)}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "split_info.json").write_text(json.dumps(out, indent=2))


# ----------------------------------------------------------------------------
# Etapa: busqueda de hiperparametros
# ----------------------------------------------------------------------------
def make_hp(arch, a_val, b_val, lr):
    a, _, b, _ = STRUCT[arch]
    hp = dict(FIXED); hp[a] = a_val; hp[b] = b_val; hp["lr"] = lr
    return hp


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def summarize(rows, arch, cfg):
    """{key: (mean, std, n_params, hp)} sobre las filas de (arch, cfg)."""
    by = {}
    for r in rows:
        if r["arch"] == arch and r["cfg"] == cfg:
            by.setdefault(r["key"], []).append(r)
    out = {}
    for k, rs in by.items():
        v = np.array([float(r["val_mae_cm"]) for r in rs])
        out[k] = (v.mean(), v.std(), int(rs[0]["n_params"]), json.loads(rs[0]["hp"]), len(v))
    return out


def stage_search(args, device):
    Path(args.out).mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.out) / "hpsearch.csv"
    cols = ["arch", "cfg", "key", "phase", "hp", "seed", "val_mae_cm", "best_epoch",
            "epochs_run", "secs", "n_params"]
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(cols)
    done = {(r["arch"], r["cfg"], r["key"], int(r["seed"])) for r in read_csv(csv_path)}

    n_total = len(args.cfgs) * len(ARCHS) * (9 + len(LR_SWEEP)) * args.search_seeds
    n_done_now, t_start = 0, time.time()

    def run(arch, cfg, hp, phase):
        nonlocal n_done_now
        key = hp_key(arch, hp)
        for seed in range(args.search_seeds):
            if (arch, cfg, key, seed) in done:
                continue
            t0 = time.time()
            _, info = train_run(arch, cfg, hp, seed, args, device)
            dt = time.time() - t0
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([arch, cfg, key, phase, json.dumps(hp), seed,
                                        f"{info['best_val_mae']*100:.4f}", info["best_epoch"],
                                        info["epochs_run"], f"{dt:.1f}", info["n_params"]])
            done.add((arch, cfg, key, seed))
            n_done_now += 1
            el = time.time() - t_start
            left = n_total - len(done)
            print(f"  [{arch:11s}|{cfg:8s}] {key:28s} seed {seed}: val_MAE="
                  f"{info['best_val_mae']*100:.3f} cm  ep {info['best_epoch']}/{info['epochs_run']}"
                  f"  {dt:.0f}s  | quedan ~{left} ejecuciones, ETA ~{el/n_done_now*left/60:.0f} min")

    for cfg in args.cfgs:
        for arch in ARCHS:
            a, va, b, vb = STRUCT[arch]
            print(f"\n=== Busqueda {arch} | {cfg} | fase A (estructura) ===")
            for x, y in itertools.product(va, vb):
                run(arch, cfg, make_hp(arch, x, y, FIXED["lr"]), "A")
            S = summarize(read_csv(csv_path), arch, cfg)
            best_struct = min((k for k in S if S[k][3]["lr"] == FIXED["lr"]), key=lambda k: S[k][0])
            hpb = S[best_struct][3]
            print(f"=== Busqueda {arch} | {cfg} | fase B (lr) sobre {best_struct} ===")
            for lr in LR_SWEEP:
                run(arch, cfg, make_hp(arch, hpb[a], hpb[b], lr), "B")

    select(args)


def select(args):
    rows = read_csv(Path(args.out) / "hpsearch.csv")
    sel = {}
    for cfg in args.cfgs:
        for arch in ARCHS:
            S = summarize(rows, arch, cfg)
            if not S:
                continue
            kbest = min(S, key=lambda k: S[k][0])
            thr = S[kbest][0] + S[kbest][1]
            cands = [k for k in S if S[k][0] <= thr]
            kchosen = min(cands, key=lambda k: (S[k][2], S[k][0]))
            sel[f"{arch}_{cfg}"] = {"key": kchosen, "hp": S[kchosen][3],
                                    "val_mae_cm": S[kchosen][0], "val_std_cm": S[kchosen][1],
                                    "n_params": S[kchosen][2],
                                    "best_key": kbest, "best_val_mae_cm": S[kbest][0],
                                    "threshold_cm": thr}
            print(f"\n{arch} | {cfg}:  (media +/- desv de val_MAE sobre semillas)")
            for k in sorted(S, key=lambda k: S[k][0]):
                mark = "<- ELEGIDA" if k == kchosen else ("(mejor media)" if k == kbest else "")
                print(f"   {k:28s} {S[k][0]:.3f} +/- {S[k][1]:.3f} cm  params={S[k][2]:7d}  {mark}")
    (Path(args.out) / "selected_hparams.json").write_text(json.dumps(sel, indent=2))
    return sel


# ----------------------------------------------------------------------------
# Etapa: modelos finales
# ----------------------------------------------------------------------------
def stage_final(args, device, sel=None):
    if sel is None:
        sel = json.loads((Path(args.out) / "selected_hparams.json").read_text())
    fdir = Path(args.out) / "final"; fdir.mkdir(parents=True, exist_ok=True)
    summary = []
    for cfg in args.cfgs:
        test_loader, tinfo = load_test_mixed(args.base, cfg)
        dte = np.load(Path(args.base) / f"dataset_{cfg}_mixed_test.npz", allow_pickle=True)
        for arch in ARCHS:
            hp = sel[f"{arch}_{cfg}"]["hp"]
            recs = []
            for seed in range(args.final_seeds):
                jpath = fdir / f"{arch}_{cfg}_seed{seed}.json"
                if jpath.exists():
                    recs.append(json.loads(jpath.read_text())); continue
                model, info = train_run(arch, cfg, hp, seed, args, device)
                P, T = evaluate(model, test_loader, device)
                mae, rmse, mae_g, rmse_g = metrics(P, T)
                bias = (P - T).mean(axis=0)
                inf_b, inf_s = time_inference(model, dte["X"], device)
                rec = {"arch": arch, "cfg": cfg, "seed": seed, "hp": hp, **info,
                       "infer_ms_per_window_batched": inf_b, "infer_ms_per_window_single": inf_s,
                       "test": {"mae": mae.tolist(), "rmse": rmse.tolist(), "bias": bias.tolist(),
                                "mae_g": float(mae_g), "rmse_g": float(rmse_g),
                                "mean_true": T.mean(axis=0).tolist(),
                                "mean_abs_true": np.abs(T).mean(axis=0).tolist(),
                                "n_test": int(len(T))}}
                jpath.write_text(json.dumps(rec, indent=1))
                save = {"pred": P, "true": T, "seg_id": dte["seg_id"]}
                if "t_start" in dte.files:
                    save["t_start"] = dte["t_start"]
                np.savez_compressed(fdir / f"{arch}_{cfg}_seed{seed}_test.npz", **save)
                torch.save(model.state_dict(), fdir / f"{arch}_{cfg}_seed{seed}.pt")
                print(f"  [{arch:11s}|{cfg:8s}] seed {seed}: test MAE fwd/lat/up = "
                      f"{mae[0]*100:.2f}/{mae[1]*100:.2f}/{mae[2]*100:.2f} cm  "
                      f"sesgo fwd {bias[0]*100:+.2f} cm  ep {info['best_epoch']}/{info['epochs_run']}  "
                      f"{info['train_time_s']:.0f}s  inf {inf_b:.3f}/{inf_s:.2f} ms")
                recs.append(rec)
            def ms(f):
                v = np.array([f(r) for r in recs]); return v.mean(), v.std()
            row = {"arch": arch, "cfg": cfg, "hp": hp_key(arch, hp), "n_params": recs[0]["n_params"],
                   "seeds": len(recs)}
            for i, c in enumerate(["fwd", "lat", "up"]):
                for met in ["mae", "rmse", "bias"]:
                    m, s = ms(lambda r: r["test"][met][i] * 100)
                    row[f"{met}_{c}_cm"] = f"{m:.3f}"; row[f"{met}_{c}_sd"] = f"{s:.3f}"
            for k, f in [("mae_g_cm", lambda r: r["test"]["mae_g"] * 100),
                         ("rmse_g_cm", lambda r: r["test"]["rmse_g"] * 100),
                         ("best_epoch", lambda r: r["best_epoch"]),
                         ("epochs_run", lambda r: r["epochs_run"]),
                         ("train_time_s", lambda r: r["train_time_s"]),
                         ("time_per_epoch_s", lambda r: r["time_per_epoch_s"]),
                         ("infer_ms_batched", lambda r: r["infer_ms_per_window_batched"]),
                         ("infer_ms_single", lambda r: r["infer_ms_per_window_single"])]:
                m, s = ms(f); row[k] = f"{m:.4f}"; row[k + "_sd"] = f"{s:.4f}"
            summary.append(row)
    with open(Path(args.out) / "summary_final.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    print("\n=== RESUMEN (conjunto de PRUEBA, media sobre semillas) ===")
    for r in summary:
        print(f"  {r['arch']:11s} {r['cfg']:8s}  MAE fwd {r['mae_fwd_cm']}+/-{r['mae_fwd_sd']}  "
              f"lat {r['mae_lat_cm']}  up {r['mae_up_cm']}  global {r['mae_g_cm']} cm  "
              f"params {r['n_params']}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "split", "search", "select", "final", "all"],
                    required=True)
    ap.add_argument("--base", default=r"C:\Users\Usuario\Desktop\TFM\extracted")
    ap.add_argument("--out", default=r"C:\Users\Usuario\Desktop\TFM\results")
    ap.add_argument("--cfgs", nargs="+", default=["imu", "imuwheel"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--search_seeds", type=int, default=2)
    ap.add_argument("--final_seeds", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"Dispositivo: {device} ({gpu}) | torch {torch.__version__}")

    if args.stage == "smoke":
        args.out = str(Path(args.out) / "_smoke"); args.epochs = 1
        args.search_seeds = 1; args.final_seeds = 1
        stage_split(args)
        sel = {}
        for cfg in args.cfgs:
            for arch in ARCHS:
                a, va, b, vb = STRUCT[arch]
                hp = make_hp(arch, va[1], vb[1], FIXED["lr"])
                t0 = time.time()
                _, info = train_run(arch, cfg, hp, 0, args, device)
                print(f"  smoke {arch:11s} {cfg:8s}: 1 epoca en {time.time()-t0:.1f}s, "
                      f"params {info['n_params']}")
                sel[f"{arch}_{cfg}"] = {"hp": hp}
        stage_final(args, device, sel)
        print("\nSMOKE OK. Multiplica el tiempo por epoca x ~25 epocas para estimar cada ejecucion.")
        return
    if args.stage in ("split", "all"):
        stage_split(args)
    if args.stage in ("search", "all"):
        stage_search(args, device)
    if args.stage == "select":
        select(args)
    if args.stage in ("final", "all"):
        stage_final(args, device)


if __name__ == "__main__":
    main()
