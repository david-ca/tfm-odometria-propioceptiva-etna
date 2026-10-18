"""
hyperparam_search.py -- Busqueda de hiperparametros en DOS FASES sobre la GRU, con
la particion mixta. Guiada por el error de VALIDACION (nunca el test). 3 semillas
por combinacion. Guarda resultados INCREMENTALMENTE en un CSV (si se interrumpe,
no se pierde lo hecho; al relanzar, salta las combinaciones ya evaluadas).

Fase 1: barre learning_rate x hidden_size (lo mas influyente).
Fase 2: alrededor de la mejor combinacion de la fase 1, barre layers x dropout.

Uso:
  python hyperparam_search.py --phase 1 --epochs 60
  python hyperparam_search.py --phase 2 --epochs 60 --best_lr 1e-3 --best_hidden 128

Al terminar la fase 1, mira el CSV, identifica la mejor (menor val_MAE medio) y
pasala a la fase 2 con --best_lr y --best_hidden.
"""
import argparse, csv, time, itertools
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from dataset import make_loaders_mixed
from train import metrics, evaluate

# --- modelo GRU con dropout parametrizable (autonomo, no depende de train.GRURegressor) ---
class GRUNet(nn.Module):
    def __init__(self, n_ch, hidden, layers, dropout):
        super().__init__()
        self.gru = nn.GRU(n_ch, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


def train_eval(cfg, base, lr, hidden, layers, dropout, seed, epochs, patience, batch, device):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, info = make_loaders_mixed(base, cfg, batch)
    model = GRUNet(info["n_channels"], hidden, layers, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best = float("inf"); bad = 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
        P, T = evaluate(model, va, device)
        _, _, mae_g, _ = metrics(P, T)
        if mae_g < best:
            best = mae_g; bad = 0
        else:
            bad += 1
            if bad >= patience: break
    return best


def load_done(csv_path):
    done = set()
    if Path(csv_path).exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add((float(row["lr"]), int(row["hidden"]), int(row["layers"]),
                          float(row["dropout"]), int(row["seed"])))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=[1, 2], required=True)
    ap.add_argument("--cfg", default="imuwheel")
    ap.add_argument("--base", default=r"C:\Users\Usuario\Desktop\TFM\extracted")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--best_lr", type=float, default=1e-3)
    ap.add_argument("--best_hidden", type=int, default=128)
    ap.add_argument("--out", default=r"C:\Users\Usuario\Desktop\TFM\hpsearch")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.out) / f"phase{args.phase}_{args.cfg}.csv"

    if args.phase == 1:
        lrs = [3e-4, 1e-3, 3e-3]
        hiddens = [64, 128, 256]
        combos = [(lr, h, 2, 0.2) for lr, h in itertools.product(lrs, hiddens)]
    else:
        layerss = [1, 2, 3]
        dropouts = [0.0, 0.2, 0.4]
        combos = [(args.best_lr, args.best_hidden, l, d)
                  for l, d in itertools.product(layerss, dropouts)]

    done = load_done(csv_path)
    new_file = not Path(csv_path).exists()
    f = open(csv_path, "a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow(["lr", "hidden", "layers", "dropout", "seed", "val_mae_cm", "secs"])
        f.flush()

    print(f"=== Fase {args.phase} | {args.cfg} | {device} | {len(combos)} combos x {args.seeds} semillas ===\n")
    for (lr, hidden, layers, dropout) in combos:
        vals = []
        for seed in range(args.seeds):
            key = (float(lr), int(hidden), int(layers), float(dropout), int(seed))
            if key in done:
                continue
            t0 = time.time()
            v = train_eval(args.cfg, args.base, lr, hidden, layers, dropout,
                           seed, args.epochs, args.patience, args.batch, device)
            dt = time.time() - t0
            w.writerow([lr, hidden, layers, dropout, seed, f"{v*100:.3f}", f"{dt:.0f}"])
            f.flush()   # guardado incremental
            vals.append(v*100)
            print(f"  lr={lr:.0e} h={hidden} L={layers} drop={dropout} seed={seed}: "
                  f"val_MAE={v*100:.2f}cm ({dt:.0f}s)")
        if vals:
            print(f"  -> media val_MAE={np.mean(vals):.2f} +/- {np.std(vals):.2f} cm\n")
    f.close()
    print(f"Resultados en {csv_path}")
    print("Analiza el CSV: la combinacion con menor val_mae_cm medio es la mejor.")


if __name__ == "__main__":
    main()