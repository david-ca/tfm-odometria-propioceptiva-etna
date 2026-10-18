"""
train.py -- Modelo GRU base y entrenamiento para el Bloque 6 (pipeline completo).

Modelo: GRU (2 capas, hidden 128) -> capa lineal -> 3 salidas (d_fwd, d_lat, d_up).
Perdida: MSE. Metricas reportadas: MAE y RMSE por componente y global.
Validacion: tramos reservados de Run_B. Early stopping sobre MAE de validacion.

Uso:
  python train.py --cfg imuwheel --epochs 60
  python train.py --cfg imu --epochs 60
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from dataset import make_loaders, load_test, make_loaders_mixed, load_test_mixed


class GRURegressor(nn.Module):
    def __init__(self, n_channels, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=n_channels, hidden_size=hidden,
                          num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 3)
        )

    def forward(self, x):                 # x: (B, 100, C)
        out, h = self.gru(x)              # h: (layers, B, hidden)
        return self.head(h[-1])           # usa el estado final de la ultima capa


class TemporalBlock(nn.Module):
    """Bloque residual del TCN: dos convoluciones causales dilatadas + residual."""
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation      # padding causal (solo mira al pasado)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.pad = pad
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):                  # x: (B, C, T)
        y = self.conv1(x)[:, :, :-self.pad]      # recorta el padding futuro (causal)
        y = self.drop(self.relu(y))
        y = self.conv2(y)[:, :, :-self.pad]
        y = self.drop(self.relu(y))
        res = x if self.down is None else self.down(x)
        return self.relu(y + res)


class TCNRegressor(nn.Module):
    def __init__(self, n_channels, channels=64, levels=4, kernel=3, dropout=0.2):
        super().__init__()
        blocks = []
        in_ch = n_channels
        for i in range(levels):
            blocks.append(TemporalBlock(in_ch, channels, kernel, dilation=2 ** i, dropout=dropout))
            in_ch = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(channels, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):                  # x: (B, T, C)
        y = self.tcn(x.transpose(1, 2))    # -> (B, channels, T)
        return self.head(y[:, :, -1])      # usa el ultimo paso temporal


def build_model(name, n_channels, hidden=128, layers=2):
    if name == "gru":
        return GRURegressor(n_channels, hidden, layers)
    if name == "tcn":
        return TCNRegressor(n_channels)
    if name == "transformer":
        return TransformerRegressor(n_channels)
    raise ValueError(f"modelo desconocido: {name}")


class TransformerRegressor(nn.Module):
    """Transformer encoder pequeno para secuencias, con codificacion posicional
    sinusoidal y pooling medio sobre la secuencia. Tamano contenido para no
    sobreajustar con ~12k ventanas."""
    def __init__(self, n_channels, d_model=64, nhead=4, layers=2, dim_ff=128, dropout=0.2, max_len=110):
        super().__init__()
        self.proj = nn.Linear(n_channels, d_model)
        # codificacion posicional sinusoidal (fija, no entrenable)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))    # (1, max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):                  # x: (B, T, C)
        y = self.proj(x) + self.pe[:, :x.size(1)]
        y = self.encoder(y)                # (B, T, d_model)
        return self.head(y.mean(dim=1))    # pooling medio sobre la secuencia


def metrics(pred, true):
    """MAE y RMSE por componente (en metros) y global."""
    err = pred - true
    mae = np.abs(err).mean(axis=0)
    rmse = np.sqrt((err ** 2).mean(axis=0))
    mae_g = np.abs(err).mean()
    rmse_g = np.sqrt((err ** 2).mean())
    return mae, rmse, mae_g, rmse_g


def evaluate(model, loader, device):
    model.eval()
    P, T = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            P.append(model(xb).cpu().numpy())
            T.append(yb.numpy())
    return np.concatenate(P), np.concatenate(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", choices=["imu", "imuwheel"], default="imuwheel")
    ap.add_argument("--model", choices=["gru", "tcn", "transformer"], default="gru")
    ap.add_argument("--partition", choices=["holdout", "mixed"], default="holdout",
                    help="holdout = B->train/A->test (original); mixed = A+porcionB / porcionB")
    ap.add_argument("--base", default=r"C:\Users\Usuario\Desktop\TFM\extracted")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="semilla del modelo (init/shuffle); el split de val es fijo")
    ap.add_argument("--out", default=r"C:\Users\Usuario\Desktop\TFM\models")
    args = ap.parse_args()

    # semilla del MODELO (init de pesos, dropout, shuffle). El split de validacion
    # NO depende de esta semilla (usa su propia semilla fija en dataset.py), para
    # que todas las corridas comparen sobre el MISMO conjunto de validacion y solo
    # varie la inicializacion del modelo.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device} | semilla modelo: {args.seed}")

    if args.partition == "mixed":
        tr_loader, va_loader, info = make_loaders_mixed(args.base, args.cfg, args.batch)
    else:
        tr_loader, va_loader, info = make_loaders(args.base, args.cfg, args.batch)
    print(f"Config {args.cfg} [{args.partition}]: {info['n_channels']} canales | "
          f"train {info['n_train']} ventanas ({info.get('n_train_segs', '?')} tramos) | "
          f"val {info['n_val']} ventanas ({info['n_val_segs']} tramos)")

    model = build_model(args.model, info["n_channels"], args.hidden, args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    best_path = Path(args.out) / f"{args.model}_{args.cfg}_best.pt"
    hist = []
    best_val = float("inf"); best_epoch = -1; bad = 0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        tl = 0.0; n = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(xb); n += len(xb)
        train_loss = tl / n

        P, T = evaluate(model, va_loader, device)
        mae, rmse, mae_g, rmse_g = metrics(P, T)
        hist.append({"epoch": ep, "train_mse": train_loss,
                     "val_mae": float(mae_g), "val_rmse": float(rmse_g),
                     "val_mae_fwd": float(mae[0]), "val_mae_lat": float(mae[1]),
                     "val_mae_up": float(mae[2])})
        print(f"ep {ep:3d}  train_mse={train_loss:.5f}  "
              f"val_MAE={mae_g*100:.2f}cm (fwd {mae[0]*100:.2f} lat {mae[1]*100:.2f} "
              f"up {mae[2]*100:.2f})  val_RMSE={rmse_g*100:.2f}cm")

        if mae_g < best_val:
            best_val = mae_g; best_epoch = ep; bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping en epoch {ep} (mejor: {best_epoch}, "
                      f"val_MAE {best_val*100:.2f}cm)")
                break

    dt = time.time() - t0
    print(f"\nEntrenamiento: {dt:.1f}s. Mejor val_MAE={best_val*100:.2f}cm (ep {best_epoch})")

    model.load_state_dict(torch.load(best_path))

    def bias_report(tag, P, T):
        names = ["d_fwd", "d_lat", "d_up"]
        print(f"\n--- Sesgo por componente [{tag}] ---")
        for i, nm in enumerate(names):
            err = P[:, i] - T[:, i]
            frac = abs(err.mean()) / np.abs(err).mean() * 100 if np.abs(err).mean() > 0 else 0
            print(f"  {nm}: real={T[:,i].mean()*100:6.2f}cm  pred={P[:,i].mean()*100:6.2f}cm  "
                  f"sesgo={err.mean()*100:+6.2f}cm  MAE={np.abs(err).mean()*100:5.2f}cm  "
                  f"(sesgo={frac:.0f}% del MAE)")

    # --- diagnostico sobre VALIDACION (Run B) ---
    Pv, Tv = evaluate(model, va_loader, device)
    bias_report("VAL Run_B", Pv, Tv)

    # --- evaluacion final sobre Run_A (test) con el mejor modelo ---
    if args.partition == "mixed":
        test_loader, tinfo = load_test_mixed(args.base, args.cfg)
    else:
        test_loader, tinfo = load_test(args.base, args.cfg)
    P, T = evaluate(model, test_loader, device)
    mae, rmse, mae_g, rmse_g = metrics(P, T)
    print(f"\n=== TEST (Run_A, {tinfo['n_test']} ventanas) ===")
    print(f"  MAE  global: {mae_g*100:.2f} cm  (fwd {mae[0]*100:.2f}  "
          f"lat {mae[1]*100:.2f}  up {mae[2]*100:.2f})")
    print(f"  RMSE global: {rmse_g*100:.2f} cm  (fwd {rmse[0]*100:.2f}  "
          f"lat {rmse[1]*100:.2f}  up {rmse[2]*100:.2f})")
    bias_report("TEST Run_A", P, T)

    # guardar historial y predicciones de test (para reconstruccion posterior)
    Path(args.out).joinpath(f"history_{args.model}_{args.cfg}.json").write_text(json.dumps(hist, indent=2))
    np.savez_compressed(Path(args.out) / f"test_pred_{args.model}_{args.cfg}.npz",
                        pred=P, true=T, seg_id=tinfo["seg_id"])
    print(f"\nGuardado: {best_path.name}, history_{args.model}_{args.cfg}.json, "
          f"test_pred_{args.model}_{args.cfg}.npz")


if __name__ == "__main__":
    main()