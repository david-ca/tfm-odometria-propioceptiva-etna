"""
robustness.py -- Verifica si el hallazgo (IMU-only generaliza el avance mejor que
IMU+wheel en Run_A) es robusto o casualidad de una semilla. Entrena N semillas de
cada configuracion y resume, sobre el test (Run_A), el sesgo y MAE de d_fwd, mas
el MAE global.

El split de validacion es FIJO (no depende de la semilla del modelo), de modo que
todas las corridas comparan sobre el mismo conjunto; solo cambia la inicializacion.

Uso:  python robustness.py --seeds 5 --epochs 60
"""
import argparse, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from dataset import make_loaders, load_test
from train import GRURegressor, metrics, evaluate


def train_one(cfg, base, seed, epochs, patience, batch, lr, hidden, layers, device):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_loader, va_loader, info = make_loaders(base, cfg, batch)
    model = GRURegressor(info["n_channels"], hidden, layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf"); best_state = None; bad = 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        Pv, Tv = evaluate(model, va_loader, device)
        _, _, mae_g, _ = metrics(Pv, Tv)
        if mae_g < best_val:
            best_val = mae_g; best_state = {k: v.clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--base", default=r"C:\Users\Usuario\Desktop\TFM\extracted")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device} | {args.seeds} semillas por configuracion\n")

    results = {}
    for cfg in ["imu", "imuwheel"]:
        test_loader, tinfo = load_test(args.base, cfg)
        rows = []
        for seed in range(args.seeds):
            t0 = time.time()
            model, info = train_one(cfg, args.base, seed, args.epochs, args.patience,
                                    args.batch, args.lr, args.hidden, args.layers, device)
            P, T = evaluate(model, test_loader, device)
            mae, rmse, mae_g, rmse_g = metrics(P, T)
            fwd_bias = (P[:, 0] - T[:, 0]).mean()
            fwd_pred = P[:, 0].mean()
            rows.append({"seed": seed, "mae_g": mae_g, "fwd_mae": mae[0],
                         "fwd_bias": fwd_bias, "fwd_pred": fwd_pred,
                         "lat_mae": mae[1], "up_mae": mae[2]})
            print(f"  [{cfg:9s}] seed {seed}: MAE_glob={mae_g*100:.2f}cm  "
                  f"fwd_pred={fwd_pred*100:.1f}cm (real {T[:,0].mean()*100:.1f})  "
                  f"fwd_MAE={mae[0]*100:.2f}cm  ({time.time()-t0:.0f}s)")
        results[cfg] = rows
        print()

    # --- resumen comparativo ---
    print("=== RESUMEN (media +/- desv. entre semillas, test = Run_A) ===")
    real_fwd = None
    for cfg in ["imu", "imuwheel"]:
        R = results[cfg]
        def ms(key):
            v = np.array([r[key] for r in R])
            return v.mean(), v.std()
        mg, sg = ms("mae_g"); fm, fs = ms("fwd_mae")
        fp, fps = ms("fwd_pred"); fb, fbs = ms("fwd_bias")
        lm, _ = ms("lat_mae"); um, _ = ms("up_mae")
        print(f"\n{cfg}:")
        print(f"  MAE global:   {mg*100:.2f} +/- {sg*100:.2f} cm")
        print(f"  d_fwd MAE:    {fm*100:.2f} +/- {fs*100:.2f} cm")
        print(f"  d_fwd pred:   {fp*100:.1f} +/- {fps*100:.1f} cm")
        print(f"  d_fwd sesgo:  {fb*100:+.2f} +/- {fbs*100:.2f} cm")
        print(f"  d_lat MAE:    {lm*100:.2f} cm   d_up MAE: {um*100:.2f} cm")

    im = np.array([r["fwd_mae"] for r in results["imu"]])
    iw = np.array([r["fwd_mae"] for r in results["imuwheel"]])
    print(f"\n=== VEREDICTO ===")
    print(f"d_fwd MAE:  IMU-only={im.mean()*100:.2f}cm  IMU+wheel={iw.mean()*100:.2f}cm")
    if im.mean() < iw.mean() and im.max() < iw.min():
        print("IMU-only bate a IMU+wheel en TODAS las semillas: hallazgo ROBUSTO.")
    elif im.mean() < iw.mean():
        print("IMU-only mejor en media pero con solape entre semillas: tendencia, no concluyente.")
    else:
        print("No hay ventaja consistente de IMU-only: el hallazgo previo era ruido de una semilla.")


if __name__ == "__main__":
    main()