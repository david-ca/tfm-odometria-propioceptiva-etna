"""
robustness_arch.py -- Compara las TRES arquitecturas (GRU, TCN, Transformer) en la
configuracion IMU-only, con varias semillas, para confirmar con rigor si empatan.
Mismo protocolo para todas (split fijo, mismas semillas, mismas metricas). Resume,
sobre el test (Run_A), el MAE global y el MAE/sesgo del avance (media +/- desv).

Uso:  python robustness_arch.py --seeds 5 --epochs 60 --cfg imu
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn

from dataset import make_loaders, load_test
from train import build_model, metrics, evaluate


def train_one(model_name, cfg, base, seed, epochs, patience, batch, lr, device):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_loader, va_loader, info = make_loaders(base, cfg, batch)
    model = build_model(model_name, info["n_channels"]).to(device)
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
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cfg", choices=["imu", "imuwheel"], default="imu")
    ap.add_argument("--base", default=r"C:\Users\Usuario\Desktop\TFM\extracted")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device} | cfg={args.cfg} | {args.seeds} semillas x 3 arquitecturas\n")

    test_loader, tinfo = load_test(args.base, args.cfg)
    results = {}
    for model_name in ["gru", "tcn", "transformer"]:
        rows = []
        for seed in range(args.seeds):
            t0 = time.time()
            model = train_one(model_name, args.cfg, args.base, seed, args.epochs,
                              args.patience, args.batch, args.lr, device)
            P, T = evaluate(model, test_loader, device)
            mae, rmse, mae_g, rmse_g = metrics(P, T)
            rows.append({"mae_g": mae_g, "fwd_mae": mae[0],
                         "fwd_bias": (P[:, 0] - T[:, 0]).mean(),
                         "lat_mae": mae[1], "up_mae": mae[2]})
            print(f"  [{model_name:11s}] seed {seed}: MAE_glob={mae_g*100:.2f}cm  "
                  f"fwd_MAE={mae[0]*100:.2f}cm  ({time.time()-t0:.0f}s)")
        results[model_name] = rows
        print()

    print("=== RESUMEN (media +/- desv entre semillas, test = Run_A, cfg={}) ===".format(args.cfg))
    for model_name in ["gru", "tcn", "transformer"]:
        R = results[model_name]
        def ms(k):
            v = np.array([r[k] for r in R]); return v.mean(), v.std()
        mg, sg = ms("mae_g"); fm, fs = ms("fwd_mae"); fb, fbs = ms("fwd_bias")
        lm, _ = ms("lat_mae"); um, _ = ms("up_mae")
        print(f"\n{model_name}:")
        print(f"  MAE global: {mg*100:.2f} +/- {sg*100:.2f} cm")
        print(f"  d_fwd MAE:  {fm*100:.2f} +/- {fs*100:.2f} cm   sesgo {fb*100:+.2f} cm")
        print(f"  d_lat MAE:  {lm*100:.2f} cm    d_up MAE: {um*100:.2f} cm")

    print("\n=== VEREDICTO ===")
    gru = np.array([r["mae_g"] for r in results["gru"]])
    tcn = np.array([r["mae_g"] for r in results["tcn"]])
    trf = np.array([r["mae_g"] for r in results["transformer"]])
    spread = max(gru.mean(), tcn.mean(), trf.mean()) - min(gru.mean(), tcn.mean(), trf.mean())
    within = max(gru.std(), tcn.std(), trf.std())
    print(f"MAE global medio:  GRU={gru.mean()*100:.2f}  TCN={tcn.mean()*100:.2f}  Transformer={trf.mean()*100:.2f} cm")
    print(f"Diferencia entre arquitecturas: {spread*100:.2f} cm  |  desv. dentro de una arquitectura: {within*100:.2f} cm")
    if spread <= within:
        print("La diferencia entre arquitecturas es menor que la variacion por semilla:")
        print("-> las tres EMPATAN; la arquitectura no es un factor relevante en este problema.")
    else:
        print("Hay diferencias entre arquitecturas mayores que el ruido de semilla; revisar cual destaca.")


if __name__ == "__main__":
    main()