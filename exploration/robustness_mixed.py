"""
robustness_mixed.py -- Verifica la robustez de la particion MIXTA (propuesta del
tutor) sobre varias semillas, para ambas configuraciones (IMU e IMU+wheel), y la
compara con la particion Hold-Out original.

Para cada config y semilla entrena sobre el train mixto (Run A completo + porcion
de Run B) y evalua sobre el test mixto (porcion de Run B). Reporta MAE global y
MAE/sesgo del avance (media +/- desv entre semillas).

Referencia Hold-Out (de corridas previas, 5 semillas):
  IMU-only:   d_fwd MAE 9.10+/-0.12 cm | IMU+wheel: d_fwd MAE 17.38+/-1.70 cm

Uso:  python robustness_mixed.py --seeds 5 --epochs 60
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn

from dataset import make_loaders_mixed, load_test_mixed
from train import build_model, metrics, evaluate

HOLDOUT_REF = {   # d_fwd MAE de la particion original (5 semillas), para comparar
    "imu": (9.10, 0.12),
    "imuwheel": (17.38, 1.70),
}


def train_one(model_name, cfg, base, seed, epochs, patience, batch, lr, device):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_loader, va_loader, info = make_loaders_mixed(base, cfg, batch)
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
    ap.add_argument("--base", default=r"C:\Users\Usuario\Desktop\TFM\extracted")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device} | particion MIXTA | {args.seeds} semillas x 2 configs\n")

    results = {}
    for cfg in ["imu", "imuwheel"]:
        test_loader, tinfo = load_test_mixed(args.base, cfg)
        rows = []
        for seed in range(args.seeds):
            t0 = time.time()
            model = train_one("gru", cfg, args.base, seed, args.epochs, args.patience,
                              args.batch, args.lr, device)
            P, T = evaluate(model, test_loader, device)
            mae, rmse, mae_g, rmse_g = metrics(P, T)
            rows.append({"mae_g": mae_g, "fwd_mae": mae[0],
                         "fwd_bias": (P[:, 0] - T[:, 0]).mean(),
                         "lat_mae": mae[1], "up_mae": mae[2]})
            print(f"  [{cfg:9s}] seed {seed}: MAE_glob={mae_g*100:.2f}cm  "
                  f"fwd_MAE={mae[0]*100:.2f}cm  sesgo={ (P[:,0]-T[:,0]).mean()*100:+.2f}cm  ({time.time()-t0:.0f}s)")
        results[cfg] = rows
        print()

    print("=== RESUMEN particion MIXTA (media +/- desv, test = porcion de Run B) ===")
    for cfg in ["imu", "imuwheel"]:
        R = results[cfg]
        def ms(k):
            v = np.array([r[k] for r in R]); return v.mean(), v.std()
        mg, sg = ms("mae_g"); fm, fs = ms("fwd_mae"); fb, fbs = ms("fwd_bias")
        lm, _ = ms("lat_mae"); um, _ = ms("up_mae")
        ho = HOLDOUT_REF[cfg]
        print(f"\n{cfg}:")
        print(f"  MAE global:  {mg*100:.2f} +/- {sg*100:.2f} cm")
        print(f"  d_fwd MAE:   {fm*100:.2f} +/- {fs*100:.2f} cm   (Hold-Out era {ho[0]:.2f}+/-{ho[1]:.2f} cm)")
        print(f"  d_fwd sesgo: {fb*100:+.2f} cm")
        print(f"  d_lat MAE:   {lm*100:.2f} cm    d_up MAE: {um*100:.2f} cm")

    print("\n=== INTERPRETACION ===")
    im = np.array([r["fwd_mae"] for r in results["imu"]]).mean()*100
    iw = np.array([r["fwd_mae"] for r in results["imuwheel"]]).mean()*100
    print(f"d_fwd MAE mixta:  IMU={im:.2f}cm  IMU+wheel={iw:.2f}cm")
    print(f"Recordatorio Hold-Out: IMU=9.10cm  IMU+wheel=17.38cm")
    print("- Si en mixta el avance mejora MUCHO en ambas (p.ej. <3 cm): la cobertura")
    print("  de velocidades era el problema dominante (confirma el diagnostico).")
    if iw <= im * 1.3:
        print("- En mixta el wheel ya NO perjudica claramente: con ambos regimenes de")
        print("  velocidad cubiertos, el anclaje al regimen de B desaparece.")
    else:
        print("- En mixta el wheel AUN perjudica: apunta a que el error del modelo")
        print("  cinematico/slip pesa aparte de la cobertura de velocidad.")


if __name__ == "__main__":
    main()