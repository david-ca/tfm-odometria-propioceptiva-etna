"""
plot_loss.py -- Curvas de entrenamiento (pérdida y MAE de validación por época).
Lee history_<model>_<cfg>.json (guardado por train.py) y genera figura_loss.png.
Uso:  python plot_loss.py --model gru --cfg imuwheel
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gru")
ap.add_argument("--cfg", default="imuwheel")
ap.add_argument("--models", default=r"C:\Users\Usuario\Desktop\TFM\models")
args = ap.parse_args()

hist = json.loads((Path(args.models) / f"history_{args.model}_{args.cfg}.json").read_text())
ep = [h["epoch"] for h in hist]
train_mse = [h["train_mse"] for h in hist]
val_mae = [h["val_mae"]*100 for h in hist]      # a cm
best = int(np.argmin(val_mae))

fig, ax1 = plt.subplots(figsize=(8, 5))
# eje izquierdo: pérdida de entrenamiento (MSE)
c1 = "tab:blue"
ax1.plot(ep, train_mse, color=c1, lw=1.8, label="Pérdida de entrenamiento (MSE)")
ax1.set_xlabel("Época")
ax1.set_ylabel("MSE de entrenamiento", color=c1)
ax1.tick_params(axis="y", labelcolor=c1)
ax1.set_yscale("log")
# eje derecho: MAE de validación (cm)
ax2 = ax1.twinx()
c2 = "tab:red"
ax2.plot(ep, val_mae, color=c2, lw=1.8, label="MAE de validación")
ax2.set_ylabel("MAE de validación (cm)", color=c2)
ax2.tick_params(axis="y", labelcolor=c2)
# marcar la mejor época (parada temprana)
ax2.axvline(ep[best], color="gray", ls="--", lw=1, alpha=0.7)
ax2.annotate(f"mejor época ({ep[best]})", xy=(ep[best], val_mae[best]),
             xytext=(ep[best]+2, val_mae[best]+0.05), fontsize=9, color="gray")

fig.suptitle("Evolución del entrenamiento (GRU, IMU + odometría de ruedas)", fontsize=12)
# leyenda combinada
l1,lab1 = ax1.get_legend_handles_labels(); l2,lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lab1+lab2, loc="upper right", fontsize=9)
ax1.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figura_loss.png", dpi=150, bbox_inches="tight")
print("Guardado: figura_loss.png")