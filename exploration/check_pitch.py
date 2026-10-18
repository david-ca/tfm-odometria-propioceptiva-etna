import numpy as np
d = np.load(r"extracted\Run_A\dataset_imuwheel_Run_A.npz", allow_pickle=True)
X, ch = d["X"], [str(c) for c in d["channels"]]
if X.shape[-1] != len(ch): X = np.swapaxes(X, 1, 2)  # canales al final
F = X.reshape(-1, len(ch))[::10]                       # todas las muestras, submuestreadas
np.set_printoptions(precision=2, suppress=True, linewidth=160)
print("Canales:", ch)
print(np.corrcoef(F.T))