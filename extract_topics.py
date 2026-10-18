"""
extract_topics.py — Extrae topics propioceptivos de un run a CSV en UNA sola
pasada por el bag, con barra de progreso.
Uso:  python extract_topics.py Run_A
      python extract_topics.py Run_B
Extrae: /imu, /wheel_odometry -> CSV. Ademas normaliza el ground truth
DGPS (.txt -> ground_truth.csv). NO extrae odometria visual (va aparte).
Timestamp = header stamp, en NANOSEGUNDOS enteros (sin perdida).
"""
import sys
import csv
import time
from pathlib import Path
from rosbags.highlevel import AnyReader

if len(sys.argv) < 2:
    print("Uso: python extract_topics.py <NOMBRE_DEL_RUN>")
    sys.exit(1)

RUN = sys.argv[1]
ROOT = Path(r"C:\Users\Usuario\Desktop\TFM")
RUN_DIR = ROOT / RUN
OUT_DIR = ROOT / "extracted" / RUN
OUT_DIR.mkdir(parents=True, exist_ok=True)

bags = list(RUN_DIR.glob("*.bag"))
if not bags:
    print(f"No se encontró .bag en {RUN_DIR}")
    sys.exit(1)
BAG = bags[0]

# Topics a extraer y sus ficheros de salida
WANTED = {"/imu", "/wheel_odometry"}

def header_stamp_ns(msg):
    """Timestamp del header en nanosegundos enteros (sin float)."""
    st = msg.header.stamp
    return int(st.sec) * 1_000_000_000 + int(st.nanosec)

# ---------- 1) Normalizar ground truth (.txt -> .csv) ----------
gt_txt = RUN_DIR / "HiDrive-dgps" / "processed" / "dgps_aligned_nav.txt"
gt_csv = OUT_DIR / "ground_truth.csv"
print(f"=== RUN: {RUN} ===")
print(f"[1/2] Normalizando ground truth -> {gt_csv.name}")
n_gt = 0
with open(gt_txt) as fin, open(gt_csv, "w", newline="") as fout:
    w = csv.writer(fout)
    w.writerow(["timestamp_s", "x", "y", "z"])  # GT viene en segundos (float)
    for line in fin:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        w.writerow([p[0], p[1], p[2], p[3]])  # t, x, y, z (descartamos cols 5-8)
        n_gt += 1
print(f"      {n_gt} filas de ground truth escritas.\n")

# ---------- 2) Extraer IMU y wheel_odometry en UNA pasada ----------
imu_csv = OUT_DIR / "imu.csv"
wo_csv = OUT_DIR / "wheel_odometry.csv"

print(f"[2/2] Extrayendo {WANTED} en una sola pasada por el bag")
print(f"      Bag: {BAG.name}  ({BAG.stat().st_size/1e9:.1f} GB)")

with AnyReader([BAG]) as reader:
    total = getattr(reader, "message_count", None)

    # Connections de los topics deseados
    conns = [c for c in reader.connections if c.topic in WANTED]
    found = {c.topic for c in conns}
    missing = WANTED - found
    if missing:
        print(f"      AVISO: no encontrados en el bag: {missing}")

    fout_imu = open(imu_csv, "w", newline="")
    fout_wo = open(wo_csv, "w", newline="")
    w_imu = csv.writer(fout_imu)
    w_wo = csv.writer(fout_wo)
    w_imu.writerow(["timestamp_ns", "gyro_x", "gyro_y", "gyro_z",
                    "acc_x", "acc_y", "acc_z"])
    w_wo.writerow(["timestamp_ns", "lin_x", "lin_y", "lin_z",
                   "ang_x", "ang_y", "ang_z"])

    n_imu = 0
    n_wo = 0
    seen = 0
    t_start = time.time()
    last_print = t_start

    # Recorremos SOLO los topics deseados (rosbags filtra por connections)
    for connection, timestamp, rawdata in reader.messages(connections=conns):
        seen += 1
        topic = connection.topic
        msg = reader.deserialize(rawdata, connection.msgtype)

        if topic == "/imu":
            ts = header_stamp_ns(msg)
            g = msg.angular_velocity
            a = msg.linear_acceleration
            w_imu.writerow([ts, g.x, g.y, g.z, a.x, a.y, a.z])
            n_imu += 1
        elif topic == "/wheel_odometry":
            ts = header_stamp_ns(msg)
            tw = msg.twist.twist  # TwistWithCovariance -> Twist
            lin = tw.linear
            ang = tw.angular
            w_wo.writerow([ts, lin.x, lin.y, lin.z, ang.x, ang.y, ang.z])
            n_wo += 1

        # --- Progreso cada 3 segundos ---
        now = time.time()
        if now - last_print >= 3.0:
            elapsed = now - t_start
            rate = seen / elapsed if elapsed > 0 else 0
            print(f"      procesados: imu={n_imu:>7}  wheel={n_wo:>7}  "
                  f"| {seen} msgs en {elapsed:5.0f}s ({rate:,.0f} msg/s)")
            last_print = now

    fout_imu.close()
    fout_wo.close()
    elapsed = time.time() - t_start

print(f"\n      IMU escritos:            {n_imu}")
print(f"      Wheel odometry escritos: {n_wo}")
print(f"      Tiempo total extracción: {elapsed:.0f} s")
print(f"\nCSVs en: {OUT_DIR}")
print("Hecho.")