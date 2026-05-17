"""
fog_feature_builder.py
-----------------------
Construye los vectores de features por ventana deslizante a partir de las
señales extraídas por fog_extractor.py.

Nombres de las features generadas que espera cada modelo:
    Modelo C (26 features):
        Acc_Ankle_mean, Acc_Ankle_std,
        Gyro_LShank_mean, Gyro_LShank_std,
        Coord_Acc_Arm_Ankle, Coord_Gyro_Arm_LShank,
        FI_Acc_Ankle_Q1..Q4, Entropy_Acc_Ankle_Q1..Q4,
        FI_Gyro_LShank_Q1..Q4, Entropy_Gyro_LShank_Q1..Q4,
        Jerk_Acc_Ankle_Std_H1/H2, Jerk_Gyro_LShank_Std_H1/H2

    Modelo B (15 features):
        Acc_Ankle_mean, Acc_Ankle_std,
        Acc_Waist_mean, Acc_Waist_std,
        Coord_Acc_Waist_Ankle,
        FI_Acc_Ankle_Q1..Q4, Entropy_Acc_Ankle_Q1..Q4,
        Jerk_Acc_Ankle_Std_H1/H2

Ventana deslizante: 2 segundos, 75% solapamiento
Frecuencia de muestreo estimada dinámicamente desde los timestamps del DGI (~30 Hz).
"""

from typing import List

import numpy as np
import pywt
import scipy.stats
from scipy.signal import periodogram

# ---------------------------------------------------------------------------
# Parámetros de ventana
# ---------------------------------------------------------------------------
T_VENTANA_SEG = 2.0    # 2 segundos
P_SOLAPE      = 0.75   # 75% solapamiento
N_BLOQUES     = 4


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _estimate_fs(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        return 30.0
    diffs = np.diff(timestamps)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 30.0
    return 1000.0 / np.mean(diffs)


def _magnitude(x, y, z) -> np.ndarray:
    return np.sqrt(x**2 + y**2 + z**2)


def _jerk(mag: np.ndarray, ts_ms: np.ndarray) -> np.ndarray:
    dt = np.diff(ts_ms) / 1000.0
    dt = np.where(dt <= 0, 1e-3, dt)
    j = np.diff(mag) / dt
    return np.append(j, j[-1] if len(j) > 0 else 0.0)


def _freezing_index(signal: np.ndarray, fs: float) -> float:
    if len(signal) < 8:
        return 0.0
    freqs, pwr = periodogram(signal, fs=fs)
    loco = np.sum(pwr[(freqs >= 0.5) & (freqs <= 3.0)])
    fog  = np.sum(pwr[(freqs >  3.0) & (freqs <= 8.0)])
    return float(fog / loco) if loco > 1e-9 else 0.0


def _wavelet_entropy(signal: np.ndarray) -> float:
    if len(signal) < 8:
        return 0.0
    try:
        coefs  = pywt.wavedec(signal, "db4", level=3)
        energy = np.array([np.sum(c**2) for c in coefs])
        total  = np.sum(energy)
        if total == 0:
            return 0.0
        p = energy / total
        p = p[p > 0]
        return float(scipy.stats.entropy(p))
    except Exception:
        return 0.0


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(a) != len(b):
        return 0.0
    a_n = (a - np.mean(a)) / (np.std(a) + 1e-9)
    b_n = (b - np.mean(b)) / (np.std(b) + 1e-9)
    return float(np.dot(a_n, b_n) / len(a))


def _mean_std(arr: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std":  float(np.std(arr)),
    }


def _micro_blocks(arr: np.ndarray, prefix: str, fs: float) -> dict:
    feats = {}
    if len(arr) == 0:
        return feats
    tam = max(len(arr) // N_BLOQUES, 1)
    for b in range(N_BLOQUES):
        ini = b * tam
        fin = (b + 1) * tam if b < N_BLOQUES - 1 else len(arr)
        blk = arr[ini:fin]
        feats[f"FI_{prefix}_Q{b+1}"]      = _freezing_index(blk, fs)
        feats[f"Entropy_{prefix}_Q{b+1}"] = _wavelet_entropy(blk)
    return feats


def _jerk_halves(jrk: np.ndarray, prefix: str) -> dict:
    mid = len(jrk) // 2
    return {
        f"Jerk_{prefix}_Std_H1": float(np.std(jrk[:mid])) if mid > 0 else 0.0,
        f"Jerk_{prefix}_Std_H2": float(np.std(jrk[mid:])) if mid > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Builders de features por construcción
# ---------------------------------------------------------------------------

def _features_c(window: dict, fs: float) -> dict:
    """26 features para Modelo C. window: 'ankle', 'lshank', 'arm'"""
    ankle  = window["ankle"]
    lshank = window["lshank"]
    arm    = window["arm"]

    acc_ankle = _magnitude(ankle["acc_x"],   ankle["acc_y"],   ankle["acc_z"])
    gyro_lsh  = _magnitude(lshank["gyro_x"], lshank["gyro_y"], lshank["gyro_z"])
    acc_arm   = _magnitude(arm["acc_x"],     arm["acc_y"],     arm["acc_z"])
    gyro_arm  = _magnitude(arm["gyro_x"],    arm["gyro_y"],    arm["gyro_z"])

    ts = ankle["timestamp"]
    jerk_acc_ankle = _jerk(acc_ankle, ts)
    jerk_gyro_lsh  = _jerk(gyro_lsh,  lshank["timestamp"])

    feats = {}
    feats.update(_mean_std(acc_ankle, "Acc_Ankle"))
    feats.update(_mean_std(gyro_lsh,  "Gyro_LShank"))
    feats["Coord_Acc_Arm_Ankle"]   = _correlation(acc_arm,  acc_ankle) if len(acc_arm)  == len(acc_ankle) else 0.0
    feats["Coord_Gyro_Arm_LShank"] = _correlation(gyro_arm, gyro_lsh)  if len(gyro_arm) == len(gyro_lsh)  else 0.0
    feats.update(_micro_blocks(acc_ankle, "Acc_Ankle",   fs))
    feats.update(_micro_blocks(gyro_lsh,  "Gyro_LShank", fs))
    feats.update(_jerk_halves(jerk_acc_ankle, "Acc_Ankle"))
    feats.update(_jerk_halves(jerk_gyro_lsh,  "Gyro_LShank"))

    return feats


def _features_b(window: dict, fs: float) -> dict:
    """15 features para Modelo B. window: 'ankle', 'waist'"""
    ankle = window["ankle"]
    waist = window["waist"]

    acc_ankle = _magnitude(ankle["acc_x"], ankle["acc_y"], ankle["acc_z"])
    acc_waist = _magnitude(waist["acc_x"], waist["acc_y"], waist["acc_z"])

    ts = ankle["timestamp"]
    jerk_ankle = _jerk(acc_ankle, ts)

    feats = {}
    feats.update(_mean_std(acc_ankle, "Acc_Ankle"))
    feats.update(_mean_std(acc_waist, "Acc_Waist"))
    feats["Coord_Acc_Waist_Ankle"] = _correlation(acc_waist, acc_ankle) if len(acc_waist) == len(acc_ankle) else 0.0
    feats.update(_micro_blocks(acc_ankle, "Acc_Ankle", fs))
    feats.update(_jerk_halves(jerk_ankle, "Acc_Ankle"))

    return feats


# ---------------------------------------------------------------------------
# Ventana deslizante
# ---------------------------------------------------------------------------

def _slice_window(signals: dict, keys: list, start: int, end: int) -> dict:
    return {k: {field: arr[start:end] for field, arr in signals[k].items()} for k in keys}


def _build_windows(signals: dict, ref_key: str, keys: list, fs: float) -> list:
    n      = len(signals[ref_key]["timestamp"])
    size   = max(int(T_VENTANA_SEG * fs), 2)
    step   = max(int(size * (1 - P_SOLAPE)), 1)
    result = []
    for i in range(0, n - size + 1, step):
        window    = _slice_window(signals, keys, i, i + size)
        ts_arr    = signals[ref_key]["timestamp"][i:i + size]
        ts_center = float((ts_arr[0] + ts_arr[-1]) / 2)
        result.append((window, ts_center))
    return result


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def build_features_model_b(extracted: dict) -> List[dict]:
    """Entrada: dict de fog_extractor.extract_for_model_b(). Salida: 15 features + timestamp."""
    fs      = _estimate_fs(extracted["ankle"]["timestamp"])
    windows = _build_windows(extracted, "ankle", ["ankle", "waist"], fs)
    results = []
    for window, ts in windows:
        feats = _features_b(window, fs)
        feats["timestamp"] = ts
        results.append(feats)
    return results


def build_features_model_c(extracted: dict) -> List[dict]:
    """Entrada: dict de fog_extractor.extract_for_model_c(). Salida: 26 features + timestamp."""
    fs      = _estimate_fs(extracted["ankle"]["timestamp"])
    windows = _build_windows(extracted, "ankle", ["ankle", "lshank", "arm"], fs)
    results = []
    for window, ts in windows:
        feats = _features_c(window, fs)
        feats["timestamp"] = ts
        results.append(feats)
    return results