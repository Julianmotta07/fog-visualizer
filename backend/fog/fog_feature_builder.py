"""
fog_feature_builder.py
-----------------------
Construcción de features para inferencia FOG a partir de señales DGI.

Implementa el mismo pipeline que se usó en el entrenamiento de los tres modelos:
  1. Ventana deslizante de 1 segundo con 50 % de solapamiento.
  2. Extracción de biomarcadores estadísticos, espectrales y wavelet por ventana.
  3. Retorna una lista de dicts {features, timestamp_center} listos para el predictor.

FRECUENCIA DE MUESTREO:
  El dataset de entrenamiento (multimodal) operaba a 500 Hz.
  El DGI muestrea a ~30 Hz (≈33 ms por muestra).
  Las ventanas se definen siempre en TIEMPO (1 segundo), por lo que el número
  de muestras se ajusta dinámicamente usando la frecuencia estimada de cada señal.

FEATURES GENERADAS::
    - mean, std, rms, min, max, rango, p2p, asimetria, curtosis, energia, iqr, mad
    - Freezing Index (bandas 0.5-3 Hz y 3-8 Hz)
    - Entropía Wavelet (db4, nivel 3)
    - Micro-resúmenes en 4 bloques temporales (FI + Entropy por bloque)
    - Jerk lineal (derivada de la magnitud de aceleración)
    - Magnitud vectorial de aceleración y giroscopio
    - Coordinación cruzada (correlación brazo-tobillo) donde aplique

Cada construcción usa un subconjunto distinto de estas features según los
sensores disponibles. El scaler del modelo ya fue ajustado a ese subconjunto,
así que fog_predictor.py se encarga de seleccionar las columnas correctas
antes de escalar.
"""

from typing import Optional

import numpy as np
import pywt
import scipy.stats
from scipy.signal import butter, filtfilt, periodogram


# ---------------------------------------------------------------------------
# Parámetros globales del pipeline
# ---------------------------------------------------------------------------
T_VENTANA_SEG = 1.0       # Duración de cada ventana en segundos
P_SOLAPE = 0.5            # Fracción de solapamiento entre ventanas
N_BLOQUES = 4             # Micro-resúmenes temporales por ventana
CUTOFF_LOWPASS_HZ = 10.0  # Frecuencia de corte del filtro pasa-bajas (Hz)
FILTER_ORDER = 4           # Orden del filtro Butterworth


# ---------------------------------------------------------------------------
# Utilidades de señal
# ---------------------------------------------------------------------------

def _estimate_fs(timestamps: np.ndarray) -> float:
    """
    Estima la frecuencia de muestreo en Hz a partir del vector de timestamps
    (en milisegundos). Devuelve un mínimo de 1 Hz como salvaguarda.
    """
    if len(timestamps) < 2:
        return 25.0  # Valor por defecto conservador
    diffs = np.diff(timestamps)
    diffs_positivos = diffs[diffs > 0]
    if len(diffs_positivos) == 0:
        return 25.0
    mean_ms = np.mean(diffs_positivos)
    if mean_ms <= 0:
        return 25.0
    return 1000.0 / mean_ms


def _lowpass_filter(signal: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Filtro pasa-bajas Butterworth. Devuelve la señal original si fs <= 2*cutoff."""
    if len(signal) < 5:
        return signal
    nyquist = 0.5 * fs
    if cutoff >= nyquist:
        return signal
    b, a = butter(order, cutoff / nyquist, btype="low", analog=False)
    if len(signal) < 3 * max(len(b), len(a)):
        return signal  # Señal demasiado corta para filtrar
    try:
        return filtfilt(b, a, signal)
    except Exception:
        return signal


def _magnitude(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def _jerk(magnitude: np.ndarray, timestamps_ms: np.ndarray) -> np.ndarray:
    """
    Derivada numérica de la magnitud (jerk). Mismo tamaño que la entrada.
    Versión corregida que maneja arrays vacíos o de tamaño 1.
    """
    # Manejar casos donde no hay suficientes datos
    if len(magnitude) < 2 or len(timestamps_ms) < 2:
        return np.zeros_like(magnitude)
    
    dt_s = np.diff(timestamps_ms) / 1000.0
    dt_s = np.where(dt_s <= 0, 1e-3, dt_s)  # Evitar divisiones por cero
    j = np.diff(magnitude) / dt_s
    
    # Si j está vacío, retornar ceros
    if len(j) == 0:
        return np.zeros_like(magnitude)
    
    return np.append(j, j[-1])


# ---------------------------------------------------------------------------
# Features estadísticos y espectrales sobre un array 1-D
# ---------------------------------------------------------------------------

def _calcular_freezing_index(signal: np.ndarray, fs: float) -> float:
    """
    Freezing Index (FI) = potencia FOG (3-8 Hz) / potencia locomoción (0.5-3 Hz).
    Implementación idéntica a calcular_freezing_index() de los notebooks.
    """
    if len(signal) < 10:
        return 0.0
    try:
        freqs, powers = periodogram(signal, fs=fs)
        banda_loco = (freqs >= 0.5) & (freqs <= 3.0)
        banda_fog = (freqs > 3.0) & (freqs <= 8.0)
        poder_loco = np.sum(powers[banda_loco])
        poder_fog = np.sum(powers[banda_fog])
        if poder_loco < 1e-5:
            return 0.0
        return float(poder_fog / poder_loco)
    except Exception:
        return 0.0


def _calcular_entropia_wavelet(signal: np.ndarray, wavelet: str = "db4", nivel: int = 3) -> float:
    """
    Entropía de Shannon sobre la distribución de energía wavelet.
    Implementación idéntica a calcular_entropia_wavelet() de los notebooks.
    """
    if len(signal) < 10:
        return 0.0
    try:
        coefs = pywt.wavedec(signal, wavelet, level=nivel)
        energia = np.array([np.sum(np.square(c)) for c in coefs])
        total = np.sum(energia)
        if total == 0:
            return 0.0
        p = energia / total
        p = p[p > 0]
        return float(scipy.stats.entropy(p))
    except Exception:
        return 0.0


def _calcular_coordinacion(signal_a: np.ndarray, signal_b: np.ndarray) -> float:
    """
    Correlación cruzada normalizada entre dos señales (brazo vs tobillo).
    """
    if len(signal_a) == 0 or len(signal_b) == 0 or len(signal_a) != len(signal_b):
        return 0.0
    try:
        a_norm = (signal_a - np.mean(signal_a)) / (np.std(signal_a) + 1e-5)
        b_norm = (signal_b - np.mean(signal_b)) / (np.std(signal_b) + 1e-5)
        corr = np.correlate(a_norm, b_norm, mode="valid")
        if len(corr) == 0:
            return 0.0
        return float(np.max(corr) / len(signal_a))
    except Exception:
        return 0.0


def _stats_from_array(arr: np.ndarray, prefix: str) -> dict:
    """
    Extrae los mismos estadísticos que extraer_caracteristicas_ventanas()
    de los notebooks para un array 1-D.
    """
    if len(arr) == 0:
        return {}
    try:
        return {
            f"{prefix}_mean": float(np.mean(arr)),
            f"{prefix}_std": float(np.std(arr)),
            f"{prefix}_rms": float(np.sqrt(np.mean(arr ** 2))),
            f"{prefix}_min": float(np.min(arr)),
            f"{prefix}_max": float(np.max(arr)),
            f"{prefix}_rango": float(np.max(arr) - np.min(arr)),
            f"{prefix}_p2p": float(np.ptp(arr)),
            f"{prefix}_asimetria": float(scipy.stats.skew(arr)),
            f"{prefix}_curtosis": float(scipy.stats.kurtosis(arr)),
            f"{prefix}_q25": float(np.percentile(arr, 25)),
            f"{prefix}_q75": float(np.percentile(arr, 75)),
            f"{prefix}_iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            f"{prefix}_energia": float(np.sum(arr ** 2)),
            f"{prefix}_mad": float(np.mean(np.abs(arr - np.mean(arr)))),
        }
    except Exception:
        return {}


def _micro_resumen(arr: np.ndarray, prefix: str, fs: float) -> dict:
    """
    Micro-resúmenes en N_BLOQUES bloques temporales (FI + Entropía por bloque).
    """
    feats = {}
    if len(arr) < N_BLOQUES:
        return feats
    tam_bloque = max(len(arr) // N_BLOQUES, 1)
    for b in range(N_BLOQUES):
        inicio = b * tam_bloque
        fin = (b + 1) * tam_bloque if b < N_BLOQUES - 1 else len(arr)
        bloque = arr[inicio:fin]
        feats[f"{prefix}_FI_Q{b + 1}"] = _calcular_freezing_index(bloque, fs)
        feats[f"{prefix}_Entropy_Q{b + 1}"] = _calcular_entropia_wavelet(bloque)
    return feats


# ---------------------------------------------------------------------------
# Construcción de features por ventana para cada construcción
# ---------------------------------------------------------------------------

def _features_from_window_a(window_data: dict, fs: float) -> dict:
    """
    Features para Construcción A.
    Sensores: base_spine, left_hand, left_ankle, right_ankle (acc + gyro).
    """
    feats = {}

    # Magnitudes de aceleración
    acc_ankle = _magnitude(
        window_data["left_ankle"]["acc_x"],
        window_data["left_ankle"]["acc_y"],
        window_data["left_ankle"]["acc_z"],
    )
    acc_r_ankle = _magnitude(
        window_data["right_ankle"]["acc_x"],
        window_data["right_ankle"]["acc_y"],
        window_data["right_ankle"]["acc_z"],
    )
    acc_arm = _magnitude(
        window_data["left_hand"]["acc_x"],
        window_data["left_hand"]["acc_y"],
        window_data["left_hand"]["acc_z"],
    )
    acc_spine = _magnitude(
        window_data["base_spine"]["acc_x"],
        window_data["base_spine"]["acc_y"],
        window_data["base_spine"]["acc_z"],
    )

    # Magnitudes de giroscopio
    gyro_ankle = _magnitude(
        window_data["left_ankle"]["gyro_x"],
        window_data["left_ankle"]["gyro_y"],
        window_data["left_ankle"]["gyro_z"],
    )
    gyro_arm = _magnitude(
        window_data["left_hand"]["gyro_x"],
        window_data["left_hand"]["gyro_y"],
        window_data["left_hand"]["gyro_z"],
    )
    gyro_spine = _magnitude(
        window_data["base_spine"]["gyro_x"],
        window_data["base_spine"]["gyro_y"],
        window_data["base_spine"]["gyro_z"],
    )

    # Jerk
    ts_ankle = window_data["left_ankle"]["timestamp"]
    ts_arm = window_data["left_hand"]["timestamp"]
    jerk_ankle = _jerk(acc_ankle, ts_ankle)
    jerk_arm = _jerk(acc_arm, ts_arm)
    jerk_gyro_ankle = _jerk(gyro_ankle, ts_ankle)

    # Estadísticos base
    feats.update(_stats_from_array(acc_ankle, "Acc_LAnkle"))
    feats.update(_stats_from_array(acc_r_ankle, "Acc_RAnkle"))
    feats.update(_stats_from_array(acc_arm, "Acc_LHand"))
    feats.update(_stats_from_array(acc_spine, "Acc_Spine"))
    feats.update(_stats_from_array(gyro_ankle, "Gyro_LAnkle"))
    feats.update(_stats_from_array(gyro_arm, "Gyro_LHand"))
    feats.update(_stats_from_array(gyro_spine, "Gyro_Spine"))

    # Micro-resúmenes
    feats.update(_micro_resumen(acc_ankle, "Acc_LAnkle", fs))
    feats.update(_micro_resumen(gyro_ankle, "Gyro_LAnkle", fs))
    feats.update(_micro_resumen(acc_spine, "Acc_Spine", fs))

    # Coordinación brazo-tobillo
    if len(acc_arm) == len(acc_ankle):
        feats["Coord_Acc_Arm_LAnkle"] = _calcular_coordinacion(acc_arm, acc_ankle)
    if len(gyro_arm) == len(gyro_ankle):
        feats["Coord_Gyro_Arm_LAnkle"] = _calcular_coordinacion(gyro_arm, gyro_ankle)

    # Jerk
    if len(jerk_ankle) > 0:
        mitad_jerk = len(jerk_ankle) // 2
        feats["Jerk_Acc_LAnkle_Std_H1"] = float(np.std(jerk_ankle[:mitad_jerk])) if mitad_jerk > 0 else 0.0
        feats["Jerk_Acc_LAnkle_Std_H2"] = float(np.std(jerk_ankle[mitad_jerk:])) if mitad_jerk > 0 else 0.0
    else:
        feats["Jerk_Acc_LAnkle_Std_H1"] = 0.0
        feats["Jerk_Acc_LAnkle_Std_H2"] = 0.0
        
    if len(jerk_arm) > 0:
        mitad_jerk_arm = len(jerk_arm) // 2
        feats["Jerk_Acc_LHand_Std_H1"] = float(np.std(jerk_arm[:mitad_jerk_arm])) if mitad_jerk_arm > 0 else 0.0
        feats["Jerk_Acc_LHand_Std_H2"] = float(np.std(jerk_arm[mitad_jerk_arm:])) if mitad_jerk_arm > 0 else 0.0
    else:
        feats["Jerk_Acc_LHand_Std_H1"] = 0.0
        feats["Jerk_Acc_LHand_Std_H2"] = 0.0
        
    if len(jerk_gyro_ankle) > 0:
        mitad_jerk_gyro = len(jerk_gyro_ankle) // 2
        feats["Jerk_Gyro_LAnkle_Std_H1"] = float(np.std(jerk_gyro_ankle[:mitad_jerk_gyro])) if mitad_jerk_gyro > 0 else 0.0
        feats["Jerk_Gyro_LAnkle_Std_H2"] = float(np.std(jerk_gyro_ankle[mitad_jerk_gyro:])) if mitad_jerk_gyro > 0 else 0.0
    else:
        feats["Jerk_Gyro_LAnkle_Std_H1"] = 0.0
        feats["Jerk_Gyro_LAnkle_Std_H2"] = 0.0

    return feats


def _features_from_window_b(window_data: dict, fs: float) -> dict:
    """
    Features para Construcción B.
    Sensores: base_spine y left_ankle (solo acelerómetro).
    """
    feats = {}

    acc_ankle = _magnitude(
        window_data["left_ankle"]["acc_x"],
        window_data["left_ankle"]["acc_y"],
        window_data["left_ankle"]["acc_z"],
    )
    acc_spine = _magnitude(
        window_data["base_spine"]["acc_x"],
        window_data["base_spine"]["acc_y"],
        window_data["base_spine"]["acc_z"],
    )

    ts_ankle = window_data["left_ankle"]["timestamp"]
    ts_spine = window_data["base_spine"]["timestamp"]
    jerk_ankle = _jerk(acc_ankle, ts_ankle)
    jerk_spine = _jerk(acc_spine, ts_spine)

    feats.update(_stats_from_array(acc_ankle, "Acc_LAnkle"))
    feats.update(_stats_from_array(acc_spine, "Acc_Spine"))
    feats.update(_micro_resumen(acc_ankle, "Acc_LAnkle", fs))
    feats.update(_micro_resumen(acc_spine, "Acc_Spine", fs))

    if len(jerk_ankle) > 0:
        mitad_ankle = len(jerk_ankle) // 2
        feats["Jerk_Acc_LAnkle_Std_H1"] = float(np.std(jerk_ankle[:mitad_ankle])) if mitad_ankle > 0 else 0.0
        feats["Jerk_Acc_LAnkle_Std_H2"] = float(np.std(jerk_ankle[mitad_ankle:])) if mitad_ankle > 0 else 0.0
    else:
        feats["Jerk_Acc_LAnkle_Std_H1"] = 0.0
        feats["Jerk_Acc_LAnkle_Std_H2"] = 0.0
        
    if len(jerk_spine) > 0:
        mitad_spine = len(jerk_spine) // 2
        feats["Jerk_Acc_Spine_Std_H1"] = float(np.std(jerk_spine[:mitad_spine])) if mitad_spine > 0 else 0.0
        feats["Jerk_Acc_Spine_Std_H2"] = float(np.std(jerk_spine[mitad_spine:])) if mitad_spine > 0 else 0.0
    else:
        feats["Jerk_Acc_Spine_Std_H1"] = 0.0
        feats["Jerk_Acc_Spine_Std_H2"] = 0.0

    if len(acc_ankle) == len(acc_spine):
        feats["Coord_Acc_Spine_LAnkle"] = _calcular_coordinacion(acc_spine, acc_ankle)

    return feats


def _features_from_window_c(window_data: dict, fs: float) -> dict:
    """
    Features para Construcción C.
    Sensores: left_ankle y left_hand (acc + gyro).
    """
    feats = {}

    acc_ankle = _magnitude(
        window_data["left_ankle"]["acc_x"],
        window_data["left_ankle"]["acc_y"],
        window_data["left_ankle"]["acc_z"],
    )
    acc_arm = _magnitude(
        window_data["left_hand"]["acc_x"],
        window_data["left_hand"]["acc_y"],
        window_data["left_hand"]["acc_z"],
    )
    gyro_ankle = _magnitude(
        window_data["left_ankle"]["gyro_x"],
        window_data["left_ankle"]["gyro_y"],
        window_data["left_ankle"]["gyro_z"],
    )
    gyro_arm = _magnitude(
        window_data["left_hand"]["gyro_x"],
        window_data["left_hand"]["gyro_y"],
        window_data["left_hand"]["gyro_z"],
    )

    ts_ankle = window_data["left_ankle"]["timestamp"]
    ts_arm = window_data["left_hand"]["timestamp"]
    jerk_ankle = _jerk(acc_ankle, ts_ankle)
    jerk_gyro_ankle = _jerk(gyro_ankle, ts_ankle)
    jerk_gyro_arm = _jerk(gyro_arm, ts_arm)

    feats.update(_stats_from_array(acc_ankle, "Acc_LAnkle"))
    feats.update(_stats_from_array(acc_arm, "Acc_LHand"))
    feats.update(_stats_from_array(gyro_ankle, "Gyro_LAnkle"))
    feats.update(_stats_from_array(gyro_arm, "Gyro_LHand"))

    feats.update(_micro_resumen(acc_ankle, "Acc_LAnkle", fs))
    feats.update(_micro_resumen(gyro_ankle, "Gyro_LAnkle", fs))
    feats.update(_micro_resumen(gyro_arm, "Gyro_LHand", fs))

    if len(acc_arm) == len(acc_ankle):
        feats["Coord_Acc_Arm_LAnkle"] = _calcular_coordinacion(acc_arm, acc_ankle)
    if len(gyro_arm) == len(gyro_ankle):
        feats["Coord_Gyro_Arm_LAnkle"] = _calcular_coordinacion(gyro_arm, gyro_ankle)

    if len(jerk_ankle) > 0:
        mitad_ankle = len(jerk_ankle) // 2
        feats["Jerk_Acc_LAnkle_Std_H1"] = float(np.std(jerk_ankle[:mitad_ankle])) if mitad_ankle > 0 else 0.0
        feats["Jerk_Acc_LAnkle_Std_H2"] = float(np.std(jerk_ankle[mitad_ankle:])) if mitad_ankle > 0 else 0.0
    else:
        feats["Jerk_Acc_LAnkle_Std_H1"] = 0.0
        feats["Jerk_Acc_LAnkle_Std_H2"] = 0.0
        
    if len(jerk_gyro_ankle) > 0:
        mitad_gyro_ankle = len(jerk_gyro_ankle) // 2
        feats["Jerk_Gyro_LAnkle_Std_H1"] = float(np.std(jerk_gyro_ankle[:mitad_gyro_ankle])) if mitad_gyro_ankle > 0 else 0.0
        feats["Jerk_Gyro_LAnkle_Std_H2"] = float(np.std(jerk_gyro_ankle[mitad_gyro_ankle:])) if mitad_gyro_ankle > 0 else 0.0
    else:
        feats["Jerk_Gyro_LAnkle_Std_H1"] = 0.0
        feats["Jerk_Gyro_LAnkle_Std_H2"] = 0.0
        
    if len(jerk_gyro_arm) > 0:
        mitad_gyro_arm = len(jerk_gyro_arm) // 2
        feats["Jerk_Gyro_LHand_Std_H1"] = float(np.std(jerk_gyro_arm[:mitad_gyro_arm])) if mitad_gyro_arm > 0 else 0.0
        feats["Jerk_Gyro_LHand_Std_H2"] = float(np.std(jerk_gyro_arm[mitad_gyro_arm:])) if mitad_gyro_arm > 0 else 0.0
    else:
        feats["Jerk_Gyro_LHand_Std_H1"] = 0.0
        feats["Jerk_Gyro_LHand_Std_H2"] = 0.0

    return feats


# ---------------------------------------------------------------------------
# Ventana deslizante genérica
# ---------------------------------------------------------------------------

def _slice_sensor(sensor_dict: dict, idx_start: int, idx_end: int) -> dict:
    """Corta todos los arrays de un dict de señales en el rango [idx_start, idx_end)."""
    return {k: v[idx_start:idx_end] for k, v in sensor_dict.items()}


def _build_windows(signals: dict, sensor_keys: list, fs: float,
                   t_ventana: float = T_VENTANA_SEG, p_solape: float = P_SOLAPE) -> list:
    """
    Genera una lista de dicts de señales recortadas por ventana deslizante.
    Usa el primer sensor de sensor_keys como referencia de longitud y timestamps.
    Retorna lista de (window_data_dict, timestamp_center_ms).
    """
    ref_key = sensor_keys[0]
    n_total = len(signals[ref_key]["timestamp"])
    
    # Verificar que haya suficientes datos
    if n_total < 2:
        return []
    
    tamano = max(int(t_ventana * fs), 2)
    paso = max(int(tamano * (1 - p_solape)), 1)

    windows = []
    for i in range(0, n_total - tamano + 1, paso):
        fin = i + tamano
        window_data = {
            k: _slice_sensor(signals[k], i, fin)
            for k in sensor_keys
        }
        ts_array = signals[ref_key]["timestamp"][i:fin]
        
        # Verificar que la ventana tenga datos
        if len(ts_array) == 0:
            continue
            
        ts_center = float((ts_array[0] + ts_array[-1]) / 2)
        windows.append((window_data, ts_center))

    return windows


# ---------------------------------------------------------------------------
# Funciones públicas de la API del módulo
# ---------------------------------------------------------------------------

def build_features_model_a(extracted_signals: dict) -> list:
    """
    Construye el vector de features para cada ventana de la Construcción A.

    Parámetro:
        extracted_signals : dict retornado por fog_extractor.extract_for_model_a()

    Retorna:
        Lista de dicts con claves 'timestamp' (ms, centro de ventana) y
        todas las features del modelo A.
    """
    if not extracted_signals or "left_ankle" not in extracted_signals:
        return []
    
    fs = _estimate_fs(extracted_signals["left_ankle"]["timestamp"])
    sensor_keys = ["base_spine", "left_hand", "left_ankle", "right_ankle"]
    windows = _build_windows(extracted_signals, sensor_keys, fs)

    results = []
    for window_data, ts_center in windows:
        try:
            feats = _features_from_window_a(window_data, fs)
            feats["timestamp"] = ts_center
            results.append(feats)
        except Exception:
            continue
    return results


def build_features_model_b(extracted_signals: dict) -> list:
    """
    Construye el vector de features para cada ventana de la Construcción B.

    Parámetro:
        extracted_signals : dict retornado por fog_extractor.extract_for_model_b()

    Retorna:
        Lista de dicts con claves 'timestamp' y features del modelo B.
    """
    if not extracted_signals or "left_ankle" not in extracted_signals:
        return []
    
    fs = _estimate_fs(extracted_signals["left_ankle"]["timestamp"])
    sensor_keys = ["base_spine", "left_ankle"]
    windows = _build_windows(extracted_signals, sensor_keys, fs)

    results = []
    for window_data, ts_center in windows:
        try:
            feats = _features_from_window_b(window_data, fs)
            feats["timestamp"] = ts_center
            results.append(feats)
        except Exception:
            continue
    return results


def build_features_model_c(extracted_signals: dict) -> list:
    """
    Construye el vector de features para cada ventana de la Construcción C.

    Parámetro:
        extracted_signals : dict retornado por fog_extractor.extract_for_model_c()

    Retorna:
        Lista de dicts con claves 'timestamp' y features del modelo C.
    """
    if not extracted_signals or "left_ankle" not in extracted_signals:
        return []
    
    fs = _estimate_fs(extracted_signals["left_ankle"]["timestamp"])
    sensor_keys = ["left_ankle", "left_hand"]
    windows = _build_windows(extracted_signals, sensor_keys, fs)

    results = []
    for window_data, ts_center in windows:
        try:
            feats = _features_from_window_c(window_data, fs)
            feats["timestamp"] = ts_center
            results.append(feats)
        except Exception:
            continue
    return results