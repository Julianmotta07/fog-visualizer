"""
fog_extractor.py
----------------
Extrae y organiza las señales de acelerómetro y giroscopio del JSON DGI
para cada una de las tres construcciones de modelos FOG.

Solo aplica a archivos de tipo 'dgi'.
Usa exclusivamente el subtest "Marcha normal".

Mapeo deviceId → nombre de sensor en los features:
    LEFT-ANKLE  → 'ankle'  (acc)  y  'lshank' (gyro)
    BASE-SPINE  → 'waist'  (acc)
    LEFT-HAND   → 'arm'    (acc + gyro)

Cobertura por construcción:
    Modelo B : LEFT-ANKLE (solo acc)  + BASE-SPINE (solo acc)
    Modelo C : LEFT-ANKLE (acc+gyro)  + LEFT-HAND (acc+gyro)
"""

from collections import defaultdict
from typing import Optional

import numpy as np

DGI_SUBTEST_MARCHA_NORMAL = "Marcha normal"
CONVERSION_FACTOR = 9.8 / 4130


def _group_by_device(imu_data_list: list) -> dict:
    devices = defaultdict(list)
    for entry in imu_data_list:
        devices[entry["deviceId"]].append(entry)
    for device_id in devices:
        devices[device_id].sort(key=lambda x: x["timestamp"])
    return dict(devices)


def _get_marcha_normal_imu(json_content: dict) -> list:
    dgi_results = json_content.get("dgiResults")
    if not dgi_results:
        return []
    normal_gait = next(
        (item for item in dgi_results if item.get("subtest") == DGI_SUBTEST_MARCHA_NORMAL),
        None
    )
    return normal_gait["imuData"] if normal_gait else []


def _extract_signals(device_data: list, include_gyro: bool = True) -> dict:
    if not device_data:
        return {}

    result = {
        "acc_x":     np.array([e["accelerometer"]["x"] * CONVERSION_FACTOR for e in device_data]),
        "acc_y":     np.array([e["accelerometer"]["y"] * CONVERSION_FACTOR for e in device_data]),
        "acc_z":     np.array([e["accelerometer"]["z"] * CONVERSION_FACTOR for e in device_data]),
        "timestamp": np.array([e["timestamp"] for e in device_data]),
    }

    if include_gyro:
        result["gyro_x"] = np.array([e["gyroscope"]["x"] for e in device_data])
        result["gyro_y"] = np.array([e["gyroscope"]["y"] for e in device_data])
        result["gyro_z"] = np.array([e["gyroscope"]["z"] for e in device_data])

    return result


def extract_for_model_b(json_content: dict) -> Optional[dict]:
    """
    Modelo B — LEFT-ANKLE (solo acc) + BASE-SPINE (solo acc).
    Retorna dict con claves 'ankle' y 'waist', o None si faltan sensores.
    """
    imu_list = _get_marcha_normal_imu(json_content)
    if not imu_list:
        return None

    grouped   = _group_by_device(imu_list)
    ankle_raw = grouped.get("LEFT-ANKLE", [])
    waist_raw = grouped.get("BASE-SPINE", [])

    if not ankle_raw or not waist_raw:
        return None

    return {
        "ankle": _extract_signals(ankle_raw, include_gyro=False),
        "waist": _extract_signals(waist_raw, include_gyro=False),
    }


def extract_for_model_c(json_content: dict) -> Optional[dict]:
    """
    Modelo C — LEFT-ANKLE (acc+gyro) + LEFT-HAND (acc+gyro).
    Retorna dict con claves 'ankle', 'lshank' y 'arm', o None si faltan sensores.
    """
    imu_list = _get_marcha_normal_imu(json_content)
    if not imu_list:
        return None

    grouped   = _group_by_device(imu_list)
    ankle_raw = grouped.get("LEFT-ANKLE", [])
    arm_raw   = grouped.get("LEFT-HAND", [])

    if not ankle_raw or not arm_raw:
        return None

    ankle_signals = _extract_signals(ankle_raw, include_gyro=True)
    arm_signals   = _extract_signals(arm_raw,   include_gyro=True)

    return {
        "ankle":  {k: ankle_signals[k] for k in ["acc_x", "acc_y", "acc_z", "timestamp"]},
        "lshank": {**{k: ankle_signals[k] for k in ["gyro_x", "gyro_y", "gyro_z"]},
                   "timestamp": ankle_signals["timestamp"]},
        "arm":    arm_signals,
    }