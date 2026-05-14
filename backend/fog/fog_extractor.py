"""
fog_extractor.py
----------------
Extrae y organiza las señales de acelerómetro y giroscopio del JSON DGI
para cada una de las tres construcciones de modelos FOG.

Restricción de uso: SOLO aplica a archivos de tipo 'dgi' (sensor_type == 'dgi').

Señales disponibles en el JSON DGI (deviceId):
  - BASE-SPINE  : acelerómetro + giroscopio (cadera/espalda)
  - LEFT-HAND   : acelerómetro + giroscopio (brazo izquierdo)
  - RIGHT-HAND  : acelerómetro + giroscopio (brazo derecho)
  - LEFT-ANKLE  : acelerómetro + giroscopio (tobillo izquierdo)
  - RIGHT-ANKLE : acelerómetro + giroscopio (tobillo derecho)

Cobertura por construcción:
  - Construcción A : BASE-SPINE (acc+gyro), LEFT-HAND (acc+gyro),
                     LEFT-ANKLE (acc+gyro), RIGHT-ANKLE (acc+gyro)
  - Construcción B : BASE-SPINE (solo acc), LEFT-ANKLE (solo acc)
  - Construcción C : LEFT-ANKLE (acc+gyro), LEFT-HAND (acc+gyro)

Frecuencia de muestreo real del DGI: ~30 Hz (≈33 ms por muestra).
Se usa el subtest "Marcha normal" del DGI, consistente con el resto
de la plataforma (ImuV2Data).
"""

from collections import defaultdict
from typing import Optional

import numpy as np

# Nombre del subtest DGI que contiene la marcha válida para FOG
DGI_SUBTEST_MARCHA_NORMAL = "Marcha normal"

# Factor de conversión idéntico al usado en ImuV2Data
CONVERSION_FACTOR = 9.8 / 4130


def _group_by_device(imu_data_list: list) -> dict:
    """
    Agrupa las muestras IMU por deviceId y las ordena por timestamp.
    Replica la lógica de ImuV2Data.group_imu_data_by_device para consistencia.
    """
    devices = defaultdict(list)
    for entry in imu_data_list:
        devices[entry["deviceId"]].append(entry)
    for device_id in devices:
        devices[device_id].sort(key=lambda x: x["timestamp"])
    return dict(devices)


def _extract_imu_data_list(json_content: dict) -> list:
    """
    Obtiene la lista de muestras IMU del subtest 'Marcha normal' del DGI.
    Retorna lista vacía si el JSON no tiene el formato esperado.
    """
    dgi_results = json_content.get("dgiResults")
    if not dgi_results:
        return []
    normal_gait = next(
        (item for item in dgi_results if item.get("subtest") == DGI_SUBTEST_MARCHA_NORMAL),
        None
    )
    if normal_gait is None:
        return []
    return normal_gait.get("imuData", [])


def _signals_from_device(device_data: list, include_gyro: bool = True) -> dict:
    """
    Extrae arrays numpy de acc y (opcionalmente) gyro de una lista de muestras
    de un mismo deviceId.

    Aplica el mismo factor de conversión que ImuV2Data para el acelerómetro.
    El giroscopio se retorna en unidades crudas del sensor (igual que ImuV2Data).

    Retorna un dict con claves:
      acc_x, acc_y, acc_z, timestamp
      gyro_x, gyro_y, gyro_z  (solo si include_gyro=True y hay datos)
    """
    if not device_data:
        return {}

    acc_x = np.array([e["accelerometer"]["x"] * CONVERSION_FACTOR for e in device_data])
    acc_y = np.array([e["accelerometer"]["y"] * CONVERSION_FACTOR for e in device_data])
    acc_z = np.array([e["accelerometer"]["z"] * CONVERSION_FACTOR for e in device_data])
    timestamps = np.array([e["timestamp"] for e in device_data])

    result = {
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "timestamp": timestamps,
    }

    if include_gyro:
        result["gyro_x"] = np.array([e["gyroscope"]["x"] for e in device_data])
        result["gyro_y"] = np.array([e["gyroscope"]["y"] for e in device_data])
        result["gyro_z"] = np.array([e["gyroscope"]["z"] for e in device_data])

    return result


def extract_for_model_a(json_content: dict) -> Optional[dict]:
    """
    Construcción A — Datos multimodal completo.
    Sensores: BASE-SPINE, LEFT-HAND, LEFT-ANKLE, RIGHT-ANKLE (acc + gyro).

    Retorna dict con claves:
      'base_spine', 'left_hand', 'left_ankle', 'right_ankle'
    Cada valor es un dict con acc_x/y/z, gyro_x/y/z, timestamp.
    Retorna None si algún sensor crítico no tiene datos.
    """
    imu_data_list = _extract_imu_data_list(json_content)
    if not imu_data_list:
        return None

    grouped = _group_by_device(imu_data_list)

    base_spine_data = grouped.get("BASE-SPINE", [])
    left_hand_data = grouped.get("LEFT-HAND", [])
    left_ankle_data = grouped.get("LEFT-ANKLE", [])
    right_ankle_data = grouped.get("RIGHT-ANKLE", [])

    # Construcción A requiere todos los sensores
    if not base_spine_data or not left_hand_data or not left_ankle_data or not right_ankle_data:
        return None

    return {
        "base_spine": _signals_from_device(base_spine_data, include_gyro=True),
        "left_hand": _signals_from_device(left_hand_data, include_gyro=True),
        "left_ankle": _signals_from_device(left_ankle_data, include_gyro=True),
        "right_ankle": _signals_from_device(right_ankle_data, include_gyro=True),
    }


def extract_for_model_b(json_content: dict) -> Optional[dict]:
    """
    Construcción B — Datos Daphnet + multimodal.
    Sensores: BASE-SPINE y LEFT-ANKLE (solo acelerómetro).

    Retorna dict con claves:
      'base_spine', 'left_ankle'
    Cada valor es un dict con acc_x/y/z, timestamp (sin gyro).
    Retorna None si algún sensor crítico no tiene datos.
    """
    imu_data_list = _extract_imu_data_list(json_content)
    if not imu_data_list:
        return None

    grouped = _group_by_device(imu_data_list)

    base_spine_data = grouped.get("BASE-SPINE", [])
    left_ankle_data = grouped.get("LEFT-ANKLE", [])

    if not base_spine_data or not left_ankle_data:
        return None

    return {
        "base_spine": _signals_from_device(base_spine_data, include_gyro=False),
        "left_ankle": _signals_from_device(left_ankle_data, include_gyro=False),
    }


def extract_for_model_c(json_content: dict) -> Optional[dict]:
    """
    Construcción C — Datos multimodal (tobillo izquierdo + brazo izquierdo).
    Sensores: LEFT-ANKLE y LEFT-HAND (acc + gyro).

    Retorna dict con claves:
      'left_ankle', 'left_hand'
    Cada valor es un dict con acc_x/y/z, gyro_x/y/z, timestamp.
    Retorna None si algún sensor crítico no tiene datos.
    """
    imu_data_list = _extract_imu_data_list(json_content)
    if not imu_data_list:
        return None

    grouped = _group_by_device(imu_data_list)

    left_ankle_data = grouped.get("LEFT-ANKLE", [])
    left_hand_data = grouped.get("LEFT-HAND", [])

    if not left_ankle_data or not left_hand_data:
        return None

    return {
        "left_ankle": _signals_from_device(left_ankle_data, include_gyro=True),
        "left_hand": _signals_from_device(left_hand_data, include_gyro=True),
    }
