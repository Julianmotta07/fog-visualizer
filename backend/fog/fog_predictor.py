"""
fog_predictor.py
-----------------
Modelo compuesto FOG — votación ponderada

    P_final = 0.35 × P(SVM) + 0.65 × P(XGBoost)
    FOG si P_final >= 0.30

Modelo A descartado (peso = 0).

Modelos esperados en fog/models/:
    model_b.pkl   ← ModeloB_E6_SVM.pkl       (renombrado)
    model_c.pkl   ← ModeloC_E6_XGBoost.pkl   (renombrado)
    scaler_b.pkl  ← ScalerB_E6.pkl           (renombrado)
    scaler_c.pkl  ← ScalerC_E6.pkl           (renombrado)

Salida por ventana:
    {"timestamp": <int ms>, "fog": <0|1>, "prob": <float 0-1>}
"""

import logging
import os
from typing import List, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

WEIGHT_B           = 0.35
WEIGHT_C           = 0.65
ENSEMBLE_THRESHOLD = 0.30

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def _load(filename: str):
    path = os.path.join(_MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Archivo no encontrado: {path}. Coloca los .pkl en {_MODELS_DIR}")
    return joblib.load(path)


def _get_feature_names(model) -> list:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "_feature_names"):
        return list(model._feature_names)
    raise AttributeError(f"El modelo {type(model).__name__} no tiene nombres de features.")


def _build_matrix(feature_dicts: list, feature_names: list) -> np.ndarray:
    X = np.zeros((len(feature_dicts), len(feature_names)), dtype=np.float64)
    for i, fd in enumerate(feature_dicts):
        for j, col in enumerate(feature_names):
            X[i, j] = fd.get(col, 0.0)
    return X


def _get_probs(model, scaler, feature_dicts: list) -> tuple:
    """Retorna (timestamps, probs_array)."""
    feature_names = _get_feature_names(model)
    timestamps    = [fd.get("timestamp", 0.0) for fd in feature_dicts]
    clean_dicts   = [{k: v for k, v in fd.items() if k != "timestamp"} for fd in feature_dicts]
    X             = _build_matrix(clean_dicts, feature_names)
    X_scaled      = scaler.transform(X)
    probs         = model.predict_proba(X_scaled)[:, 1]
    return timestamps, probs


class FogPredictor:
    """Predictor FOG — modelo compuesto (wB=0.35, wC=0.65, umbral=0.30)."""

    def __init__(self):
        self._model_b = self._scaler_b = None
        self._model_c = self._scaler_c = None

    def _ensure_b(self):
        if self._model_b is None:
            self._model_b  = _load("model_b.pkl")
            self._scaler_b = _load("scaler_b.pkl")
            logger.info("Modelo FOG B cargado (%s).", type(self._model_b).__name__)

    def _ensure_c(self):
        if self._model_c is None:
            self._model_c  = _load("model_c.pkl")
            self._scaler_c = _load("scaler_c.pkl")
            logger.info("Modelo FOG C cargado (%s).", type(self._model_c).__name__)

    def predict_ensemble(self, features_b: list, features_c: list) -> Optional[List[dict]]:
        """
        Calcula P_final = 0.35*P(B) + 0.65*P(C) y aplica umbral 0.30.
        Retorna lista de {timestamp, fog, prob} o None si falla.
        """
        try:
            self._ensure_b()
            self._ensure_c()

            if not features_b or not features_c:
                return None

            n          = min(len(features_b), len(features_c))
            features_b = features_b[:n]
            features_c = features_c[:n]

            timestamps, probs_b = _get_probs(self._model_b, self._scaler_b, features_b)
            _,          probs_c = _get_probs(self._model_c, self._scaler_c, features_c)

            probs_ensemble = WEIGHT_B * probs_b + WEIGHT_C * probs_c

            return [
                {
                    "timestamp": int(ts),
                    "fog":       int(p >= ENSEMBLE_THRESHOLD),
                    "prob":      round(float(p), 4),
                }
                for ts, p in zip(timestamps, probs_ensemble)
            ]

        except FileNotFoundError as e:
            logger.warning("Modelo no disponible: %s", e)
            return None
        except Exception as e:
            logger.error("Error en predicción FOG ensemble: %s", e)
            return None


_predictor: Optional[FogPredictor] = None


def get_predictor() -> FogPredictor:
    global _predictor
    if _predictor is None:
        _predictor = FogPredictor()
    return _predictor