import logging
import os
from typing import List, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

FOG_THRESHOLD = 0.15
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def _load(filename: str):
    path = os.path.join(_MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    return joblib.load(path)


def _get_feature_names(model) -> list:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "_feature_names"):
        return list(model._feature_names)
    raise AttributeError(f"El modelo {type(model).__name__} no tiene atributo de nombres de features.")


def _build_matrix(feature_dicts: list, feature_names: list) -> np.ndarray:
    X = np.zeros((len(feature_dicts), len(feature_names)), dtype=np.float64)
    for i, fd in enumerate(feature_dicts):
        for j, col in enumerate(feature_names):
            X[i, j] = fd.get(col, 0.0)
    return X


def _run_inference(model, scaler, feature_dicts: list, threshold: float) -> list:
    if not feature_dicts:
        return []
    feature_names = _get_feature_names(model)
    timestamps = [fd.get("timestamp", 0.0) for fd in feature_dicts]
    clean_dicts = [{k: v for k, v in fd.items() if k != "timestamp"} for fd in feature_dicts]
    X = _build_matrix(clean_dicts, feature_names)
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    return [
        {"timestamp": int(ts), "fog": int(p >= threshold), "prob": round(float(p), 4)}
        for ts, p in zip(timestamps, probs)
    ]


class FogPredictor:
    def __init__(self, threshold: float = FOG_THRESHOLD):
        self.threshold = threshold
        self._model_a = self._scaler_a = None
        self._model_b = self._scaler_b = None
        self._model_c = self._scaler_c = None

    def _ensure_a(self):
        if self._model_a is None:
            self._model_a = _load("model_a.pkl")
            self._scaler_a = _load("scaler_a.pkl")
            logger.info("Modelo FOG A cargado (%s).", type(self._model_a).__name__)

    def _ensure_b(self):
        if self._model_b is None:
            self._model_b = _load("model_b.pkl")
            self._scaler_b = _load("scaler_b.pkl")
            logger.info("Modelo FOG B cargado (%s).", type(self._model_b).__name__)

    def _ensure_c(self):
        if self._model_c is None:
            self._model_c = _load("model_c.pkl")
            self._scaler_c = _load("scaler_c.pkl")
            logger.info("Modelo FOG C cargado (%s).", type(self._model_c).__name__)

    def predict_model_a(self, feature_dicts: list) -> Optional[List[dict]]:
        try:
            self._ensure_a()
            return _run_inference(self._model_a, self._scaler_a, feature_dicts, self.threshold)
        except FileNotFoundError as e:
            logger.warning("Modelo A no disponible: %s", e)
            return None
        except Exception as e:
            logger.error("Error en predicción FOG modelo A: %s", e)
            return None

    def predict_model_b(self, feature_dicts: list) -> Optional[List[dict]]:
        try:
            self._ensure_b()
            return _run_inference(self._model_b, self._scaler_b, feature_dicts, self.threshold)
        except FileNotFoundError as e:
            logger.warning("Modelo B no disponible: %s", e)
            return None
        except Exception as e:
            logger.error("Error en predicción FOG modelo B: %s", e)
            return None

    def predict_model_c(self, feature_dicts: list) -> Optional[List[dict]]:
        try:
            self._ensure_c()
            return _run_inference(self._model_c, self._scaler_c, feature_dicts, self.threshold)
        except FileNotFoundError as e:
            logger.warning("Modelo C no disponible: %s", e)
            return None
        except Exception as e:
            logger.error("Error en predicción FOG modelo C: %s", e)
            return None


_predictor: Optional[FogPredictor] = None

def get_predictor() -> FogPredictor:
    global _predictor
    if _predictor is None:
        _predictor = FogPredictor()
    return _predictor