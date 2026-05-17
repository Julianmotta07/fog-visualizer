"""
app.py — Servidor FOG Visualizer
Procesa archivos JSON DGI desde la carpeta data/ y expone los resultados
del modelo compuesto (wB=0.35, wC=0.65, umbral=0.30) vía una API REST.
"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import sys
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fog import fog_extractor, fog_feature_builder
from fog.fog_predictor import get_predictor

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
DEBUG = True

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG app]", *args, **kwargs)


def _process_json(data: dict) -> dict | None:
    """Corre el pipeline del modelo compuesto sobre un JSON DGI."""
    debug_print("\n" + "="*50)
    debug_print("_process_json - INICIO (Modelo Compuesto)")

    predictor = get_predictor()

    debug_print("\n--- Extrayendo señales Modelo B ---")
    signals_b = fog_extractor.extract_for_model_b(data)
    if signals_b:
        debug_print("✅ Señales B obtenidas")
    else:
        debug_print("❌ No se obtuvieron señales B (falta LEFT-ANKLE o BASE-SPINE)")

    debug_print("\n--- Extrayendo señales Modelo C ---")
    signals_c = fog_extractor.extract_for_model_c(data)
    if signals_c:
        debug_print("✅ Señales C obtenidas")
    else:
        debug_print("❌ No se obtuvieron señales C (falta LEFT-ANKLE o LEFT-HAND)")

    if signals_b is None or signals_c is None:
        debug_print("❌ Faltan sensores críticos para el modelo compuesto")
        return None

    try:
        features_b = fog_feature_builder.build_features_model_b(signals_b)
        features_c = fog_feature_builder.build_features_model_c(signals_c)
        debug_print(f"   Ventanas B: {len(features_b)} | Ventanas C: {len(features_c)}")

        ensemble = predictor.predict_ensemble(features_b, features_c)
        if ensemble is None:
            debug_print("❌ predict_ensemble retornó None")
            return None

        debug_print(f"✅ Ensemble calculado: {len(ensemble)} ventanas")
        fog_count = sum(1 for p in ensemble if p["fog"] == 1)
        debug_print(f"   Ventanas con FOG: {fog_count} / {len(ensemble)}")
        return {"ensemble": ensemble}

    except Exception as e:
        debug_print(f"❌ ERROR en pipeline: {str(e)}")
        traceback.print_exc()
        return None


@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')


@app.route('/api/files', methods=['GET'])
def list_files():
    """Lista todos los archivos JSON disponibles en data/."""
    try:
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
        files.sort(reverse=True)
        debug_print(f"Archivos encontrados: {len(files)}")
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/fog/<filename>', methods=['GET'])
def compute_fog(filename):
    """Procesa el pipeline del modelo compuesto sobre el archivo especificado."""
    debug_print(f"\n{'='*60}")
    debug_print(f"Procesando: {filename}")

    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return jsonify({'error': f'Archivo no encontrado: {filename}'}), 404

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        debug_print(f"Keys del JSON: {list(data.keys())}")

        if not data.get('dgiResults'):
            return jsonify({'error': 'El archivo no contiene dgiResults'}), 400

        debug_print(f"dgiResults: {len(data['dgiResults'])} subtests")
        for item in data['dgiResults']:
            subtest   = item.get('subtest', 'SIN NOMBRE')
            imu_count = len(item.get('imuData', []))
            debug_print(f"  '{subtest}': {imu_count} muestras")
            if subtest == "Marcha normal" and imu_count > 0:
                devices = set(s.get('deviceId') for s in item['imuData'][:20])
                debug_print(f"    Dispositivos: {devices}")

        result = _process_json(data)
        if result is None:
            return jsonify({'error': 'No hay datos suficientes (faltan sensores requeridos)'}), 404

        patient = data.get('patient', {})
        meta = {
            'filename': filename,
            'patient':  f"{patient.get('name', '')} {patient.get('lastName', '')}".strip() or 'Desconocido',
            'timestamp': data.get('timestamp', ''),
        }

        debug_print(f"✅ Éxito — Paciente: {meta['patient']}")
        return jsonify({'meta': meta, 'fog': result})

    except json.JSONDecodeError as e:
        return jsonify({'error': f'JSON inválido: {str(e)}'}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print('=' * 55)
    print('  FOG Visualizer — Modelo Compuesto (wB=0.35, wC=0.65)')
    print('  http://localhost:5000')
    print('  Archivos en: data/')
    print('  Umbral: 30% | Ventana: 2s / 75% solapamiento')
    print('=' * 55)
    app.run(debug=True, host='0.0.0.0', port=5000)