"""
app.py — Servidor FOG con debugs completos
Procesa archivos JSON DGI desde la carpeta data/ y expone los resultados
de los tres modelos FOG vía una API REST.
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

# Flag para debug
DEBUG = True

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG app]", *args, **kwargs)


def _process_json(data: dict) -> dict | None:
    """Corre el pipeline FOG completo sobre un JSON DGI."""
    debug_print("\n" + "🔍"*30)
    debug_print("_process_json - INICIO")
    
    predictor = get_predictor()
    result = {}

    debug_print("\n--- Procesando Modelo A ---")
    try:
        signals_a = fog_extractor.extract_for_model_a(data)
        if signals_a:
            debug_print("✅ Modelo A: Señales obtenidas correctamente")
            features_a = fog_feature_builder.build_features_model_a(signals_a)
            debug_print(f"   Ventanas generadas: {len(features_a)}")
            result['model_a'] = predictor.predict_model_a(features_a)
            if result['model_a']:
                debug_print(f"   Predicciones: {len(result['model_a'])}")
            else:
                debug_print("   ⚠️ Predicción retornó None")
        else:
            debug_print("❌ Modelo A: No se obtuvieron señales")
            result['model_a'] = None
    except Exception as e:
        debug_print(f"❌ Modelo A - ERROR: {str(e)}")
        traceback.print_exc()
        result['model_a'] = None

    debug_print("\n--- Procesando Modelo B ---")
    try:
        signals_b = fog_extractor.extract_for_model_b(data)
        if signals_b:
            debug_print("✅ Modelo B: Señales obtenidas correctamente")
            features_b = fog_feature_builder.build_features_model_b(signals_b)
            debug_print(f"   Ventanas generadas: {len(features_b)}")
            result['model_b'] = predictor.predict_model_b(features_b)
            if result['model_b']:
                debug_print(f"   Predicciones: {len(result['model_b'])}")
            else:
                debug_print("   ⚠️ Predicción retornó None")
        else:
            debug_print("❌ Modelo B: No se obtuvieron señales")
            result['model_b'] = None
    except Exception as e:
        debug_print(f"❌ Modelo B - ERROR: {str(e)}")
        traceback.print_exc()
        result['model_b'] = None

    debug_print("\n--- Procesando Modelo C ---")
    try:
        signals_c = fog_extractor.extract_for_model_c(data)
        if signals_c:
            debug_print("✅ Modelo C: Señales obtenidas correctamente")
            features_c = fog_feature_builder.build_features_model_c(signals_c)
            debug_print(f"   Ventanas generadas: {len(features_c)}")
            result['model_c'] = predictor.predict_model_c(features_c)
            if result['model_c']:
                debug_print(f"   Predicciones: {len(result['model_c'])}")
            else:
                debug_print("   ⚠️ Predicción retornó None")
        else:
            debug_print("❌ Modelo C: No se obtuvieron señales")
            result['model_c'] = None
    except Exception as e:
        debug_print(f"❌ Modelo C - ERROR: {str(e)}")
        traceback.print_exc()
        result['model_c'] = None

    available = [k for k, v in result.items() if v is not None]
    debug_print(f"\n📊 Modelos disponibles: {available}")
    
    if all(v is None for v in result.values()):
        debug_print("❌ Todos los modelos fallaron")
        return None

    debug_print("✅ _process_json - Éxito")
    return result


@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')


@app.route('/api/files', methods=['GET'])
def list_files():
    """Lista todos los archivos JSON disponibles en data/."""
    try:
        debug_print(f"Buscando archivos en: {DATA_DIR}")
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
        files.sort(reverse=True)
        debug_print(f"Archivos encontrados: {len(files)}")
        return jsonify({'files': files})
    except Exception as e:
        debug_print(f"Error listando archivos: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/fog/<filename>', methods=['GET'])
def compute_fog(filename):
    """Procesa el pipeline FOG sobre el archivo especificado."""
    debug_print(f"\n{'='*60}")
    debug_print(f"📁 Procesando archivo: {filename}")
    
    path = os.path.join(DATA_DIR, filename)
    debug_print(f"Ruta completa: {path}")
    
    if not os.path.exists(path):
        debug_print(f"❌ Archivo no encontrado")
        return jsonify({'error': f'Archivo no encontrado: {filename}'}), 404

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        debug_print(f"✅ JSON cargado correctamente")
        debug_print(f"Keys principales del JSON: {list(data.keys())}")
        
        # Verificar estructura
        if not data.get('dgiResults'):
            debug_print("❌ El archivo NO contiene 'dgiResults'")
            # Verificar si tiene otras estructuras conocidas
            if 'imuData' in data:
                debug_print("   Nota: Tiene 'imuData' pero no 'dgiResults' (posible formato IMU v2)")
            if 'patient' in data:
                debug_print(f"   Paciente: {data.get('patient', {}).get('name', 'Desconocido')}")
            return jsonify({'error': 'El archivo no contiene dgiResults'}), 400
        
        debug_print(f"✅ Tiene 'dgiResults' con {len(data['dgiResults'])} items")
        
        # Mostrar detalles de los subtests
        for i, item in enumerate(data['dgiResults']):
            subtest = item.get('subtest', 'SIN NOMBRE')
            imu_count = len(item.get('imuData', []))
            debug_print(f"   Subtest {i+1}: '{subtest}' - {imu_count} muestras IMU")
            
            # Si es Marcha normal, mostrar dispositivos
            if subtest == "Marcha normal" and imu_count > 0:
                devices = set()
                for sample in item.get('imuData', [])[:10]:  # primeras 10 muestras
                    devices.add(sample.get('deviceId', 'unknown'))
                debug_print(f"      Dispositivos encontrados: {list(devices)}")

        results = _process_json(data)
        
        if results is None:
            debug_print("❌ No hay datos suficientes (faltan sensores requeridos)")
            return jsonify({'error': 'No hay datos suficientes (faltan sensores requeridos)'}), 404

        # Info del paciente
        patient = data.get('patient', {})
        meta = {
            'filename': filename,
            'patient': f"{patient.get('name', '')} {patient.get('lastName', '')}".strip() or 'Desconocido',
            'timestamp': data.get('timestamp', ''),
        }
        
        debug_print(f"\n✅ PROCESAMIENTO EXITOSO")
        debug_print(f"   Paciente: {meta['patient']}")
        debug_print(f"   Modelos con datos: {[k for k,v in results.items() if v is not None]}")
        
        if results.get('model_a'):
            debug_print(f"   Modelo A: {len(results['model_a'])} ventanas")
        if results.get('model_b'):
            debug_print(f"   Modelo B: {len(results['model_b'])} ventanas")
        if results.get('model_c'):
            debug_print(f"   Modelo C: {len(results['model_c'])} ventanas")

        return jsonify({'meta': meta, 'fog': results})

    except json.JSONDecodeError as e:
        debug_print(f"❌ Error decodificando JSON: {e}")
        return jsonify({'error': f'JSON inválido: {str(e)}'}), 400
    except Exception as e:
        debug_print(f"❌ Error inesperado: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print('=' * 55)
    print('  FOG Visualizer — Backend (con debugs)')
    print('  http://localhost:5000')
    print('  Archivos en: data/')
    print('=' * 55)
    print('\n🔍 Modo DEBUG activado - Se mostrarán detalles de procesamiento\n')
    app.run(debug=True, host='0.0.0.0', port=5000)