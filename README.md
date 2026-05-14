# FOG Visualizer

Proyecto standalone para visualizar los resultados del pipeline de detección de
Freezing of Gait (FOG) sobre archivos JSON DGI.

## Estructura

```
fog_visualizer/
├── backend/
│   ├── app.py                 
│   ├── requirements.txt
│   └── fog/
│       ├── __init__.py
│       ├── fog_extractor.py
│       ├── fog_feature_builder.py
│       ├── fog_predictor.py
│       └── models/             
│           ├── model_a.pkl
│           ├── model_b.pkl
│           ├── model_c.pkl
│           ├── scaler_a.pkl
│           ├── scaler_b.pkl
│           └── scaler_c.pkl
├── data/                      
│   └── 20250716234548_...json
└── frontend/
    └── index.html             
```

## Setup

### 1. Instalar dependencias Python
```bash
cd backend
pip install -r requirements.txt
```

### 2. Colocar los modelos
Renombra y copia los .pkl en `backend/fog/models/`:
```
ModeloA.pkl  →  model_a.pkl
ModeloB.pkl  →  model_b.pkl
ModeloC.pkl  →  model_c.pkl
ScalerA.pkl  →  scaler_a.pkl
ScalerB.pkl  →  scaler_b.pkl
ScalerC.pkl  →  scaler_c.pkl
```

### 3. Colocar los archivos JSON
Copia los archivos DGI en la carpeta `data/`.

### 4. Correr el servidor
```bash
cd backend
python app.py
```

### 5. Abrir el visualizador
Abre en el navegador: http://localhost:5000

## Uso
- El panel izquierdo lista todos los archivos JSON en `data/`
- Haz clic en un archivo para procesarlo
- El pipeline corre automáticamente y muestra la gráfica
- Se muestran los tres modelos (A, B, C) con su probabilidad FOG por ventana
- La línea roja punteada indica el umbral
