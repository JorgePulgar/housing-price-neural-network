# Red Neuronal para Predicción de Precios de Viviendas

Pipeline completo de machine learning para predecir precios de viviendas usando una red neuronal, con análisis comparativo frente a una regresión lineal como baseline. Desarrollado como caso de estudio para una firma de inversión inmobiliaria, el proyecto cubre el ciclo de vida completo de un modelo ML: análisis exploratorio de datos, preprocesamiento, entrenamiento, ajuste de hiperparámetros, evaluación y despliegue como API REST.

**Conclusión principal:** en este dataset (545 muestras), la regresión lineal supera a la red neuronal en todos los indicadores — un resultado habitual con datasets pequeños que ilustra cuándo los modelos más simples son la elección correcta.

---

## Resultados

| Modelo | RMSE | RMSE% | R² |
|--------|------|-------|----|
| Red Neuronal (Adam lr=0.001) | 1.546M INR | 32.4% | 0.5272 |
| Regresión Lineal (baseline) | 1.325M INR | 27.8% | **0.6529** |

Objetivos de negocio (RMSE < 15%, R² > 0.60): **no alcanzados por ningún modelo** con este tamaño de dataset. Se recomienda la regresión lineal para producción.

---

## Estructura del repositorio

```
housing-price-neural-network/
├── housing_neural_network.ipynb   # Notebook principal: pipeline completo + análisis
├── app.py                         # API REST Flask para predicción de precios
├── housing_model.keras            # Modelo de red neuronal serializado
├── housing_scaler.pkl             # StandardScaler ajustado para inferencia
├── housing_features.pkl           # Nombres de features en orden de entrenamiento
├── Housing.csv                    # Dataset (de Kaggle)
└── README.md
```

## Estructura del notebook

1. **Configuración e imports**
2. **Data Processing** — carga, EDA (histogramas, boxplots, matriz de correlación), preprocesamiento, división train-test
3. **Model Planning** — diseño de la arquitectura y decisiones tomadas
4. **Model Building & Selection** — entrenamiento, experimentos de hiperparámetros (Adam vs SGD, learning rates, batch sizes), validación cruzada K-Fold
5. **Presentación de resultados** — scatter de predicciones, análisis de residuos, distribución de errores
6. **Comparación con baseline** — red neuronal vs regresión lineal
7. **Deployment** — serialización del modelo y API Flask
8. **Conclusiones**

## Dataset

[Housing Prices Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

545 propiedades con 12 features: área, habitaciones, baños, pisos, acceso a calle principal, habitación de invitados, sótano, calefacción de agua caliente, aire acondicionado, plazas de aparcamiento, zona preferencial y estado del mobiliario. Variable objetivo: precio de venta en INR.

Features más correlacionadas con el precio: `area` (+0.54), `bathrooms` (+0.52), `airconditioning` (+0.45).

## Stack

| Capa | Tecnología |
|------|-----------|
| ML / Deep Learning | TensorFlow · Keras · scikit-learn |
| Datos y EDA | pandas · numpy · matplotlib · seaborn |
| API | Flask |
| Serialización | joblib · h5py |

## Cómo ejecutarlo

### Requisitos previos

- Python 3.12 (TensorFlow no es compatible con Python 3.13+)
- Se recomienda entorno virtual

```bash
python -m venv venv_ml
venv_ml\Scripts\activate        # Windows
source venv_ml/bin/activate     # macOS/Linux
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn flask joblib
```

### Ejecutar el notebook

Coloca `Housing.csv` en la misma carpeta que el notebook y ejecuta todas las celdas en orden. El notebook generará `housing_model.keras`, `housing_scaler.pkl` y `housing_features.pkl`.

### Ejecutar la API

```bash
# Con el entorno virtual activado
python app.py
```

El servidor arranca en `http://localhost:5000`. Pruébalo con:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3,
       "mainroad": "yes", "guestroom": "no", "basement": "no",
       "hotwaterheating": "no", "airconditioning": "yes",
       "parking": 2, "prefarea": "yes", "furnishingstatus": "furnished"}'
```

**Respuesta:**
```json
{
  "moneda": "INR",
  "precio_predicho": 8386399.5,
  "precio_predicho_millones": 8.3864
}
```

Comprobar que la API está activa:
```bash
curl http://localhost:5000/health
```

---

## Conclusiones principales

- **La regresión lineal supera a la red neuronal** en todos los indicadores con este tamaño de dataset. Las redes neuronales necesitan significativamente más datos para superar a modelos más simples.
- **SGD con momentum (lr=0.001) superó a Adam** — de forma contraintuitiva, el optimizador más lento encontró un mejor mínimo (R²=0.61 vs 0.53 para Adam).
- **Alta varianza en K-Fold** (R² entre 0.44 y -4.54 según el fold) confirma que el dataset es demasiado pequeño para un entrenamiento estable de redes neuronales.
- **El 74% de las predicciones tienen un error superior al 10%**, concentrado en propiedades de precio bajo que el modelo sobreestima sistemáticamente.
- **Features más importantes:** área, baños, aire acondicionado y número de pisos — consistente entre el análisis de correlaciones y los coeficientes de la regresión lineal.
