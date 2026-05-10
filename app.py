
from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow import keras

app = Flask(__name__)

# Cargar modelo y preprocesadores al arrancar
model   = keras.models.load_model("housing_model.keras")
scaler  = joblib.load("housing_scaler.pkl")
features = joblib.load("housing_features.pkl")
PRICE_SCALE = 1e6

BINARY_COLS = ["mainroad", "guestroom", "basement",
               "hotwaterheating", "airconditioning", "prefarea"]


def preprocess_input(data: dict) -> np.ndarray:
    """Preprocesa un diccionario de entrada al mismo formato que el entrenamiento."""
    # Variables binarias
    for col in BINARY_COLS:
        data[col] = 1 if str(data.get(col, "no")).lower() == "yes" else 0

    # One-hot encoding de furnishingstatus
    fs = str(data.pop("furnishingstatus", "unfurnished")).lower()
    data["furnishingstatus_furnished"]      = 1 if fs == "furnished" else 0
    data["furnishingstatus_semi-furnished"] = 1 if fs == "semi-furnished" else 0
    data["furnishingstatus_unfurnished"]    = 1 if fs == "unfurnished" else 0

    # Construir vector en el orden correcto
    vector = np.array([[data.get(f, 0) for f in features]], dtype=float)
    return scaler.transform(vector)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se recibieron datos JSON"}), 400

        X_proc = preprocess_input(data.copy())
        pred   = model.predict(X_proc, verbose=0)[0][0] * PRICE_SCALE

        return jsonify({
            "precio_predicho": round(float(pred), 2),
            "precio_predicho_millones": round(float(pred) / 1e6, 4),
            "moneda": "INR"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "housing_model.keras"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
