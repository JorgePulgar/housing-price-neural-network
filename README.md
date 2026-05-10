# Housing Price Neural Network

> 🌐 [Versión en español](README_ES.md)

A complete machine learning pipeline to predict housing prices using a neural network, with comparative analysis against a linear regression baseline. Built as a case study for a real estate investment firm, the project covers the full ML lifecycle: exploratory data analysis, preprocessing, model training, hyperparameter tuning, evaluation and REST API deployment.

**Key finding:** on this dataset (545 samples), linear regression outperforms the neural network across all metrics — a common result with small datasets that illustrates when simpler models are the right choice.

---

## Results

| Model | RMSE | RMSE% | R² |
|-------|------|-------|----|
| Neural Network (Adam lr=0.001) | 1.546M INR | 32.4% | 0.5272 |
| Linear Regression (baseline) | 1.325M INR | 27.8% | **0.6529** |

Business objectives (RMSE < 15%, R² > 0.60): **not met by either model** with this dataset size. Linear regression is recommended for production.

---

## Repository structure

```
housing-price-neural-network/
├── housing_neural_network.ipynb   # Main notebook: full pipeline + analysis
├── app.py                         # Flask REST API for price prediction
├── housing_model.keras            # Serialized neural network model
├── housing_scaler.pkl             # Fitted StandardScaler for inference
├── housing_features.pkl           # Feature names in training order
├── Housing.csv                    # Dataset (from Kaggle)
└── README.md
```

## Notebook structure

1. **Configuration & imports**
2. **Data Processing** — loading, EDA (histograms, boxplots, correlation matrix), preprocessing, train-test split
3. **Model Planning** — architecture design and decisions
4. **Model Building & Selection** — training, hyperparameter experiments (Adam vs SGD, learning rates, batch sizes), K-Fold cross-validation
5. **Results** — predictions scatter plot, residual analysis, error distribution
6. **Baseline comparison** — neural network vs linear regression
7. **Deployment** — model serialization and Flask API
8. **Conclusions**

## Dataset

[Housing Prices Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

545 properties with 12 features: area, bedrooms, bathrooms, stories, main road access, guest room, basement, hot water heating, air conditioning, parking spaces, preferred area, and furnishing status. Target: sale price in INR.

Most correlated features with price: `area` (+0.54), `bathrooms` (+0.52), `airconditioning` (+0.45).

## Stack

| Layer | Technology |
|-------|-----------|
| ML / Deep Learning | TensorFlow · Keras · scikit-learn |
| Data & EDA | pandas · numpy · matplotlib · seaborn |
| API | Flask |
| Serialization | joblib · h5py |

## How to run it

### Prerequisites

- Python 3.12 (TensorFlow is not compatible with Python 3.13+)
- Virtual environment recommended

```bash
python -m venv venv_ml
venv_ml\Scripts\activate        # Windows
source venv_ml/bin/activate     # macOS/Linux
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn flask joblib
```

### Run the notebook

Place `Housing.csv` in the same folder as the notebook and run all cells in order. The notebook will generate `housing_model.keras`, `housing_scaler.pkl` and `housing_features.pkl`.

### Run the API

```bash
# With the virtual environment active
python app.py
```

The server starts at `http://localhost:5000`. Test it with:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3,
       "mainroad": "yes", "guestroom": "no", "basement": "no",
       "hotwaterheating": "no", "airconditioning": "yes",
       "parking": 2, "prefarea": "yes", "furnishingstatus": "furnished"}'
```

**Response:**
```json
{
  "moneda": "INR",
  "precio_predicho": 8386399.5,
  "precio_predicho_millones": 8.3864
}
```

Check API health:
```bash
curl http://localhost:5000/health
```

---

## Key findings

- **Linear regression beats the neural network** on all metrics with this dataset size. Neural networks need significantly more data to outperform simpler models.
- **SGD with momentum (lr=0.001) outperformed Adam** — counterintuitively, the slower optimizer found a better minimum (R²=0.61 vs 0.53 for Adam).
- **High K-Fold variance** (R² ranging from 0.44 to -4.54 across folds) confirms the dataset is too small for stable neural network training.
- **74% of predictions have >10% error**, concentrated in low-price properties that the model systematically overestimates.
- **Most important features:** area, bathrooms, air conditioning and stories — consistent across both the correlation analysis and linear regression coefficients.
