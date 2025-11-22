
Advanced Time Series Forecasting With Neural Networks And Explainability

Cross-Validated Hyperparameter Tuning • Synthetic Data • Explainability • Metrics  
*(Full README with Code Included)*

This repository contains a complete end-to-end **Time-Series Forecasting Project** using LSTM with real ML workflows like hyperparameter tuning, SHAP explainability, and synthetic data generation.



Project Overview

This project trains an LSTM model to forecast future values using multivariate sequential data.  
It demonstrates:

- Synthetic data generation  
- Scaling  
- Sequence creation  
- LSTM modeling  
- TimeSeriesSplit cross-validation  
- Hyperparameter tuning  
- SHAP explainability  
- Model evaluation and metrics  



Repository Structure


├── code_for_cultus_project.py   # FULL project code
├── README.md                    # Documentation (this file)




Features

- ✔ Synthetic time-series dataset (sine/cosine + noise)  
- ✔ 30-step lookback window  
- ✔ Predicts next 5 steps  
- ✔ LSTM deep learning model  
- ✔ Cross-validation via TimeSeriesSplit  
- ✔ Hyperparameter tuning (units, learning rate)  
- ✔ Evaluation metrics (MAE, RMSE, MAPE)  
- ✔ SHAP Explainability  



Dataset Description

| Feature | Description |
|--------|-------------|
| feature_1 | Noisy sine wave |
| feature_2 | Noisy cosine wave |
| feature_3 | Linear combination of feature_1 |
| feature_4 | Random uniform values |
| feature_5 | Weekly cycle |
| target | Shifted future signal |



Installation


pip install numpy pandas scikit-learn tensorflow shap matplotlib




How to Run


python code_for_cultus_project.py




Output You Will See

- Best hyperparameters from TimeSeriesSplit  
- Validation & test metrics  
- SHAP explainability plot  
- Predictions vs actual values  



Full Project Code

Below is the exact code used in this project:

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import shap

def generate_synthetic_data(seed=42, n_days=1200):
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

    df = pd.DataFrame({"date": dates})
    df["feature_1"] = np.sin(np.linspace(0, 20, n_days)) + np.random.normal(0, 0.3, n_days)
    df["feature_2"] = np.cos(np.linspace(0, 10, n_days)) + np.random.normal(0, 0.3, n_days)
    df["feature_3"] = df["feature_1"] * 0.5 + np.random.normal(0, 0.1, n_days)
    df["feature_4"] = np.random.uniform(10, 100, n_days)
    df["feature_5"] = np.arange(n_days) % 7
    df["target"] = (
        df["feature_1"].shift(-1)
        + df["feature_2"].shift(-2)
        + np.random.normal(0, 0.1, n_days)
    )

    df = df.dropna()
    return df.set_index("date")

def scale_data(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler

def create_sequences(data, lookback=30, future_steps=5):
    X, y = [], []
    for i in range(len(data) - lookback - future_steps):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + future_steps, -1])
    return np.array(X), np.array(y)

def build_lstm(input_shape, units=64, lr=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units, input_shape=input_shape),
        tf.keras.layers.Dense(5)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mae")
    return model

def evaluate_with_tscv(X, y, units_list, lr_list, epochs=10, batch_size=32, build_fn=None):
    tscv = TimeSeriesSplit(n_splits=4)
    best_score = float("inf")
    best_params = None
    best_model = None

    for units in units_list:
        for lr in lr_list:
            cv_scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = build_fn((X_train.shape[1], X_train.shape[2]), units=units, lr=lr)
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

                preds = model.predict(X_val, verbose=0)
                score = mean_absolute_error(y_val.flatten(), preds.flatten())
                cv_scores.append(score)

            avg_score = np.mean(cv_scores)

            if avg_score < best_score:
                best_score = avg_score
                best_params = {"units": units, "lr": lr}
                best_model = build_fn((X.shape[1], X.shape[2]), units=units, lr=lr)
                best_model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    return best_model, best_params

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    mape = np.mean(np.abs((y_true.flatten() - y_pred.flatten()) / y_true.flatten())) * 100
    return mae, rmse, mape

def compute_shap(model, X_sample):
    X_flat = X_sample.reshape(X_sample.shape[0], -1)

    def predict_fn(x):
        return model.predict(x.reshape(-1, X_sample.shape[1], X_sample.shape[2]), verbose=0)

    explainer = shap.KernelExplainer(predict_fn, X_flat)
    shap_vals = explainer.shap_values(X_flat, nsamples=100)

    shap_vals = shap_vals[0].reshape(X_sample.shape[0], X_sample.shape[1], X_sample.shape[2])
    shap_mean = shap_vals.mean(axis=1)

    X_mean = X_sample.mean(axis=1)

    shap.summary_plot(shap_mean, X_mean)

    return shap_mean

def main():
    df = generate_synthetic_data()
    scaled, scaler = scale_data(df)

    X, y = create_sequences(scaled, lookback=30, future_steps=5)

    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    best_model, best_params = evaluate_with_tscv(
        X_train, y_train,
        units_list=[32, 64],
        lr_list=[0.001, 0.0005],
        epochs=10,
        build_fn=build_lstm
    )

    best_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    preds = best_model.predict(X_test, verbose=0)
    mae, rmse, mape = compute_metrics(y_test, preds)

    print("Final Metrics:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MAPE:", mape)

    compute_shap(best_model, X_test[:5])

if __name__ == "__main__":
    main()

