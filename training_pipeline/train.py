import os
import warnings

# 1. Suppress all standard warnings and TensorFlow C++ logs globally
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # 3 = FATAL only (silences info, warnings, and errors)
)

import json
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import pandas as pd
import hopsworks
import shutil
from dotenv import load_dotenv, find_dotenv

# Scikit-Learn & XGBoost
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# TensorFlow/Keras (RNN)
import tensorflow as tf

tf.get_logger().setLevel("ERROR")  # Silence TensorFlow python logger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

# Initialize environment configurations
load_dotenv(find_dotenv())
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY is missing! Check your .env file.")


def fetch_and_preprocess_data():
    """Fetches historical features and targets from the Feature Store and preprocesses them."""
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    # Fetch historical data directly from the feature group
    aqi_fg = fs.get_feature_group(name="aqi_features", version=4)
    df = aqi_fg.read()

    # ---------------------------------------------------------
    # PREPROCESSING STEP 1: Chronological Sorting
    # ---------------------------------------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ---------------------------------------------------------
    # PREPROCESSING STEP 2: Missing Value Imputation
    # ---------------------------------------------------------
    print("Handling missing values...")
    df = df.set_index("timestamp")
    # Fix for Pandas FutureWarning: explicitly infer objects before interpolating
    df = df.infer_objects(copy=False)
    df = df.interpolate(method="time")
    df = df.ffill().bfill()
    df = df.reset_index()

    # ---------------------------------------------------------
    # PREPROCESSING STEP 3: Outlier Capping (Clipping)
    # ---------------------------------------------------------
    print("Capping outliers...")
    numeric_cols = ["aqi", "co", "no2", "o3", "pm2_5", "pm10", "lag_1_aqi"]
    for col in numeric_cols:
        upper_limit = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=upper_limit)

    # ---------------------------------------------------------
    # PREPROCESSING STEP 4: Feature/Target Split
    # ---------------------------------------------------------
    # Restore pollutant features to ensure accurate AQI predictions.
    # We drop 'pm10' to reduce multicollinearity (since it's 97% correlated with pm2_5).
    X = df.drop(columns=["aqi", "city", "timestamp", "pm10"])
    y = df["aqi"]

    # ---------------------------------------------------------
    # PREPROCESSING STEP 5: Chronological Train/Test Split
    # ---------------------------------------------------------
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # ---------------------------------------------------------
    # PREPROCESSING STEP 6: Feature Scaling
    # ---------------------------------------------------------
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        project,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        X_test,
        df,
        X.columns.tolist(),
    )


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Trains multiple models and evaluates their performance."""
    results = {}
    models = {}

    # ---------------------------------------------------------
    # Model 1: Ridge Regression
    # ---------------------------------------------------------
    print("Training Ridge Regression...")
    ridge = Ridge(alpha=10.0)  # Increased alpha to reduce overfitting
    ridge.fit(X_train, y_train)
    preds_ridge_test = ridge.predict(X_test)
    preds_ridge_train = ridge.predict(X_train)
    models["Ridge"] = ridge
    results["Ridge"] = {
        "test": evaluate_preds(y_test, preds_ridge_test),
        "train": evaluate_preds(y_train, preds_ridge_train),
    }

    # ---------------------------------------------------------
    # Model 2: Random Forest
    # ---------------------------------------------------------
    print("Training Random Forest...")
    # Add regularization parameters to prevent overfitting
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    preds_rf_test = rf.predict(X_test)
    preds_rf_train = rf.predict(X_train)
    models["Random_Forest"] = rf
    results["Random_Forest"] = {
        "test": evaluate_preds(y_test, preds_rf_test),
        "train": evaluate_preds(y_train, preds_rf_train),
    }

    # ---------------------------------------------------------
    # Model 3: XGBoost
    # ---------------------------------------------------------
    print("Training XGBoost...")
    # Add regularization parameters to prevent overfitting
    xgb = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    preds_xgb_test = xgb.predict(X_test)
    preds_xgb_train = xgb.predict(X_train)
    models["XGBoost"] = xgb
    results["XGBoost"] = {
        "test": evaluate_preds(y_test, preds_xgb_test),
        "train": evaluate_preds(y_train, preds_xgb_train),
    }

    # ---------------------------------------------------------
    # Model 4: Recurrent Neural Network (RNN)
    # ---------------------------------------------------------
    print("Training RNN (TensorFlow)...")
    X_train_rnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_rnn = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Fix for Keras UserWarning: Use explicit Input layer
    # Add Dropout and L2 Regularization to prevent overfitting
    rnn = Sequential(
        [
            Input(shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])),
            SimpleRNN(32, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.2),
            Dense(16, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.2),
            Dense(1),
        ]
    )
    rnn.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    rnn.fit(
        X_train_rnn,
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=[es],
        verbose=0,
    )

    preds_rnn_test = rnn.predict(X_test_rnn, verbose=0).flatten()
    preds_rnn_train = rnn.predict(X_train_rnn, verbose=0).flatten()
    models["RNN"] = rnn
    results["RNN"] = {
        "test": evaluate_preds(y_test, preds_rnn_test),
        "train": evaluate_preds(y_train, preds_rnn_train),
    }

    return models, results


def evaluate_preds(y_true, y_pred):
    """Calculates evaluation metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    # 1. Fetch & Preprocess
    (
        project,
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        X_test_unscaled,
        df_full,
        feature_names,
    ) = fetch_and_preprocess_data()

    # 2. Train & Evaluate Models
    print("\nStarting Model Training...")
    models, results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # 3. Display Results
    print("\n--- Model Evaluation Results (Train vs Test) ---")
    best_model_name = None
    best_r2 = -float("inf")

    for name, metrics in results.items():
        print(f"\n{name}:")
        print(
            f"  Train MAE:  {metrics['train']['MAE']:.2f} | Test MAE:  {metrics['test']['MAE']:.2f}"
        )
        print(
            f"  Train RMSE: {metrics['train']['RMSE']:.2f} | Test RMSE: {metrics['test']['RMSE']:.2f}"
        )
        print(
            f"  Train R²:   {metrics['train']['R2']:.4f} | Test R²:   {metrics['test']['R2']:.4f}"
        )

        # Determine the best model based on Test R2 Score
        if metrics["test"]["R2"] > best_r2:
            best_r2 = metrics["test"]["R2"]
            best_model_name = name

    print(f"\nBest Model: {best_model_name} (Test R2 = {best_r2:.4f})")

    # 4. Store the best trained model in the Model Registry
    print(f"\nSaving {best_model_name} to Model Registry...")
    mr = project.get_model_registry()
    if os.path.exists("model_dir"):
        shutil.rmtree("model_dir")
    os.makedirs("model_dir", exist_ok=True)

    # Save Analytics Data
    # 1. Model Metrics
    with open("model_dir/model_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    # 2. EDA Sample
    df_full.tail(500).to_csv("model_dir/eda_sample.csv", index=False)

    # Save Scaler
    joblib.dump(scaler, "model_dir/scaler.pkl")

    # Save the winning model
    best_model = models[best_model_name]
    if best_model_name == "RNN":
        best_model.save("model_dir/aqi_model.h5")
    elif best_model_name == "XGBoost":
        best_model.save_model("model_dir/aqi_model.json")
    else:
        joblib.dump(best_model, "model_dir/aqi_model.pkl")

    # Generate Analytics Plots
    print("Generating Analytics Plots...")

    # 2.5 Correlation Matrix
    plt.figure(figsize=(10, 8))
    cols_to_corr = ["aqi", "pm2_5", "pm10", "temp", "humidity", "co", "no2", "o3"]
    # Only keep cols that actually exist in df_full
    cols_to_corr = [c for c in cols_to_corr if c in df_full.columns]
    corr = df_full[cols_to_corr].corr()
    sns.heatmap(
        corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, fmt=".2f", linewidths=0.5
    )
    plt.title("Feature Correlation Matrix")
    plt.savefig("model_dir/correlation_matrix.png", bbox_inches="tight", dpi=150)
    plt.close()

    # 3. Residuals Plot
    best_preds = best_model.predict(X_test).flatten()
    residuals = y_test - best_preds
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="#D946EF")
    plt.title("Residual Distribution (True - Predicted)")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.savefig("model_dir/residuals.png", bbox_inches="tight", dpi=150)
    plt.close()

    # 4. Error Over Category
    def categorize_aqi(aqi):
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy/Sens."
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"

    cats = y_test.apply(categorize_aqi)
    errs = abs(y_test - best_preds)
    err_df = pd.DataFrame({"Category": cats, "Error": errs})
    cat_order = [
        "Good",
        "Moderate",
        "Unhealthy/Sens.",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=err_df,
        x="Category",
        y="Error",
        order=cat_order,
        errorbar=None,
        palette="magma",
    )
    plt.title("Mean Absolute Error by AQI Category")
    plt.ylabel("MAE")
    plt.savefig("model_dir/error_by_category.png", bbox_inches="tight", dpi=150)
    plt.close()

    # 5. SHAP Summary Plot (For XGBoost or Random Forest)
    if best_model_name in ["XGBoost", "Random_Forest"]:
        try:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test_unscaled)
            plt.figure()
            shap.summary_plot(
                shap_values, X_test_unscaled, feature_names=feature_names, show=False
            )
            plt.savefig("model_dir/shap_summary.png", bbox_inches="tight", dpi=150)
            plt.close()
        except Exception as e:
            print(f"Failed to generate SHAP: {e}")

    # Fix for Hopsworks ProvenanceWarning: Provide an input example so it infers the schema
    input_sample = X_train[:1].tolist()

    # Register in Hopsworks
    aqi_model = mr.python.create_model(
        name="aqi_predictor_v3",
        metrics=results[best_model_name]["test"],
        input_example=input_sample,
        description=f"Best performing model: {best_model_name}",
    )
    aqi_model.save("model_dir")
    print("\nModel successfully registered in Hopsworks!")


if __name__ == "__main__":
    main()
