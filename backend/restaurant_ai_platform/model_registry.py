import json
import os
from datetime import datetime
from typing import Dict, Any


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.json")


def train_and_save_model(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple demo model training.
    Calculates average daily sales and stores it.
    """

    sales_data = features.get("sales", [])

    if not sales_data:
        raise ValueError("No sales data provided for training")

    total = sum(item["daily_sales_total"] for item in sales_data)
    avg_daily_sales = total / len(sales_data)

    model = {
        "model_name": "average_daily_sales_model",
        "version": "1.0",
        "trained_at": datetime.utcnow().isoformat(),
        "avg_daily_sales": avg_daily_sales,
    }

    with open(MODEL_PATH, "w") as f:
        json.dump(model, f)

    return model


def load_model() -> Dict[str, Any]:
    """
    Loads the saved model from disk.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found")

    with open(MODEL_PATH, "r") as f:
        return json.load(f)


def predict() -> Dict[str, Any]:
    """
    Uses stored average to simulate next day prediction.
    """

    model = load_model()

    prediction = {
        "predicted_next_day_sales": round(model["avg_daily_sales"] * 1.02, 2),
        "model_version": model["version"],
        "prediction_timestamp": datetime.utcnow().isoformat(),
    }

    return prediction