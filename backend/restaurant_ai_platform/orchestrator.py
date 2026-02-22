from datetime import datetime


PIPELINE_ORDER = [
    "1_data_ingestion",
    "2_data_warehouse",
    "3_feature_engineering",
    "4_feature_store_sync",
    "5_ml_prediction",
    "6_optimization",
    "7_api_serving",
    "8_dashboard_update",
]


def run_step(step_name: str) -> None:
    print(f"[{datetime.utcnow().isoformat()}] Running step: {step_name}")


def run_pipeline() -> None:
    for step in PIPELINE_ORDER:
        run_step(step)


if __name__ == "__main__":
    run_pipeline()