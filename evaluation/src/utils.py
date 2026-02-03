import json
from pathlib import Path
from datetime import datetime


def load_predictions(predictions_file: str | Path) -> dict:
    predictions_file = Path(predictions_file)
    
    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    
    with open(predictions_file, "r") as f:
        data = json.load(f)
    
    return data


def get_latest_predictions(predictions_dir: str | Path) -> Path:
    predictions_dir = Path(predictions_dir)
    
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
    
    prediction_files = list(predictions_dir.glob("predictions_*.json"))
    
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found in: {predictions_dir}")
    
    # Sort by modification time, get latest
    latest = max(prediction_files, key=lambda p: p.stat().st_mtime)
    return latest


def save_predictions(
    predictions: list[dict],
    output_dir: str | Path,
    model_name: str = "unknown",
    filename: str | None = None
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = filename or f"predictions_{timestamp}.json"
    output_path = output_dir / filename
    
    output = {
        "metadata": {
            "timestamp": timestamp,
            "num_samples": len(predictions),
            "model": model_name
        },
        "predictions": predictions
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    return output_path


# def save_results(
#     metrics: dict,
#     output_dir: str | Path,
#     model_name: str = "unknown",
#     metrics_used: list[str] | None = None,
#     filename: str | None = None,
#     predictions_file: str | None = None
# ) -> Path:
    
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = filename or f"evaluation_metrics_{timestamp}.json"
#     output_path = output_dir / filename
    
#     output = {
#         "metadata": {
#             "timestamp": timestamp,
#             "model": model_name,
#             "metrics_used": metrics_used or list(metrics.keys()),
#             "predictions_file": str(predictions_file) if predictions_file else None
#         },
#         "metrics": metrics
#     }
    
#     with open(output_path, "w") as f:
#         json.dump(output, f, indent=2)
    
#     return output_path