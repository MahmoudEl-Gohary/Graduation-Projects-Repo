import json
from pathlib import Path
from datetime import datetime


def save_predictions(
    predictions: list[dict],
    output_dir: str | Path,
    model_name: str = "unknown",
    filename: str | None = None
) -> Path:
    """
    Save predictions to JSON file.
    
    Args:
        predictions: List of dicts with keys: filename, ground_truth, prediction
        output_dir: Directory to save results
        model_name: Name of the model used
        filename: Custom filename. If None, uses timestamp.
        
    Returns:
        Path to saved file
    """
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


def save_results(
    metrics: dict,
    output_dir: str | Path,
    model_name: str = "unknown",
    metrics_used: list[str] | None = None,
    filename: str | None = None
) -> Path:
    """
    Save evaluation metrics to JSON file.
    
    Args:
        metrics: Dictionary of metric results from RadiologyEvaluator
        output_dir: Directory to save results
        model_name: Name of the model used
        metrics_used: List of metric names used
        filename: Custom filename. If None, uses timestamp.
        
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = filename or f"evaluation_metrics_{timestamp}.json"
    output_path = output_dir / filename
    
    output = {
        "metadata": {
            "timestamp": timestamp,
            "model": model_name,
            "metrics_used": metrics_used or list(metrics.keys())
        },
        "metrics": metrics
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    return output_path