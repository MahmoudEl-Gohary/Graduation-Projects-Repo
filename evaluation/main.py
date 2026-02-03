from pathlib import Path
from src import RadiologyEvaluator, save_results, load_predictions, get_latest_predictions

# --- PATHS ---
REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = REPO_ROOT / "results" / "predictions"
METRICS_DIR = REPO_ROOT / "results" / "metrics"


def run_evaluation(
    predictions_file: Path | None = None,
    metrics: list[str] | None = None,
):
    """
    Run evaluation on predictions file.
    
    Args:
        predictions_file: Path to predictions JSON. If None, uses latest.
        metrics: List of metrics to compute. If None, computes all.
    """
    # Get predictions file
    if predictions_file is None:
        predictions_file = get_latest_predictions(PREDICTIONS_DIR)
        print(f"Using latest predictions: {predictions_file}")
    
    # Load predictions
    data = load_predictions(predictions_file)
    print(f"Loaded {data['metadata']['num_samples']} samples from {data['metadata']['model']}")
    
    # Extract references and hypotheses
    refs = [p["ground_truth"] for p in data["predictions"]]
    hyps = [p["prediction"] for p in data["predictions"]]
    
    # Initialize evaluator
    print(f"\nInitializing evaluator with metrics: {metrics or 'all'}")
    evaluator = RadiologyEvaluator(metrics=metrics)
    
    # Run evaluation
    print("Computing metrics...")
    results = evaluator(references=refs, predictions=hyps)
    
    # Save results
    metrics_path = save_results(
        metrics=results,
        output_dir=METRICS_DIR,
        model_name=data["metadata"]["model"],
        metrics_used=metrics or evaluator.AVAILABLE_METRICS,
        predictions_file=str(predictions_file)
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for metric, value in results.items():
        print(f"  {metric}: {value}")
    print("=" * 50)
    print(f"\nMetrics saved to: {metrics_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate radiology report predictions")
    parser.add_argument(
        "--predictions_file",
        type=str,
        default=None,
        help="Path to predictions JSON. If not specified, uses latest."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics to compute: radcliq, bleu, bertscore, semb, radgraph, ratescore, green"
    )
    
    args = parser.parse_args()
    
    predictions_path = Path(args.predictions_file) if args.predictions_file else None
    
    run_evaluation(
        predictions_file=predictions_path,
        metrics=args.metrics,
    )