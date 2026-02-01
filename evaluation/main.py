from pathlib import Path
from src import RadiologyEvaluator, save_results

# Example usage
if __name__ == "__main__":
    refs = [
        "Mild cardiomegaly with small bilateral pleural effusions and basilar atelectasis.",
        "No pleural effusions or pneumothoraces.",
    ]
    hyps = [
        "Mildly enlarged cardiac silhouette with small pleural effusions and dependent bibasilar atelectasis.",
        "No pleural effusions or pneumothoraces.",
    ]

    evaluator = RadiologyEvaluator(metrics=["radcliq", "green", "semb"])
    results = evaluator(references=refs, predictions=hyps)
    
    print("Results:", results)
    
    RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

    save_results(
        metrics=results,
        output_dir=RESULTS_DIR,
        model_name="example_model"
    )
