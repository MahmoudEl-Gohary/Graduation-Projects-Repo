import sys
import json
import torch
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from transformers import AutoModelForImageTextToText, AutoProcessor

from IU_dataset_loader import IndianaDataset

# --- 1. SETUP PATHS ---
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = REPO_ROOT.parent / "checkpoints" / "nvidia-reason-3b"
DATA_PATH = REPO_ROOT.parent / "data" / "indiana_university"
PREDICTIONS_DIR = REPO_ROOT / "results" / "predictions"

def load_model(model_path: Path):
    """Load model and processor."""
    print(f"Loading model from: {model_path}")
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    print("Model loaded successfully!")
    return model, processor


def generate_report(model, processor, image) -> str:
    """Generate a report for a single image."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": """
                    You are a highly skilled radiology assistant with expertise in diagnostic reasoning and differential diagnosis. Your
                    role is to assist in the interpretation of radiology reports by integrating imaging findings with the patient’s clinical
                    presentation, symptoms, and medical history. Using your in-depth knowledge of radiographic patterns, imaging features
                    of various diseases, and their underlying pathophysiology, your task is to explore potential diagnoses and causal
                    relationships. When analyzing the radiology report, consider the clinical indications for why the imaging was ordered
                    and provide a concise, evidence-based analysis that connects the radiologic findings to the primary clinical concern. Your
                    response should focus strictly on the imaging findings and their relevance to the clinical indication. If abnormalities
                    are present, prioritize the most likely diagnosis and briefly address key differentials. If findings are normal, confirm
                    whether the clinical indication is ruled out based on the imaging. Keep your reasoning brief, clear, and directly relevant
                    to the primary diagnosis. Do not include any suggestion. Output only in the format given in the examples below:
                    Radiology Report: INDICATION: $ $-year-old man with a history of end-stage renal disease, status post kidney transplant,
                    presents to the clinic with increasing fatigue and dyspnea on exertion and chest congestion. Rule out pulmonary edema.
                    COMPARISON: Preop chest radiograph, . PA AND LATERAL CHEST RADIOGRAPH: The cardiac, mediastinal, and hilar contours are
                    unchanged. Pleural thickening within both lung bases is unchanged from the prior examinations. Opacification in the right
                    lower lung medial base is consistent with right lower lobe pneumonia. Findings were discussed with Dr. at 16:31 on
                    via telephone.
                    REASONING: The opacification in the right lower lung medial base suggests right lower lobe pneumonia, which is consistent
                    with the increased density area seen in the radiograph. Additionally, pleural thickening may indicate pleurisy,
                    contributing to chest congestion and dyspnea on exertion. The findings are located in the right lower lobe and pleural
                    regions, correlating with the noted symptoms and radiographic observations.
                 """
                }
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

def save_predictions(predictions: list[dict], output_dir: Path, model_name: str) -> Path:
    """Save predictions to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"predictions_{timestamp}.json"
    
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
    
    print(f"Predictions saved to: {output_path}")
    return output_path

def run_inference(num_samples: int | None = None):
    """Run inference and save predictions."""
    # Load model
    model, processor = load_model(MODEL_PATH)
    
    # Load dataset
    print(f"Loading dataset from: {DATA_PATH}")
    if not DATA_PATH.exists():
        print(f"Error: Data path not found at {DATA_PATH}")
        print("Run 'python scripts/download_data.py' first.")
        exit()
    
    dataset = IndianaDataset(DATA_PATH)
    print(f"Dataset Size: {len(dataset)} samples")
    
    # Determine number of samples
    total = len(dataset) if num_samples is None else min(num_samples, len(dataset))
    
    # Generate predictions
    print(f"\nGenerating predictions for {total} samples...")
    predictions = []
    
    for idx in tqdm(range(total), desc="Inference"):
        sample = dataset[idx]
        pred = generate_report(model, processor, sample["image"])
        
        predictions.append({
            "index": idx,
            "filename": sample["filename"],
            "ground_truth": sample["report"],
            "prediction": pred
        })
    
    # Save predictions
    output_path = save_predictions(predictions, PREDICTIONS_DIR, model_name="nvidia-reason-3b")
    
    print("\n" + "=" * 50)
    print("INFERENCE COMPLETE")
    print(f"Results saved to: {output_path}")
    print("=" * 50)
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference on radiology images")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process. If not specified, processes all."
    )
    args = parser.parse_args()
    
    run_inference(num_samples=args.num_samples)