simport torch
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor

from IU_dataset_loader import *


# --- 1. SETUP PATHS ---
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = REPO_ROOT.parent / "checkpoints" / "nvidia-reason-3b"
DATA_PATH = REPO_ROOT.parent / "data" / "indiana_university"

# print(REPO_ROOT)
# print(MODEL_PATH)
# print(DATA_PATH)

# --- 2. LOAD MODEL (Your Code) ---
print(f"Loading model from: {MODEL_PATH}")
processor = AutoProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    local_files_only=True
)
print("Model loaded successfully!")

# --- 3. LOAD DATASET ---
print(f"Loading dataset from: {DATA_PATH}")
if not DATA_PATH.exists():
    print(f"Error: Data path not found at {DATA_PATH}")
    print("Run 'python scripts/download_data.py' first.")
    exit()

dataset = IndianaDataset(DATA_PATH)
print(f"Dataset Size: {len(dataset)} samples")

# --- 4. SELECT A SAMPLE ---
sample_idx = 0
sample = dataset[sample_idx]
image = sample["image"]
ground_truth = sample["report"]
filename = sample["filename"]

print(f"\Processing Image: {filename}")
print(f"Ground Truth Report:\n{ground_truth}")

# --- 5. PREPARE INPUT FOR NV-REASON ---
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe the findings in this chest X-ray."}
        ]
    }
]

# Apply the chat template
text = processor.apply_chat_template(messages, add_generation_prompt=True)

# Turn into Tensors
inputs = processor(text=[text], images=[image], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# --- 6. GENERATE REPORT ---
print("\nGenerating...")
with torch.no_grad():
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=1024,
        do_sample=False
    )

# --- 7. DECODE OUTPUT ---
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)[0]

print("-" * 40)
print("MODEL PREDICTION:")
print("-" * 40)
print(output_text)
print("-" * 40)