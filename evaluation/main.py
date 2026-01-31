import os
import sys

# --- CONFIG: Set paths for helper libraries ---
# Get the absolute path to the evaluation folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(BASE_DIR, ".resources")

# Tell NLTK and Stanza where to look
os.environ["NLTK_DATA"] = os.path.join(RESOURCES_DIR, "nltk_data")
os.environ["STANZA_RESOURCES_DIR"] = os.path.join(RESOURCES_DIR, "stanza_resources")

from RadEval import RadEval
import json

refs = [
    "Mild cardiomegaly with small bilateral pleural effusions and basilar atelectasis.",
    "No pleural effusions or pneumothoraces.",
]
hyps = [
    "Mildly enlarged cardiac silhouette with small pleural effusions and dependent bibasilar atelectasis.",
    "No pleural effusions or pneumothoraces.",
]

evaluator = RadEval(
    # do_green=True,
    do_radgraph=True,
    do_bleu=True
)

results = evaluator(refs=refs, hyps=hyps)
print(json.dumps(results, indent=2))