import os
import ssl
import nltk
import stanza

# --- CONFIGURATION ---

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(BASE_DIR, ".resources")

NLTK_DIR = os.path.join(RESOURCES_DIR, "nltk_data")
STANZA_DIR = os.path.join(RESOURCES_DIR, "stanza_resources")

# Create folders if they don't exist
os.makedirs(NLTK_DIR, exist_ok=True)
os.makedirs(STANZA_DIR, exist_ok=True)

print(f"Downloading resources to: {RESOURCES_DIR}")

# --- FIX SSL ERROR (Common on some servers) ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- 1. DOWNLOAD NLTK ---
print("... Downloading NLTK data (punkt)...")
nltk.download('punkt', download_dir=NLTK_DIR)
# nltk.download('stopwords', download_dir=NLTK_DIR)

# --- 2. DOWNLOAD STANZA ---
print("... Downloading Stanza models (en)...")
stanza.download('en', model_dir=STANZA_DIR, verbose=True)

print("\nSetup Complete! Resources are ready.")