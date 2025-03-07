import json
import hashlib

def compute_file_hash(file):
    """Generate SHA-256 hash for file content."""
    hasher = hashlib.sha256()
    while chunk := file.read(8192):
        hasher.update(chunk)
    return hasher.hexdigest()

def load_processed_files():
    """Load hashes of already processed files."""
    try:
        with open("processed_files.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_processed_files(hashes):
    """Save hashes of processed files."""
    with open("processed_files.json", "w") as f:
        json.dump(hashes, f)