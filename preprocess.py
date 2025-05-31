# utils/preprocess.py

def clean_text(text):
    """
    Perform basic text cleaning such as lowercasing and removing extra whitespace.
    """
    return " ".join(text.strip().lower().split())
