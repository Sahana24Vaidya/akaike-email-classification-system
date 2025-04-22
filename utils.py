import re
from typing import List, Dict, Tuple

# Regex patterns for all required PII types
PII_PATTERNS = {
    "expiry_no": r"(?<!\d)\b(0[1-9]|1[0-2])(?:\/|)(\d{2})\b(?!\d)",        # 12/25
    "phone_number": r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "aadhar_num": r"\b\d{4}[ -]?\d{4}[ -]?\d{4}(?![ -]?\d)",
    "cvv_no": r"(?i)(CVV|CVC)[: ]*(\d{3,4})\b",  # Two capture groups
    "full_name": r"My name is ([A-Z][a-z]+ [A-Z][a-z]+)"
}

class Entity:
    def __init__(self, start: int, end: int, label: str, text: str):
        self.start = start
        self.end = end
        self.label = label
        self.text = text


def mask_pii(text: str) -> Tuple[str, List[Dict]]:
    """Function to mask PII in a given text."""
    entities = []
    masked_parts = []
    last_end = 0  # Tracks where we are in the original text
    
    # Step 1: Find ALL PII matches first
    all_matches = []
    for pii_type, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            all_matches.append({
                "start": match.start(),
                "end": match.end(),
                "type": pii_type,
                "text": match.group()
            })
    
    # Prevent CVV matches inside phone numbers
    all_matches = [m for m in all_matches if not (
        m["type"] == "cvv_no" and 
        any(phone["start"] <= m["start"] <= phone["end"] 
            for phone in all_matches 
            if phone["type"] == "phone_number")
    )]
    
    # Step 2: Sort matches by their position in the text
    all_matches.sort(key=lambda x: x["start"])
    
    # Step 3: Build the masked string
    for match in all_matches:
        masked_parts.append(text[last_end:match["start"]])
        
        # Special handling for CVV
        if match["type"] == "cvv_no":
            full_match = match["text"]  # "CVV: 123"
            cvv_digits = re.search(r"\d{3,4}", full_match).group()  # Extract just "123"
            masked_parts.append(f"[{match['type']}]")
            entities.append(Entity(
             start=match["start"],
            end=match["end"],
            label=match["type"],
            text=cvv_digits
        ))

        else:
            masked_parts.append(f"[{match['type']}]")
            entities.append(Entity(
            start=match["start"],
            end=match["end"],
            label=match["type"],
            text=match["text"]
        ))
        last_end = match["end"]
        
    # Add remaining text after last match
    masked_parts.append(text[last_end:])
    
    return "".join(masked_parts), entities


def clean_text(text: str) -> str:
    """
    Lightweight text cleaning for classification models.
    Preserves negation words and multilingual content.
    """
    # Step 1: Basic cleaning
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'\d+', '', text)                      # Remove digits
    
    # Step 2: Keep essential punctuation (!, ?)
    text = re.sub(r'[^\w\s!?]', '', text)
    
    # Step 3: Remove extra spaces
    return ' '.join(text.split())

