"""
Probe script for text_final_model — prints raw softmax probabilities
(not just the final label) for a spread of test inputs, so you can see
whether the model is keying on keyword presence vs. actual informativeness.

Usage:
    python probe_text_model.py

Assumes this script sits in the same directory as model_service.py,
or adjust MODEL_DIR below to point at your models/text_final_model folder.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- adjust this path if needed ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "text_final_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model from: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
model.eval()

id2label = model.config.id2label
print(f"id2label from model config: {id2label}\n")

# Test cases ordered from "bare keyword, zero content" to "dense, specific, actionable"
test_cases = [
    ("Bare keyword only", "flood"),
    ("Minimal sentence", "There was a flood."),
    ("Vague + hedge", "I heard about the flood but don't know details."),
    ("Vague + emotional", "Everyone's talking about the flood today. Hope it's not too bad."),
    ("Emotional, longer", (
        "People are talking about the flood and everyone seems worried about "
        "what's happening. It's honestly such a disaster, my heart goes out to "
        "everyone affected. Stay safe everyone, sending love and strength."
    )),
    ("Specific + actionable", (
        "The Route 9 bridge collapsed and 40 residents were evacuated to the "
        "county shelter this morning after floodwaters rose 3 feet overnight."
    )),
    ("Specific, no hedge, short", "Flood evacuation ordered for Ward 3 residents effective 6pm."),
    ("Off-topic control (should be OOD upstream, but testing model alone)", "I binge-watched a show last night."),
]

print(f"{'Case':<45} {'Pred Label':<18} {'P(Non-Informative)':<20} {'P(Informative)':<15}")
print("-" * 100)

for name, text in test_cases:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    pred_idx = probs.argmax().item()
    pred_label = id2label[str(pred_idx)] if str(pred_idx) in id2label else id2label[pred_idx]

    p_non_informative = probs[0].item()
    p_informative = probs[1].item()

    print(f"{name:<45} {pred_label:<18} {p_non_informative:<20.4f} {p_informative:<15.4f}")

print("\nWhat to look for:")
print("- If 'Bare keyword only' and 'Specific + actionable' both show similarly")
print("  HIGH confidence for Informative, the model is likely keying on keyword")
print("  presence rather than actual content/specificity (a training-data or")
print("  class-imbalance issue).")
print("- If confidence scales down for vaguer/shorter inputs and up for dense,")
print("  specific ones, the model is behaving as intended and the earlier")
print("  results may have been edge cases near the decision boundary.")