import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import re

class ModelLoader:
    def __init__(self):
        print("Loading fine-tuned Text Model (RoBERTa)...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- OOD domain gate ---
        # The fine-tuned model was trained as a STRICT BINARY classifier
        # (Informative vs Non-Informative) on disaster-domain text only — it has
        # no concept of "out of domain" and will always force any input, even
        # something like "I binge-watched a show last night", into one of those
        # two classes. OOD is therefore detected with a lightweight rule-based
        # domain check that runs BEFORE the model is asked anything, rather than
        # by a model-trained class. If the text shows no disaster-domain signal
        # at all, we short-circuit to OOD; otherwise we let the model classify
        # Informative vs Non-Informative as it was trained to do.
        #
        # Exact port of CRISIS_KEYWORDS from text_analysis.ipynb (CELL 5) —
        # keep this list and the substring-matching behavior in
        # _has_disaster_signal identical to the notebook the model was
        # validated against.
        self.disaster_keywords = [
            # Earthquake
            'earthquake', 'quake', 'tremor', 'aftershock', 'shake', 'shaking', 'shook',
            'rumble', 'rumbling', 'vibration', 'vibrate', 'tremble', 'trembling',
            'seismic', 'epicenter', 'richter', 'magnitude',

            # Flood
            'flood', 'flooding', 'flooded', 'flash flood', 'water level', 'rising water',
            'overflow', 'inundation', 'submerged', 'swamped', 'rain', 'heavy rain',
            'monsoon', 'downpour', 'torrential rain', 'storm surge',

            # Storm
            'storm', 'cyclone', 'typhoon', 'hurricane', 'tornado', 'twister',
            'windstorm', 'thunderstorm', 'blizzard', 'snowstorm', 'hailstorm',

            # Land movement
            'landslide', 'avalanche', 'mudslide', 'rockslide', 'sinkhole',
            'collapse', 'cave in',

            # Fire
            'fire', 'wildfire', 'forest fire', 'bushfire', 'blaze', 'inferno',
            'burning', 'burn', 'smoke', 'flame', 'arson',
            'explosion', 'blast', 'detonation',

            # Rescue & Emergency
            'rescue', 'rescued', 'rescuing', 'evacuate', 'evacuation', 'evacuated',
            'shelter', 'emergency', 'urgent', 'critical', 'disaster', 'catastrophe',
            'calamity', 'tragedy', 'crisis',
            'ambulance', 'hospital', 'medical',
            'relief', 'aid', 'assistance', 'help',

            # Casualties
            'casualty', 'casualties', 'dead', 'death', 'died', 'deceased',
            'fatal', 'fatalities', 'injured', 'injury', 'wounded', 'victim',
            'survivor', 'survived', 'trapped', 'missing', 'lost',
            'bodies', 'body', 'buried', 'crushed',

            # Infrastructure
            'collapsed', 'collapse', 'destroyed', 'damage', 'shattered',
            'rubble', 'debris', 'ruins', 'wreckage', 'demolished',
            'building collapse', 'bridge collapse',

            # Security
            'terrorist', 'terrorism', 'attack', 'shooting', 'hostage',
            'bomb', 'bombing', 'explosive', 'gunfire',
            'police', 'security', 'lockdown', 'curfew',

            # Government & Official
            'government', 'ndma', 'army', 'military', 'relief camp',
            'warning', 'alert', 'advisory', 'notification',

            # General
            'danger', 'threat', 'risk', 'hazard', 'emergency services',
            'death toll', 'injured count', 'missing persons',
        ]
        # Kept for backward-compat fallback use in multimodal_service.py's
        # keyword-heuristic XAI path (not part of the notebook's OOD gate).
        self.uncertain_words = [
            "maybe", "rumor", "rumour", "unconfirmed", "allegedly", "claims",
            "reportedly", "unverified",
        ]

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'models', 'Text_Final_Model')
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(self.device)
            self.model.eval()
            print("Successfully loaded local RoBERTa text model.")
        except Exception as e:
            print(f"Failed to load local text model: {e}")
            self.tokenizer = None
            self.model = None

    def _has_disaster_signal(self, text):
        """Domain gate — exact port of text_analysis.ipynb's test_my_text()
        keyword check: `matched_keywords = [w for w in CRISIS_KEYWORDS if w in
        text_lower]`. This is a plain substring match (NOT whole-word), same
        as the notebook the model was validated against.
        """
        text_lower = text.lower()
        matched_keywords = [w for w in self.disaster_keywords if w in text_lower]
        return len(matched_keywords) > 0

    def predict(self, text):
        if not text:
            return None

        # --- OOD gate: runs before the model, since the model itself was
        # trained as binary Informative/Non-Informative only and cannot
        # represent "unrelated to disasters" on its own. ---
        if not self._has_disaster_signal(text):
            return {
                "label":      "OOD",
                "confidence": 0.0,  # exact match to notebooks (text_analysis.ipynb,
                                    # Multimodal_Fusion__4_.ipynb) — rule-based gate,
                                    # reported as 0.0 confidence, not a model probability
                "xai": {
                    "explanation":  "The text is unrelated to any crisis or disaster domain.",
                    "text_weights": ["ANALYSIS-COMPLETE"]
                }
            }

        if self.model and self.tokenizer:
            # Pure model inference — no keyword overrides whatsoever
            inputs = self.tokenizer(
                text, return_tensors='pt',
                truncation=True, max_length=128, padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits  = outputs.logits
                # Temperature scaling (T=1.3) — exact match to
                # text_analysis.ipynb's test_my_text(): reduces
                # overconfidence in the reported probability without
                # changing the argmax prediction itself.
                probs   = torch.softmax(logits / 1.3, dim=1)[0]

            # Model is a strict binary classifier: 0=Non-Informative, 1=Informative.
            # (Confirmed by class weights in training_summary.json — class 0 has
            # higher weight (minority/Non-Informative), class 1 has lower weight
            # (majority/Informative), matching the training dataset's binary_label
            # mapping of informative=1, not_informative=0.)
            pred_idx         = probs.argmax().item()
            final_confidence = probs[pred_idx].item()

            label_map = {0: "Non-Informative", 1: "Informative"}
            final_label = label_map.get(pred_idx, "Non-Informative")

            explanation = {
                "Informative":     "The text contains useful, specific information about a disaster event.",
                "Non-Informative": "The text provides no actionable or useful disaster information.",
            }[final_label]

        else:
            # Model not loaded — genuinely unknown, NOT the same as OOD.
            # Use None (not 0) so this can't be misread as "0% confidence
            # Non-Informative" — we have no real signal at all here.
            final_label      = "Non-Informative"
            final_confidence = None
            explanation      = "Model loading failed, using fallback."

        return {
            "label":      final_label,
            "confidence": round(float(final_confidence), 3) if final_confidence is not None else None,
            "xai": {
                "explanation":  explanation,
                "text_weights": ["ANALYSIS-COMPLETE"]
            }
        }