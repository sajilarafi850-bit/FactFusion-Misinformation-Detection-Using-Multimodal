import torch
import torch.nn.functional as F
import numpy as np


class XAIEngine:
    # Token count scales with input length instead of a fixed number: short
    # captions (tweets, one-liners) get a small, selective set of top words;
    # longer disaster reports (100-200+ words) get more tokens so important
    # information isn't cut off. Bounded between MIN and MAX so it's never
    # too sparse (short text) or too cluttered (very long text).
    MIN_TOKENS = 5
    MAX_TOKENS = 15
    TOKENS_PER_WORD = 0.1  # roughly 1 token shown per 10 input words

    def _top_n_for(self, word_count):
        return max(self.MIN_TOKENS, min(self.MAX_TOKENS, round(word_count * self.TOKENS_PER_WORD)))

    # SHAP text explainer gets noticeably slower (and RoBERTa truncates
    # anyway past its max length) on very long inputs. Capping the words fed
    # into attribution keeps analysis responsive without losing accuracy —
    # the model's own prediction already only "sees" roughly this much text.
    MAX_ATTRIBUTION_WORDS = 150

    def __init__(self, text_model=None, image_model=None, tokenizer=None, background_tensor=None):
        self.text_model  = text_model
        self.image_model = image_model

        # Grad-CAM storage (used by existing generate_gradcam — unchanged)
        self.gradients   = None
        self.activations = None

        # ── SHAP for EfficientNet-B3 ──────────────────────────────────────
        # Requires background_tensor (50, 3, 224, 224) saved as assets/shap_background.pt
        # If missing, falls back to Grad-CAM automatically — no crash.
        self.image_explainer = None
        if image_model is not None and background_tensor is not None:
            try:
                import shap
                self.image_explainer = shap.GradientExplainer(image_model, background_tensor)
                print("[XAI] SHAP GradientExplainer ready for EfficientNet-B3")
            except Exception as e:
                print(f"[XAI] SHAP image init failed — Grad-CAM will be used: {e}")

        # ── SHAP for RoBERTa ─────────────────────────────────────────────
        # Requires tokenizer. If missing, falls back to heuristic automatically.
        self.text_explainer = None
        if text_model is not None and tokenizer is not None:
            try:
                import shap
                from transformers import pipeline as hf_pipeline
                pipe = hf_pipeline(
                    "text-classification",
                    model=text_model,
                    tokenizer=tokenizer,
                    return_all_scores=True,
                    device=-1  # CPU — matches Railway
                )
                self.text_explainer = shap.Explainer(pipe)
                print("[XAI] SHAP Explainer ready for RoBERTa")
            except Exception as e:
                print(f"[XAI] SHAP text init failed — heuristic fallback will be used: {e}")

    # =========================================================================
    # GRAD-CAM — your original method, completely unchanged
    # =========================================================================

    def _save_gradient(self, grad):
        self.gradients = grad

    def _save_activation(self, act):
        self.activations = act

    def generate_gradcam(self, input_tensor, target_layer):
        """
        Generates a 2D feature importance heatmap using Grad-CAM.

        Args:
            input_tensor (torch.Tensor): Preprocessed image tensor shape (1, 3, H, W)
            target_layer (torch.nn.Module): The final convolutional/feature layer of your model
        Returns:
            np.ndarray: Normalized 224x224 heatmap matrix scaled between 0.0 and 1.0
        """
        if self.image_model is None:
            raise ValueError("Image model not initialized in XAIEngine.")

        self.image_model.eval()

        # Register PyTorch forward and backward hooks
        forward_hook  = target_layer.register_forward_hook(
            lambda module, input, output: self._save_activation(output)
        )
        backward_hook = target_layer.register_full_backward_hook(
            lambda module, grad_input, grad_output: self._save_gradient(grad_output[0])
        )

        try:
            # Forward pass to fetch prediction probabilities
            predictions  = self.image_model(input_tensor)

            # Target the highest scoring class index
            target_class = predictions.argmax(dim=1).item()
            score        = predictions[0, target_class]

            # Zero out gradients and run backward pass
            self.image_model.zero_grad()
            score.backward()

            if self.gradients is None or self.activations is None:
                raise ValueError("Gradients or activations were not captured. Check the target layer.")

            # Global Average Pooling (GAP) of the captured gradients
            gradients   = self.gradients.cpu().data.numpy()
            activations = self.activations.cpu().data.numpy()

            # Calculate alpha weights: mean over height and width channels
            weights = np.mean(gradients, axis=(2, 3))[0]

            # Linear combination of activations and weights
            heatmap = np.zeros(activations.shape[2:], dtype=np.float32)
            for i, w in enumerate(weights):
                heatmap += w * activations[0, i, :, :]

            # Apply ReLU to only keep features that positively correlate with the target class
            heatmap = np.maximum(heatmap, 0)

            # Normalize to avoid dividing by zero if the heatmap is uniform
            denom = np.max(heatmap) - np.min(heatmap)
            if denom == 0:
                denom = 1e-5
            heatmap = (heatmap - np.min(heatmap)) / denom

            # Resize the heatmap to 224x224 using PyTorch bilinear interpolation
            heatmap_tensor  = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            heatmap_resized = F.interpolate(heatmap_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            heatmap         = heatmap_resized.squeeze().numpy()

            return heatmap

        finally:
            # Always remove hooks to prevent GPU memory leaks
            forward_hook.remove()
            backward_hook.remove()

    # =========================================================================
    # NEW: get_visual_explanation — SHAP primary, Grad-CAM fallback
    # Called by multimodal_service.py instead of generate_gradcam directly
    # =========================================================================

    def get_visual_explanation(self, input_tensor, target_layer, pred_class_idx=None):
        """
        Returns (heatmap_ndarray, status_str, method_str).
        Priority: SHAP → Grad-CAM → (None, UNAVAILABLE, None)
        """
        # PATH 1 — SHAP (only if background tensor was loaded at init)
        if self.image_explainer is not None:
            try:
                import shap
                # Full precision: no nsamples cap, so SHAP uses its own
                # default sampling for the most accurate attribution.
                shap_values = self.image_explainer.shap_values(input_tensor)
                # shap_values is a list[ndarray], one per class, each (1, 3, 224, 224)
                idx         = pred_class_idx if pred_class_idx is not None else 0
                attribution = shap_values[idx][0]        # (3, 224, 224)
                heatmap     = attribution.mean(axis=0)   # (224, 224)
                # Normalise to 0-1 to match Grad-CAM output range
                h_min, h_max = heatmap.min(), heatmap.max()
                if h_max - h_min > 0:
                    heatmap = (heatmap - h_min) / (h_max - h_min)
                print("[XAI] Heatmap generated via SHAP")
                return heatmap, "AVAILABLE", "shap"
            except Exception as e:
                print(f"[XAI] SHAP image failed: {e} — falling back to Grad-CAM")

        # PATH 2 — Grad-CAM (your original method, always available when model loaded)
        if self.image_model is not None:
            try:
                heatmap = self.generate_gradcam(input_tensor, target_layer)
                print("[XAI] Heatmap generated via Grad-CAM")
                return heatmap, "AVAILABLE", "grad-cam"
            except Exception as e:
                print(f"[XAI] Grad-CAM failed: {e}")
                return None, f"GRAD-CAM FAILED: {str(e)}", None

        return None, "UNAVAILABLE", None

    def _merge_subwords(self, tokens, weights, top_n, combine="sum"):
        """
        RoBERTa's BPE tokenizer splits many words into multiple subword pieces
        (e.g. "disaster" → "dis" + "aster"). Ranking those pieces individually
        by weight fragments a single important word across the top-N list,
        while short, complete, low-signal words near the start of the text
        (which stay as one token) rank higher by comparison — producing a
        list that looks arbitrary/"messy" rather than genuinely important.
        This merges continuation pieces (tokens without the leading "Ġ"
        word-start marker) back onto the previous token and combines their
        weights, so the final list is whole, readable words ranked by their
        true combined importance.
        """
        merged = []
        for t, w in zip(tokens, weights):
            is_continuation = merged and not t.startswith("Ġ") and t not in ("Ġ",)
            clean = t.replace("Ġ", "")
            if not clean.strip():
                continue
            if is_continuation:
                merged[-1]["token"] += clean
                merged[-1]["weight"] += float(w)
            else:
                merged.append({"token": clean, "weight": float(w)})

        for m in merged:
            m["weight"] = round(m["weight"], 3)

        return sorted(merged, key=lambda x: abs(x["weight"]), reverse=True)[:top_n]

    # =========================================================================
    # TEXT ATTRIBUTION — SHAP primary, original heuristic fallback
    # =========================================================================

    def extract_text_attributions(self, text, tokenizer):
        """
        Calculates feature importance scores for incoming text tokens.
        SHAP primary (real per-token signed weights).
        Falls back to original position-weighted heuristic if SHAP unavailable.

        Label mapping: {0: "Non-Informative", 1: "Informative", 2: "OOD"}
        SHAP index 1 = "Informative" — correct class to explain disaster-relevant content.
        """
        # Guard against very long inputs (100+/200+ word reports) — SHAP's
        # partition explainer slows down substantially as text grows, and
        # RoBERTa truncates past its own max length anyway, so trimming here
        # keeps response times predictable without changing what the model
        # actually "sees" and explains.
        words = text.split()
        if len(words) > self.MAX_ATTRIBUTION_WORDS:
            text = " ".join(words[:self.MAX_ATTRIBUTION_WORDS])

        top_n = self._top_n_for(len(words))

        # PATH 1 — SHAP (only if text_explainer was initialised at init)
        if self.text_explainer is not None:
            try:
                # Full precision: no max_evals cap, so SHAP scores as many
                # perturbed text variants as its partition algorithm decides
                # it needs for accurate token attributions.
                shap_values = self.text_explainer([text])
                tokens      = shap_values.data[0]
                # FIX: was [:, 0] which explained "Non-Informative" (class 0)
                # Now [:, 1] explains "Informative" (class 1) — semantically correct
                # Positive weight = token pushes toward Informative/disaster
                # Negative weight = token pushes away from Informative
                weights     = shap_values.values[0][:, 1]
                keep_mask   = [t not in ["<s>", "</s>", "<pad>", "<unk>"] for t in tokens]
                clean_tokens  = [t for t, k in zip(tokens, keep_mask) if k]
                clean_weights = [w for w, k in zip(weights, keep_mask) if k]
                result = self._merge_subwords(clean_tokens, clean_weights, top_n)
                print(f"[XAI] Text attributions generated via SHAP (top_{top_n})")
                return result
            except Exception as e:
                print(f"[XAI] Text SHAP failed: {e} — falling back to heuristic")

        # PATH 2 — your original heuristic (completely unchanged)
        if self.text_model is None:
            return [{"token": word, "weight": 0.5} for word in text.split()[:top_n]]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

        # Ensure inputs are moved to the same device as the model
        device = next(self.text_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

        # Use gradient magnitude per token as importance proxy
        # Much better signal than position-decay dummy weights
        embed_layer = self.text_model.roberta.embeddings.word_embeddings
        embeds = embed_layer(inputs["input_ids"]).detach().requires_grad_(True)
        out2 = self.text_model(inputs_embeds=embeds,
                               attention_mask=inputs.get("attention_mask"))
        score = out2.logits[0, out2.logits.argmax(dim=1).item()]
        score.backward()

        grad_mag = embeds.grad[0].norm(dim=-1).detach().cpu().numpy()

        keep_mask     = [token not in ['<s>', '</s>', '<pad>', '<unk>'] for token in tokens]
        clean_tokens  = [t for t, k in zip(tokens, keep_mask) if k]
        clean_weights = [grad_mag[i] for i, k in enumerate(keep_mask) if k]

        # Normalise raw gradient magnitudes to 0-1 BEFORE merging, so the
        # frontend gauge scale stays consistent regardless of how many
        # subword pieces end up combined into a given word.
        if len(clean_weights):
            max_w = max(clean_weights) or 1.0
            clean_weights = [w / max_w for w in clean_weights]

        return self._merge_subwords(clean_tokens, clean_weights, top_n)