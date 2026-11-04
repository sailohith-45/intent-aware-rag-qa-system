"""
Faithfulness Verification Component: Natural Language Inference (BART-large-MNLI).
Verifies whether the generated answer is strictly supported by the retrieved context.
Fixes context mismatch, handles short answer fragments, and prevents hallucination.
"""

from typing import Any, Dict, Optional, Union
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import DEFAULT_ENTAILMENT_THRESHOLD, DEVICE, SAFETY_MODEL


class FaithfulnessChecker:
    """NLI entailment verifier ensuring generated answers are grounded in retrieved context."""

    def __init__(
        self,
        model_name: str = SAFETY_MODEL,
        threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
        tokenizer: Optional[AutoTokenizer] = None,
        model: Optional[AutoModelForSequenceClassification] = None,
    ):
        self.threshold = threshold
        self.device = DEVICE

        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model.eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            self.model.eval()

        # Identify label indices from model configuration
        label_mapping = {label.lower(): idx for idx, label in self.model.config.id2label.items()}
        self.entailment_idx = label_mapping.get("entailment", 2)
        self.contradiction_idx = label_mapping.get("contradiction", 0)
        self.neutral_idx = label_mapping.get("neutral", 1)

    def formulate_hypothesis(self, question: str, answer: str) -> str:
        """Convert fragments or short answers into complete propositions for reliable NLI evaluation."""
        words = answer.strip().split()
        if len(words) <= 4:
            # Fragment answer like "TCP", "O(n log n)", "Deadlock"
            return f"The answer to the question '{question.strip()}' is {answer.strip()}."
        return answer.strip()

    @torch.no_grad()
    def calculate_support(
        self,
        context: str,
        answer: str,
        question: Optional[str] = None,
    ) -> Dict[str, Union[float, bool]]:
        """Calculate entailment probability of the answer given the effective context."""
        cleaned_context = context.strip()
        cleaned_answer = answer.strip()

        if not cleaned_context or not cleaned_answer:
            return {
                "is_faithful": False,
                "entailment_prob": 0.0,
                "neutral_prob": 0.0,
                "contradiction_prob": 1.0,
            }

        # Build candidate hypothesis (handling short answer fragments)
        if question:
            hypothesis = self.formulate_hypothesis(question, cleaned_answer)
        else:
            hypothesis = cleaned_answer

        inputs = self.tokenizer(
            cleaned_context,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

        entailment = float(probabilities[self.entailment_idx].item())
        neutral = float(probabilities[self.neutral_idx].item())
        contradiction = float(probabilities[self.contradiction_idx].item())

        is_faithful = entailment >= self.threshold

        return {
            "is_faithful": is_faithful,
            "entailment_prob": round(entailment, 4),
            "neutral_prob": round(neutral, 4),
            "contradiction_prob": round(contradiction, 4),
        }
