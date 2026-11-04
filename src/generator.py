"""
Answer Generation Component: Grounded Sequence-to-Sequence Generation using FLAN-T5.
Adapts prompt guidance based on detected intent, strictly enforces context grounding,
and handles context token truncation cleanly.
"""

from typing import Any, Dict, List, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import DEVICE, GEN_MODEL, INTENT_STYLE


class Generator:
    """Context-grounded, intent-aware answer generator using FLAN-T5."""

    def __init__(
        self,
        model_name: str = GEN_MODEL,
        max_context_tokens: int = 350,
        max_new_tokens: int = 120,
    ):
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.device = DEVICE

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_effective_context(
        self,
        passages: List[Dict[str, Any]],
        num_passages: int = 2,
    ) -> str:
        """Combine and truncate passages to ensure consistent context for both generator and checker."""
        if not passages:
            return ""

        selected_texts = [p["text"] for p in passages[:num_passages] if "text" in p]
        raw_context = "\n\n".join(selected_texts)

        # Tokenize and strictly truncate context only, preserving room for query & instructions
        input_ids = self.tokenizer(
            raw_context,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_context_tokens,
        )["input_ids"]

        return self.tokenizer.decode(input_ids, skip_special_tokens=True).strip()

    def construct_prompt(
        self,
        question: str,
        intent: str,
        context: str,
    ) -> str:
        """Build intent-conditioned instruction prompt for FLAN-T5."""
        intent_instruction = INTENT_STYLE.get(intent, "Answer the question directly and concisely.")

        prompt = (
            f"You are a computer science technical assistant. "
            f"Answer the user's question using ONLY the provided context. "
            f"{intent_instruction} "
            f"If the context does not provide sufficient facts to answer, "
            f"state that the context does not contain enough information.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        return prompt

    @torch.no_grad()
    def generate(
        self,
        question: str,
        intent: str,
        passages: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate grounded answer conditioned on question, intent, and retrieved passages."""
        if not passages:
            return "I could not find relevant information in the knowledge base."

        context = self.get_effective_context(passages)
        if not context:
            return "Retrieved context passages were empty or unreadable."

        prompt = self.construct_prompt(question, intent, context)
        tokens_to_generate = max_new_tokens or self.max_new_tokens

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=tokens_to_generate,
            num_beams=4,
            early_stopping=True,
            do_sample=False,
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return answer
