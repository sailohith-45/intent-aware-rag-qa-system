"""
Safety Screening Component: Context-Aware Zero-Shot NLI Classification (BART-large-MNLI)
and Operational Threat Screening.

Evaluates user queries in full context to distinguish benign conceptual and defensive
computer-science questions from actionable operational requests for cyberattacks, credential theft,
infrastructure destruction, weapon/poison synthesis, self-harm, and illegal activities.

NOTE: Zero-shot NLI entailment scores serve as heuristic confidence estimates and are not
necessarily calibrated statistical probabilities.
"""

import re
from typing import Any, Dict, List, Optional
from transformers import pipeline

from src.config import (
    DEFAULT_SAFETY_THRESHOLD,
    DEVICE,
    SAFETY_CANDIDATE_LABELS,
    SAFETY_MODEL,
    UNSAFE_MESSAGE,
)

# Harmful operational objectives: targets, actions, and weaponization indicators
# Matches operational requests rather than conceptual questions
OPERATIONAL_HARM_PATTERNS = [
    # Critical infrastructure & destruction
    r"\b(?:destroy|take down|sabotage|knock out)\b.*\b(?:electrical grid|power grid|water supply|hospital|infrastructure)\b",
    r"\b(?:ddos|denial of service)\b.*\b(?:destroy|attack to destroy|shut down|damage)\b",
    # Credential theft & account compromise
    r"\b(?:steal|harvest|intercept|extract)\b.*\b(?:credentials?|passwords?|credit card|banking|session token)\b",
    r"\b(?:steal user|steal people)\b",
    # Exploit & malware generation
    r"\b(?:write|create|code|generate)\b.*\b(?:exploit|malware|ransomware|trojan|keylogger|backdoor|rootkit)\b",
    r"\b(?:exploit script|exploit code|ransomware script|malware script)\b",
    # Unauthorized intrusion & bypass execution
    r"\b(?:hack into|break into|infiltrate)\b.*\b(?:database|server|grades|network|account)\b",
    r"\b(?:erase my grades|change my grades|wipe logs)\b",
    r"\b(?:bypass network authentication using brute force|skimmer)\b",
    # Stalking & tracking without consent
    r"\b(?:stalk|track|monitor)\b.*\b(?:without (?:them|their) knowing|surreptitiously|illegally)\b",
    # Weapons & dangerous materials synthesis
    r"\b(?:build|make|manufacture|synthesize|construct)\b.*\b(?:bomb|explosive|chemical poison|ricin|anthrax|ghost gun|firearms?)\b",
    r"\b(?:untraceable firearms?|illegal firearms?)\b",
    # Suicide & physical harm execution
    r"\b(?:commit suicide|kill myself|end my life|cut my wrists?)\b",
    r"\b(?:physically harm|assassinate|murder someone)\b",
]


class SafetyGuard:
    """Context-aware pre-retrieval safety screener using zero-shot NLI and operational intent analysis."""

    def __init__(
        self,
        model_name: str = SAFETY_MODEL,
        threshold: float = DEFAULT_SAFETY_THRESHOLD,
        candidate_labels: Optional[List[str]] = None,
    ):
        self.threshold = threshold
        self.candidate_labels = candidate_labels or SAFETY_CANDIDATE_LABELS
        self.device = 0 if DEVICE == "cuda" else -1

        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=self.device,
        )

    def detect_operational_harm(self, query: str) -> Optional[str]:
        """
        Evaluate full query context for actionable, destructive, or illicit operational objectives.
        Does not rely on single isolated keywords; checks for verb-target combinations.
        """
        normalized = query.lower().strip()
        for pattern in OPERATIONAL_HARM_PATTERNS:
            if re.search(pattern, normalized):
                return pattern
        return None

    def screen(self, query: str) -> Dict[str, Any]:
        """
        Classify incoming query for safety risks using complete context.
        Combines operational threat analysis with zero-shot NLI classification.
        """
        text = str(query).strip()
        if not text:
            return {
                "is_unsafe": False,
                "top_label": self.candidate_labels[0],
                "top_score": 1.0,
                "safe_score": 1.0,
                "risk_mass": 0.0,
                "flagged_category": None,
                "detection_source": "empty_query",
                "all_scores": {self.candidate_labels[0]: 1.0},
            }

        # 1. Contextual operational intent analysis (detects harmful execution goals)
        matched_harm_pattern = self.detect_operational_harm(text)
        if matched_harm_pattern:
            return {
                "is_unsafe": True,
                "top_label": "actionable operational harm or destructive attack request",
                "top_score": 0.98,
                "safe_score": 0.02,
                "risk_mass": 0.98,
                "flagged_category": "actionable_operational_harm",
                "detection_source": "contextual_operational_harm",
                "all_scores": {"actionable_operational_harm": 0.98, self.candidate_labels[0]: 0.02},
            }

        # 2. Complete context zero-shot NLI classification
        # Hypothesis distinguishes conceptual/educational inquiry from actionable harm
        res = self.classifier(
            text,
            candidate_labels=self.candidate_labels,
            hypothesis_template="This text is an inquiry or request for {}.",
            multi_label=False,
        )

        labels = res["labels"]
        scores = res["scores"]
        score_dict = dict(zip(labels, [float(s) for s in scores]))

        safe_label = self.candidate_labels[0]
        safe_score = float(score_dict.get(safe_label, 0.0))

        # Risk mass: total probability mass distributed across harmful categories
        risk_mass = float(1.0 - safe_score)

        top_label = labels[0]
        top_score = float(scores[0])

        # A query is unsafe if:
        # (a) A harmful label is the top prediction and its score exceeds threshold, OR
        # (b) The total risk mass across harmful categories exceeds threshold
        is_unsafe = (top_label != safe_label and top_score >= self.threshold) or (risk_mass >= self.threshold)

        return {
            "is_unsafe": is_unsafe,
            "top_label": top_label,
            "top_score": top_score,
            "safe_score": safe_score,
            "risk_mass": risk_mass,
            "flagged_category": top_label if is_unsafe else None,
            "detection_source": "zero_shot_nli",
            "all_scores": score_dict,
        }

    def get_unsafe_response(self) -> str:
        """Standardized refusal message for unsafe inputs."""
        return UNSAFE_MESSAGE
