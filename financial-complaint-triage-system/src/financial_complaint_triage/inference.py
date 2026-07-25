from __future__ import annotations

import html
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


@dataclass(frozen=True)
class PredictionCandidate:
    """A candidate product category returned by the model."""

    product: str
    confidence_score: float


@dataclass(frozen=True)
class TriagePrediction:
    """Privacy-safe result returned by the triage service."""

    predicted_product: str
    confidence_score: float
    routing_decision: str
    decision_reason: str
    threshold: float
    top_candidates: list[PredictionCandidate]

    def to_dict(self) -> dict:
        """Return a serializable representation of the prediction."""

        return asdict(self)


class FinancialComplaintTriageService:
    """Clean complaint text, run ModernBERT, and make routing decisions."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.90,
        max_length: int = 512,
        device: str | None = None,
        *,
        tokenizer: object | None = None,
        model: object | None = None,
    ) -> None:
        """
        Initialize the service.

        ``tokenizer`` and ``model`` are an intentionally small test seam. They
        must be supplied together and avoid loading local model artifacts.
        """

        if (tokenizer is None) != (model is None):
            raise ValueError(
                "tokenizer and model must be provided together."
            )

        self.confidence_threshold = self._validate_confidence_threshold(
            confidence_threshold
        )
        self.max_length = max_length
        self.model_path = (
            Path(model_path)
            if model_path is not None
            else None
        )

        using_injected_components = (
            tokenizer is not None
            and model is not None
        )
        self._validate_configuration(
            require_model_path=not using_injected_components
        )
        self.device = self._resolve_device(device)

        if using_injected_components:
            self.tokenizer = tokenizer
            self.model = model.to(self.device)
        else:
            resolved_model_path = self.model_path.resolve()

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    resolved_model_path,
                    use_fast=True,
                )
                self.model = (
                    AutoModelForSequenceClassification
                    .from_pretrained(resolved_model_path)
                    .to(self.device)
                )
            except (OSError, ValueError):
                raise RuntimeError(
                    "The configured model artifacts could not be loaded."
                ) from None

        self.model.eval()

        self.id2label = {
            int(label_id): str(label)
            for label_id, label
            in self.model.config.id2label.items()
        }

        if set(self.id2label) != set(range(11)):
            raise ValueError(
                "The loaded model does not contain the expected 11 labels."
            )

    @staticmethod
    def _validate_confidence_threshold(
        confidence_threshold: float,
    ) -> float:
        """Return a finite confidence threshold in the inclusive [0, 1] range."""

        try:
            validated_threshold = float(confidence_threshold)
        except (TypeError, ValueError):
            raise ValueError(
                "confidence_threshold must be between 0 and 1 inclusive."
            ) from None

        if (
            not math.isfinite(validated_threshold)
            or not 0.0 <= validated_threshold <= 1.0
        ):
            raise ValueError(
                "confidence_threshold must be between 0 and 1 inclusive."
            )

        return validated_threshold

    def _validate_configuration(
        self,
        *,
        require_model_path: bool,
    ) -> None:
        """Validate model location and tokenization configuration."""

        if (
            require_model_path
            and (
                self.model_path is None
                or not self.model_path.is_dir()
            )
        ):
            raise FileNotFoundError(
                "Model artifacts were not found at the configured location."
            )

        if not isinstance(self.max_length, int) or self.max_length <= 0:
            raise ValueError(
                "max_length must be a positive integer."
            )

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        """Select CPU or an available CUDA device."""

        selected_device = (
            device
            if device is not None
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        resolved_device = torch.device(selected_device)

        if (
            resolved_device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return resolved_device

    @staticmethod
    def clean_text(text: object) -> str:
        """
        Clean a complaint using the same rules as dataset preparation.

        Punctuation, capitalization, and anonymization markers are preserved.
        """

        if text is None:
            return ""

        cleaned_text = html.unescape(str(text))
        cleaned_text = re.sub(
            pattern=r"<[^>]+>",
            repl=" ",
            string=cleaned_text,
        )
        cleaned_text = re.sub(
            pattern=r"\s+",
            repl=" ",
            string=cleaned_text,
        )

        return cleaned_text.strip()

    @staticmethod
    def _has_insufficient_text(text: str) -> bool:
        """Identify narratives below the minimum useful text length."""

        return len(text) < 20 or len(text.split()) < 5

    def _predict_probabilities(
        self,
        cleaned_texts: list[str],
    ) -> torch.Tensor:
        """Tokenize and infer over one internal batch."""

        encoded_inputs = self.tokenizer(
            cleaned_texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded_inputs = {
            name: tensor.to(self.device)
            for name, tensor in encoded_inputs.items()
        }

        use_bfloat16 = (
            self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )

        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=use_bfloat16,
            ):
                outputs = self.model(**encoded_inputs)

        return torch.softmax(
            outputs.logits.float(),
            dim=-1,
        ).cpu()

    def _normalize_top_k(self, top_k: int) -> int:
        """Validate and cap the requested number of candidates."""

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be at least 1.")

        return min(top_k, len(self.id2label))

    def _clean_and_validate_text(
        self,
        text: object,
        *,
        position: int | None = None,
    ) -> str:
        """Clean one input and reject an empty result without echoing it."""

        cleaned_text = self.clean_text(text)

        if not cleaned_text:
            if position is None:
                raise ValueError(
                    "The complaint narrative cannot be empty."
                )

            raise ValueError(
                f"The complaint narrative at position {position} "
                "cannot be empty."
            )

        return cleaned_text

    def _build_prediction(
        self,
        cleaned_text: str,
        probabilities: torch.Tensor,
        top_k: int,
    ) -> TriagePrediction:
        """Build one privacy-safe prediction from unrounded scores."""

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=top_k,
        )
        unrounded_candidates = [
            (
                self.id2label[int(label_id)],
                float(confidence),
            )
            for confidence, label_id in zip(
                top_probabilities,
                top_indices,
            )
        ]
        predicted_product, raw_confidence = unrounded_candidates[0]

        if self._has_insufficient_text(cleaned_text):
            routing_decision = "manual_review"
            decision_reason = "insufficient_text"
        elif raw_confidence < self.confidence_threshold:
            routing_decision = "manual_review"
            decision_reason = "low_confidence"
        else:
            routing_decision = "automatic_route"
            decision_reason = "confidence_threshold_met"

        top_candidates = [
            PredictionCandidate(
                product=product,
                confidence_score=round(confidence, 6),
            )
            for product, confidence in unrounded_candidates
        ]

        return TriagePrediction(
            predicted_product=predicted_product,
            confidence_score=round(raw_confidence, 6),
            routing_decision=routing_decision,
            decision_reason=decision_reason,
            threshold=self.confidence_threshold,
            top_candidates=top_candidates,
        )

    def predict_one(
        self,
        text: object,
        top_k: int = 3,
    ) -> TriagePrediction:
        """Classify one complaint narrative."""

        validated_top_k = self._normalize_top_k(top_k)
        cleaned_text = self._clean_and_validate_text(text)
        probabilities = self._predict_probabilities(
            [cleaned_text]
        )[0]

        return self._build_prediction(
            cleaned_text=cleaned_text,
            probabilities=probabilities,
            top_k=validated_top_k,
        )

    def predict_batch(
        self,
        texts: Iterable[object],
        top_k: int = 3,
        batch_size: int = 32,
    ) -> list[TriagePrediction]:
        """Classify complaints using true batched tokenization and inference."""

        if isinstance(texts, (str, bytes)):
            raise ValueError(
                "texts must be an iterable of complaint narratives."
            )

        try:
            input_texts = list(texts)
        except TypeError:
            raise ValueError(
                "texts must be an iterable of complaint narratives."
            ) from None

        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(
                "batch_size must be a positive integer."
            )

        validated_top_k = self._normalize_top_k(top_k)
        cleaned_texts = [
            self._clean_and_validate_text(
                text,
                position=position,
            )
            for position, text in enumerate(input_texts)
        ]

        predictions: list[TriagePrediction] = []

        for start in range(0, len(cleaned_texts), batch_size):
            internal_batch = cleaned_texts[
                start:start + batch_size
            ]
            batch_probabilities = self._predict_probabilities(
                internal_batch
            )

            predictions.extend(
                self._build_prediction(
                    cleaned_text=cleaned_text,
                    probabilities=probabilities,
                    top_k=validated_top_k,
                )
                for cleaned_text, probabilities in zip(
                    internal_batch,
                    batch_probabilities,
                    strict=True,
                )
            )

        return predictions
