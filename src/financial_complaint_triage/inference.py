from __future__ import annotations

import html
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
    """Una categoría candidata producida por el modelo."""

    product: str
    confidence_score: float


@dataclass(frozen=True)
class TriagePrediction:
    """Resultado completo del sistema de clasificación."""

    predicted_product: str
    confidence_score: float
    routing_decision: str
    decision_reason: str
    threshold: float
    cleaned_text: str
    top_candidates: list[PredictionCandidate]

    def to_dict(self) -> dict:
        """Convierte el resultado a un diccionario serializable."""
        return asdict(self)


class FinancialComplaintTriageService:
    """
    Servicio de inferencia para clasificar reclamaciones financieras.

    El servicio:

    1. Limpia el texto usando las mismas reglas del entrenamiento.
    2. Ejecuta ModernBERT.
    3. Obtiene el producto financiero más probable.
    4. Decide entre enrutamiento automático y revisión humana.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.90,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model_path:
            Carpeta local que contiene el modelo y tokenizador.

        confidence_threshold:
            Confianza mínima para realizar enrutamiento automático.

        max_length:
            Longitud máxima utilizada durante la tokenización.

        device:
            Dispositivo de inferencia. Ejemplos: ``cuda`` o ``cpu``.
            Cuando es None, se selecciona automáticamente.
        """
        self.model_path = Path(model_path).resolve()
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length

        self._validate_configuration()

        self.device = self._resolve_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(self.model_path)
            .to(self.device)
        )

        self.model.eval()

        self.id2label = {
            int(label_id): str(label)
            for label_id, label
            in self.model.config.id2label.items()
        }

        if len(self.id2label) != 11:
            raise ValueError(
                "El modelo cargado no contiene las 11 clases esperadas."
            )

    def _validate_configuration(self) -> None:
        """Valida las opciones principales del servicio."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo:\n{self.model_path}"
            )

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold debe estar entre 0 y 1."
            )

        if self.max_length <= 0:
            raise ValueError(
                "max_length debe ser mayor que cero."
            )

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        """Selecciona CPU o GPU."""

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
                "Se solicitó CUDA, pero CUDA no está disponible."
            )

        return resolved_device

    @staticmethod
    def clean_text(text: object) -> str:
        """
        Limpia una narrativa usando las reglas del dataset.

        Conserva puntuación, mayúsculas y marcas de anonimización.
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
        """Detecta narrativas fuera de la longitud mínima esperada."""

        character_count = len(text)
        word_count = len(text.split())

        return (
            character_count < 20
            or word_count < 5
        )

    def _predict_probabilities(
        self,
        cleaned_texts: list[str],
    ) -> torch.Tensor:
        """Ejecuta inferencia y devuelve probabilidades por clase."""

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
                outputs = self.model(
                    **encoded_inputs
                )

        return torch.softmax(
            outputs.logits.float(),
            dim=-1,
        ).cpu()

    def predict_one(
        self,
        text: object,
        top_k: int = 3,
    ) -> TriagePrediction:
        """
        Clasifica una sola reclamación financiera.

        Returns
        -------
        TriagePrediction
            Producto predicho, confianza, decisión y candidatos.
        """

        cleaned_text = self.clean_text(text)

        if not cleaned_text:
            raise ValueError(
                "La narrativa no puede estar vacía."
            )

        if top_k < 1:
            raise ValueError(
                "top_k debe ser al menos 1."
            )

        top_k = min(
            top_k,
            len(self.id2label),
        )

        probabilities = self._predict_probabilities(
            [cleaned_text]
        )[0]

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=top_k,
        )

        top_candidates = [
            PredictionCandidate(
                product=self.id2label[int(label_id)],
                confidence_score=round(
                    float(confidence),
                    6,
                ),
            )
            for confidence, label_id
            in zip(
                top_probabilities,
                top_indices,
            )
        ]

        best_candidate = top_candidates[0]

        insufficient_text = (
            self._has_insufficient_text(
                cleaned_text
            )
        )

        if insufficient_text:
            routing_decision = "manual_review"
            decision_reason = "insufficient_text"

        elif (
            best_candidate.confidence_score
            < self.confidence_threshold
        ):
            routing_decision = "manual_review"
            decision_reason = "low_confidence"

        else:
            routing_decision = "automatic_route"
            decision_reason = "confidence_threshold_met"

        return TriagePrediction(
            predicted_product=best_candidate.product,
            confidence_score=(
                best_candidate.confidence_score
            ),
            routing_decision=routing_decision,
            decision_reason=decision_reason,
            threshold=self.confidence_threshold,
            cleaned_text=cleaned_text,
            top_candidates=top_candidates,
        )

    def predict_batch(
        self,
        texts: Iterable[object],
        top_k: int = 3,
    ) -> list[TriagePrediction]:
        """
        Clasifica varias narrativas.

        Esta implementación prioriza claridad y reutiliza predict_one.
        La optimización por lotes puede añadirse durante el despliegue.
        """

        return [
            self.predict_one(
                text=text,
                top_k=top_k,
            )
            for text in texts
        ]