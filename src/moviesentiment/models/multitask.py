"""Multi-task DistilBERT — shared encoder + five heads (v2 Review Intelligence).

Heads:
    * sentiment        : 2-class softmax (existing IMDb binary)
    * aspect_sentiment : 5 aspects x 3-class softmax (acting, plot, visuals,
                         pacing, sound -> {neg, neu, pos})
    * emotion          : 6-class softmax (Ekman: joy/anger/fear/sadness/
                         surprise/disgust)
    * spoiler          : 2-class softmax (spoiler / not)
    * helpfulness      : sigmoid regression in [0, 1]

The shared encoder is `distilbert-base-uncased` (or whatever
`params.yaml::transformer.model_name` resolves to). Heads are small linear
projections on top of the pooled `[CLS]` representation.

Joint loss is task-weighted sum of cross-entropy / BCE / MSE; weights live in
`params.yaml::multitask.loss_weights`. Per-task supervision is optional via
`labels_*` kwargs in the forward signature so the same model can be trained on
sources that only label a subset of tasks (e.g. IMDb 50K has only sentiment,
GoEmotions has only emotion, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


ASPECTS = ("acting", "plot", "visuals", "pacing", "sound")
EMOTIONS = ("joy", "anger", "fear", "sadness", "surprise", "disgust")
N_ASPECTS = 5
N_EMOTIONS = 6


@dataclass
class MultiTaskOutput:
    """Outputs of the multi-task model. All logits / probs are CPU-side tensors."""

    sentiment_logits: Tensor  # (B, 2)
    aspect_logits: Tensor  # (B, 5, 3)
    emotion_logits: Tensor  # (B, 6)
    spoiler_logits: Tensor  # (B, 2)
    helpfulness: Tensor  # (B,) in [0,1]
    loss: Tensor | None = None


def build_multitask_model(model_name: str, revision: str | None = None) -> Any:
    """Construct the multi-task DistilBERT (torch). Heavy imports happen inside."""
    import torch
    from torch import nn
    from transformers import AutoModel

    class MultiTaskDistilBert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name, revision=revision)
            hidden = self.encoder.config.hidden_size
            self.sentiment_head = nn.Linear(hidden, 2)
            self.aspect_head = nn.Linear(hidden, N_ASPECTS * 3)
            self.emotion_head = nn.Linear(hidden, N_EMOTIONS)
            self.spoiler_head = nn.Linear(hidden, 2)
            self.helpfulness_head = nn.Linear(hidden, 1)

        def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor | None = None,
            labels_sentiment: Tensor | None = None,
            labels_aspect: Tensor | None = None,
            labels_emotion: Tensor | None = None,
            labels_spoiler: Tensor | None = None,
            labels_helpfulness: Tensor | None = None,
            loss_weights: dict[str, float] | None = None,
        ) -> MultiTaskOutput:
            enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # DistilBERT exposes last_hidden_state; use [CLS] token at position 0.
            cls = enc.last_hidden_state[:, 0, :]

            sent_logits = self.sentiment_head(cls)
            aspect_logits = self.aspect_head(cls).view(-1, N_ASPECTS, 3)
            emotion_logits = self.emotion_head(cls)
            spoiler_logits = self.spoiler_head(cls)
            help_score = torch.sigmoid(self.helpfulness_head(cls)).squeeze(-1)

            total_loss: Tensor | None = None
            if any(
                t is not None
                for t in (
                    labels_sentiment,
                    labels_aspect,
                    labels_emotion,
                    labels_spoiler,
                    labels_helpfulness,
                )
            ):
                w = loss_weights or {}
                total_loss = torch.zeros((), device=cls.device)
                ce = nn.CrossEntropyLoss(ignore_index=-100)
                mse = nn.MSELoss()
                if labels_sentiment is not None:
                    total_loss = total_loss + float(w.get("sentiment", 1.0)) * ce(
                        sent_logits, labels_sentiment
                    )
                if labels_aspect is not None:
                    # labels_aspect shape: (B, N_ASPECTS), values in {-100, 0, 1, 2}
                    total_loss = total_loss + float(w.get("aspect", 1.0)) * ce(
                        aspect_logits.reshape(-1, 3), labels_aspect.reshape(-1)
                    )
                if labels_emotion is not None:
                    total_loss = total_loss + float(w.get("emotion", 1.0)) * ce(
                        emotion_logits, labels_emotion
                    )
                if labels_spoiler is not None:
                    total_loss = total_loss + float(w.get("spoiler", 1.0)) * ce(
                        spoiler_logits, labels_spoiler
                    )
                if labels_helpfulness is not None:
                    # Ignore rows where label is NaN (sentinel for "no supervision").
                    mask = ~torch.isnan(labels_helpfulness)
                    if mask.any():
                        total_loss = total_loss + float(w.get("helpfulness", 1.0)) * mse(
                            help_score[mask], labels_helpfulness[mask]
                        )

            return MultiTaskOutput(
                sentiment_logits=sent_logits,
                aspect_logits=aspect_logits,
                emotion_logits=emotion_logits,
                spoiler_logits=spoiler_logits,
                helpfulness=help_score,
                loss=total_loss,
            )

    return MultiTaskDistilBert()
