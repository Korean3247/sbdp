import torch.nn as nn


class DistilBertWrapper(nn.Module):
    """DistilBERT for sequence classification, returning raw logits."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        from transformers import DistilBertForSequenceClassification
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_classes,
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


def get_distilbert(num_classes: int = 4) -> nn.Module:
    return DistilBertWrapper(num_classes)
