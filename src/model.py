from typing import Tuple
from src.utils import get_device
from abc import ABC, abstractmethod
from transformers import BertTokenizer, BertForSequenceClassification


class FineTuneModel(ABC):

    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self._init_model()

    def _init_model(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path).to(get_device())

    @abstractmethod
    def _fit(self, sentence, candidate_sentence, threshold=0.6) -> Tuple[int, float]:
        raise NotImplementedError

    def classify(self, sentence, candidate_sentence) -> Tuple[int, float]:
        label, score = self._fit(sentence, candidate_sentence)
        return label, score
