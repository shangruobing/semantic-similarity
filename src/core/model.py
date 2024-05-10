from typing import Tuple
from abc import ABC, abstractmethod
from transformers import BertTokenizer, BertForSequenceClassification

from src.core.utils import get_device


class FineTuneModel(ABC):

    def __init__(self, model_path):
        """
        Initialize the model.
        Args:
            model_path: the path of the model
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self._init_model()

    def _init_model(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path).to(get_device())

    @abstractmethod
    def _fit(self, sentence, candidate_sentence, threshold=0.5) -> Tuple[int, float]:
        """
        The input of the model.
        Args:
            sentence: sentence
            candidate_sentence: candidate sentence
            threshold: threshold

        Returns: (label, score)

        """
        raise NotImplementedError

    def classify(self, sentence, candidate_sentence, threshold=0.5) -> Tuple[int, float]:
        """
        Classify the pair of sentences.
        Args:
            sentence: sentence
            candidate_sentence: candidate sentence
            threshold: threshold

        Returns: (label, score)

        """
        label, score = self._fit(sentence, candidate_sentence, threshold=threshold)
        return label, score
