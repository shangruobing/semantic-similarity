from transformers import BertTokenizer, BertForSequenceClassification
import json
from src.config import ALL_API_JSON
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Union
from src.utils import get_device


class FineTuneModel(ABC):

    def __init__(self, model_path):
        self.model_path = model_path
        self.apis = []
        self.tokenizer = None
        self.model = None
        self.__load_apis()
        self._init_model()

    def __load_apis(self) -> None:
        with open(ALL_API_JSON, "r", encoding="utf-8") as file:
            all_api = json.load(file)

        result = []
        for domain, apis in all_api.items():
            for index, api in apis.items():
                result.append(api)

        self.apis = [{"name": item["name"], "description": item["description"]} for item in result]

    def _init_model(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path).to(get_device())

    @abstractmethod
    def _fit(self, sentence, candidate_sentence, threshold=0.5) -> Tuple[int, float]:
        raise NotImplementedError

    def similarity_match(self, sentence, top=3, only_name=False) -> Union[List[Dict], List[str]]:
        actions = []
        for api in self.apis:
            label, score = self._fit(sentence=sentence, candidate_sentence=api["description"])
            if label == 1:
                actions.append({"name": api["name"], "description": api["description"], "score": score})
        actions = sorted(actions, key=lambda x: x["score"], reverse=True)[:top]
        return actions if not only_name else [action["name"] for action in actions]

    def classify(self, sentence1, sentence2) -> Tuple[int, float]:
        raise NotImplementedError

    def classify_with_score(self, sentence1, sentence2) -> int:
        raise NotImplementedError
