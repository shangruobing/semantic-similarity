import torch
from src.model import FineTuneModel
from src.utils import get_device
from transformers import BertTokenizer, BertModel
from src.cross.finetune import CrossBertModel


class CrossFineTuneModel(FineTuneModel):

    def __init__(self, model_path, state_dict_path):
        self.state_dict_path = state_dict_path
        super().__init__(model_path)

    def _init_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        _state_dict = torch.load(self.state_dict_path, map_location=get_device())
        model = BertModel.from_pretrained(self.model_path)
        self.model = CrossBertModel(self.tokenizer, model).to(get_device())
        self.model.load_state_dict(_state_dict)

    def _fit(self, sentence, candidate_sentence, threshold=0.5):
        outputs = self.model(sentence, candidate_sentence)
        return 1 if outputs.item() > threshold else 0, outputs

    def classify(self, sentence1, sentence2):
        label, score = self._fit(sentence1, sentence2)
        return label

    def classify_with_score(self, sentence1, sentence2):
        label, score = self._fit(sentence1, sentence2)
        return label, score


if __name__ == '__main__':
    from src.config import SIMILARITY_MODEL, CROSS_MODEL_STATE_DICT

    corpus = ["下雨就打车去苏州大学", "将ABC添加到我的歌单"]
    model = CrossFineTuneModel(model_path=SIMILARITY_MODEL, state_dict_path=CROSS_MODEL_STATE_DICT)
    select_list = [model.similarity_match(item, only_name=True) for item in corpus]
    print(select_list)
    print(model.classify_with_score("吃饭了吗", "我正在吃饭"))
    print(model.classify_with_score("吃饭了吗", "吃饭了吗"))
    print(model.classify_with_score("为什么不一样", "为什么不一样"))
    print(model.classify_with_score("下雨就打车去苏州大学", "将ABC添加到我的歌单"))
