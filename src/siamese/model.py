from transformers import BertTokenizer, BertModel
import torch
from src.siamese.finetune import SiameseBertModel
from src.model import FineTuneModel
from src.utils import get_device


class SiameseFineTuneModel(FineTuneModel):

    def __init__(self, model_path, state_dict_path):
        self.state_dict_path = state_dict_path
        super().__init__(model_path)

    def _init_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        _state_dict = torch.load(self.state_dict_path, map_location=get_device())
        model = BertModel.from_pretrained(self.model_path)
        self.model = SiameseBertModel(self.tokenizer, model).to(get_device())
        self.model.load_state_dict(_state_dict)

    def _fit(self, sentence, candidate_sentence, threshold=0.6):
        outputs = self.model(sentence, candidate_sentence)
        return 1 if outputs.item() > threshold else 0, outputs.item()


if __name__ == '__main__':
    from src.config import SIMILARITY_MODEL, SIAMESE_MODEL_STATE_DICT

    corpus = ["下雨就打车去苏州大学", "将ABC添加到我的歌单"]
    model = SiameseFineTuneModel(model_path=SIMILARITY_MODEL, state_dict_path=SIAMESE_MODEL_STATE_DICT)
    print(model.classify(*corpus))
