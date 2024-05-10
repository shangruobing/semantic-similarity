import torch
from transformers import BertTokenizer, BertModel

from src.siamese.finetune import SiameseBertModel
from src.core.model import FineTuneModel
from src.core.utils import get_device


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

    def _fit(self, sentence, candidate_sentence, threshold=0.5):
        outputs = self.model(sentence, candidate_sentence)
        return 1 if outputs.item() > threshold else 0, outputs.item()
