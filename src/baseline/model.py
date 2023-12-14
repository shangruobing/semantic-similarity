import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

from src.model import FineTuneModel


class BaselineModel(FineTuneModel):

    def __init__(self, model_path):
        super().__init__(model_path)

    def _init_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertModel.from_pretrained(self.model_path)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _fit(self, sentence, candidate_sentence, threshold=0.6):
        sentences = [sentence, candidate_sentence]
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            cos_similarities = F.cosine_similarity(sentence_embeddings[0].unsqueeze(0),
                                                   sentence_embeddings[1].unsqueeze(0))
            return 1 if cos_similarities.item() > threshold else 0, cos_similarities.item()


if __name__ == '__main__':
    from src.config import SIMILARITY_MODEL

    model = BaselineModel(model_path=SIMILARITY_MODEL)
    corpus = ("下雨就打车去苏州大学", "将ABC添加到我的歌单")
    print(model.classify_with_score(*corpus))
