from transformers import BertTokenizer, BertModel
from src.config import SIMILARITY_MODEL
import torch.nn as nn
import torch
from src.utils import get_device
from src.trainer import Trainer


class SiameseBertModel(nn.Module):
    def __init__(self, encoder, model):
        super(SiameseBertModel, self).__init__()
        self.encoder = encoder
        self.model = model
        self.linear = nn.Linear(in_features=768, out_features=64)
        self.relu = nn.ReLU()
        self.device = get_device()

    def forward(self, input1, input2):
        input1 = input1[0] if isinstance(input1, tuple) else input1
        input2 = input2[0] if isinstance(input2, tuple) else input2
        inputs1 = self.encoder(input1, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        inputs2 = self.encoder(input2, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        outputs1 = self.model(**inputs1)
        outputs2 = self.model(**inputs2)
        sent1_rep = outputs1.last_hidden_state[:, 0, :]
        sent2_rep = outputs2.last_hidden_state[:, 0, :]
        sent1_rep = self.linear(sent1_rep)
        sent2_rep = self.linear(sent2_rep)
        sent1_rep_norm = torch.pow(torch.sum(sent1_rep * sent1_rep, -1), 0.5)
        sent2_rep_norm = torch.pow(torch.sum(sent2_rep * sent2_rep, -1), 0.5)
        dot = torch.sum(sent1_rep * sent2_rep, dim=-1)
        logits = dot / (sent1_rep_norm * sent2_rep_norm)
        logits = self.relu(logits)
        return logits

    def inference(self, sentence):
        tokenized = self.encoder(sentence, padding=True, truncation=True, max_length=128, return_tensors="pt").to(
            self.device)
        outputs1 = self.model(**tokenized)
        sent1_rep = outputs1.last_hidden_state[:, 0, :]
        sent1_rep = self.linear(sent1_rep)
        sent1_rep_norm = torch.pow(torch.sum(sent1_rep * sent1_rep, -1), 0.5)
        return sent1_rep_norm


def train():
    tokenizer = BertTokenizer.from_pretrained(SIMILARITY_MODEL)
    model = BertModel.from_pretrained(SIMILARITY_MODEL)
    siamese_model = SiameseBertModel(tokenizer, model)
    trainer = Trainer(model=siamese_model, model_name="siamese")
    trainer.train()


if __name__ == '__main__':
    train()
