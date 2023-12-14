import torch
from transformers import BertTokenizer, BertModel
from src.config import SIMILARITY_MODEL
import torch.nn as nn
from src.utils import get_device
from src.trainer import Trainer


class CrossBertModel(nn.Module):
    def __init__(self, encoder, model):
        super(CrossBertModel, self).__init__()
        self.encoder = encoder
        self.model = model
        self.linear = nn.Linear(in_features=768, out_features=64)
        self.relu = nn.ReLU()
        self.device = get_device()

    def forward(self, input1, input2):
        input1 = input1[0] if isinstance(input1, tuple) else input1
        input2 = input2[0] if isinstance(input2, tuple) else input2
        inputs = self.encoder([input1, input2], padding=True, truncation=True, max_length=128,
                              return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        sent1_rep = outputs.last_hidden_state[0, 0, :]
        sent2_rep = outputs.last_hidden_state[1, 0, :]
        sent1_rep = self.linear(sent1_rep)
        sent2_rep = self.linear(sent2_rep)
        sent1_rep_norm = torch.pow(torch.sum(sent1_rep * sent1_rep, -1), 0.5)
        sent2_rep_norm = torch.pow(torch.sum(sent2_rep * sent2_rep, -1), 0.5)
        dot = torch.sum(sent1_rep * sent2_rep, dim=-1)
        logits = dot / (sent1_rep_norm * sent2_rep_norm)
        logits = self.relu(logits)
        return logits.unsqueeze(dim=0)

    def inference(self, sentence1, sentence2):
        return self.forward(sentence1, sentence2)


def train():
    tokenizer = BertTokenizer.from_pretrained(SIMILARITY_MODEL)
    model = BertModel.from_pretrained(SIMILARITY_MODEL)
    siamese_model = CrossBertModel(tokenizer, model)
    trainer = Trainer(tokenizer=tokenizer, model=siamese_model, model_name="cross")
    trainer.train()


if __name__ == '__main__':
    train()
