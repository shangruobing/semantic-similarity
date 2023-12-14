import os
import csv
import torch
import torch.nn as nn
from tqdm import tqdm
from src.config import FINE_TUNE
from src.utils import get_device
from src.dataset import train_dataloader, test_dataloader


class Trainer:

    def __init__(self, tokenizer, model, model_name, criterion=nn.BCELoss(), num_epochs=10, learning_rate=1e-5):
        self.device = get_device()
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.model_name = model_name
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for intent, description, label in tqdm(train_dataloader, disable=False):
                label = label.float().to(self.device)
                outputs = self.model(intent, description)
                assert outputs.shape == label.shape
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()

            average_loss = total_loss / len(train_dataloader)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}] - Average Loss: {round(average_loss, 4)}")

            self.model.eval()
            with torch.no_grad(), open(self.model_name + "_log.csv", "a", encoding="utf-8-sig") as log, \
                    open(self.model_name + "_metrics.csv", "a", encoding="utf-8-sig") as metrics:
                log_writer = csv.writer(log)
                metrics_writer = csv.writer(metrics)
                log_writer.writerow(["intent", "description", "output", "predict", "label"])
                metrics_writer.writerow(["epoch", "threshold", "accuracy", "precision", "recall", "f1"])
                for threshold in [i / 10.0 for i in range(1, 10)]:
                    print(f"Threshold:{threshold}")
                    counts = [0, 0, 0, 0]  # [true_positive, false_positive, true_negative, false_negative]
                    for intent, description, label in test_dataloader:
                        label = label.item()
                        outputs = self.model(intent, description)
                        predicted = 1 if outputs > threshold else 0
                        log_writer.writerow([intent, description, round(outputs.item(), 4), predicted, label])
                        assert type(predicted) is type(label)
                        if predicted == label:
                            counts[label] += 1
                        else:
                            counts[2 + label] += 1

                    accuracy, precision, recall, f1 = self._calculate_metrics(counts=counts)
                    metrics_writer.writerow([epoch, threshold, accuracy, precision, recall, f1])
                    print(f'Validation Accuracy:{accuracy}')
                    print(f'Validation Precision:{precision}')
                    print(f'Validation Recall:{recall}')
                    print(f'Validation F1:{f1}')

        if not os.path.exists(FINE_TUNE):
            os.makedirs(FINE_TUNE)
        torch.save(self.model.state_dict(), FINE_TUNE / 'pytorch_model.pth')

    @staticmethod
    def _calculate_metrics(counts):
        true_positive, false_positive, true_negative, false_negative = counts
        print("true_positive, true_negative, false_positive, false_negative", counts)
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return round(accuracy, 4), round(precision, 4), round(recall, 4), round(f1, 4)
