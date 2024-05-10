import os
import csv
import torch
import torch.nn as nn
from tqdm import tqdm
from tabulate import tabulate
from config import FINE_TUNE
from utils import get_device
from dataset import get_dataloader


class Trainer:

    def __init__(
            self,
            model,
            model_name,
            criterion=nn.BCELoss(),
            num_epochs=10,
            learning_rate=1e-5,
            save_path=FINE_TUNE
    ):
        self.device = get_device()
        self.model = model.to(self.device)
        self.model_name = model_name
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.save_path = save_path
        self.train_dataloader, self.test_dataloader = get_dataloader()

    def train(self):
        for epoch in tqdm(iterable=range(1, self.num_epochs + 1), desc="Training", disable=True):
            self.model.train()
            total_loss = 0
            for intent, description, label in self.train_dataloader:
                label = label.float().to(self.device)
                outputs = self.model(intent, description)
                assert outputs.shape == label.shape
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()
            average_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch [{epoch}/{self.num_epochs}] - Average Loss: {round(average_loss, 4)}")

            self._evaluate(epoch)

        self._save_model()

    def _evaluate(self, epoch):
        self.model.eval()
        with (
            torch.no_grad(),
            open(self.model_name + "_log.csv", "w", encoding="utf-8-sig") as log,
            open(self.model_name + "_metrics.csv", "w", encoding="utf-8-sig") as metrics
        ):
            log_writer = csv.writer(log)
            metrics_writer = csv.writer(metrics)
            log_writer.writerow(["intent", "description", "output", "predict", "label"])
            metrics_writer.writerow(["epoch", "threshold", "accuracy", "precision", "recall", "f1"])
            headers = ["Epoch", "Counts", "Threshold", "Accuracy", "Precision", "Recall", "F1"]
            table = []
            for threshold in [i / 10.0 for i in range(1, 10)]:
                counts = [0, 0, 0, 0]  # [true_positive, false_positive, true_negative, false_negative]
                for intent, description, label in self.test_dataloader:
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
                table.append([epoch, str(counts), threshold, accuracy, precision, recall, f1])
            print(tabulate(tabular_data=table, headers=headers, tablefmt="simple_outline"))

    @staticmethod
    def _calculate_metrics(counts):
        true_positive, false_positive, true_negative, false_negative = counts
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return round(accuracy, 2), round(precision, 2), round(recall, 2), round(f1, 2)

    def _save_model(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.model.state_dict(), self.save_path / f'{self.model_name}_pytorch_model.pth')
