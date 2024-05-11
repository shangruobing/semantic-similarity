import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score

from src.core.model import FineTuneModel


class Evaluator:
    def __init__(self, data_path, folder_path, model_name):
        self.dataset = pd.read_csv(data_path).values
        self.folder_path = folder_path / "evaluate"
        self.model_name = model_name
        self.file_path = self.folder_path / f"{self.model_name}.csv"

    def _init_writer(self):
        os.makedirs(self.folder_path, exist_ok=True)
        # print(f"The {self.file_path} does not exist, a new file has been created.")
        writer = pd.DataFrame(columns=["name", "rewrite", "label", "predict", "confidence"])
        return writer

    def evaluate(self, model: FineTuneModel):
        """
        Evaluate the model.
        Args:
            model: the finetune model

        Returns:

        """
        writer = self._init_writer()
        predicts = []
        for data in tqdm(iterable=self.dataset, desc="Evaluating"):
            name, rewrite, label = data
            predict, confidence = model.classify(*[name, rewrite], threshold=0.8)
            writer.loc[len(writer)] = {
                'name': name,
                'rewrite': rewrite,
                'label': label,
                'predict': predict,
                "confidence": round(confidence, 4)
            }
            predicts.append(predict)
        writer.to_csv(self.file_path, index=False, encoding="utf-8-sig")

        labels = self.dataset[:, -1].tolist()
        print("Confusion Matrix:")
        print(confusion_matrix(y_true=labels, y_pred=predicts))
        print("Accuracy:", accuracy_score(y_true=labels, y_pred=predicts))
