import os

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from baseline.model import BaselineModel
from cross.model import CrossFineTuneModel
from model import FineTuneModel
from siamese.model import SiameseFineTuneModel


class Evaluator:
    def __init__(self, data_path, folder_path, model_name):
        self.dataset = pd.read_csv(data_path).values
        self.folder_path = folder_path
        self.model_name = model_name
        self.file_path = f"{self.folder_path}/{self.model_name}.csv"

    def _init_writer(self):
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        if os.path.exists(self.file_path):
            writer = pd.read_csv(self.file_path, encoding="utf-8-sig")
        else:
            print(f"The {self.file_path} does not exist, a new file has been created.")
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


if __name__ == '__main__':
    from config import DATA_PATH, FOLDER_PATH, SIMILARITY_MODEL, SIAMESE_MODEL_STATE_DICT, CROSS_MODEL_STATE_DICT

    data_path = DATA_PATH / "dataset/test.csv"

    print("=============BaseLineModel=============")
    evaluator = Evaluator(data_path=data_path, folder_path=FOLDER_PATH, model_name="baseline")
    fineTuneModel = BaselineModel(model_path=SIMILARITY_MODEL)
    evaluator.evaluate(fineTuneModel)

    print("===========CrossFineTuneModel===========")
    evaluator = Evaluator(data_path=data_path, folder_path=FOLDER_PATH, model_name="cross")
    model = CrossFineTuneModel(model_path=SIMILARITY_MODEL, state_dict_path=CROSS_MODEL_STATE_DICT)
    evaluator.evaluate(model)

    print("==========SiameseFineTuneModel==========")
    evaluator = Evaluator(data_path=data_path, folder_path=FOLDER_PATH, model_name="siamese")
    model = SiameseFineTuneModel(model_path=SIMILARITY_MODEL, state_dict_path=SIAMESE_MODEL_STATE_DICT)
    evaluator.evaluate(model)
