import pandas as pd
from src.config import SENTENCE_CLASSIFY_PATH, SENTENCE_CLASSIFY_METRICS_PATH
from tqdm import tqdm


def calculate_precision_f1_recall(actual, predicted):
    true_positives = len(set(actual) & set(predicted))
    false_positives = len(set(predicted) - set(actual))
    false_negatives = len(set(actual) - set(predicted))

    try:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    except ZeroDivisionError:
        return 0, 0, 0


class Metrics:
    def __init__(self):
        self.dataframe = pd.read_csv(SENTENCE_CLASSIFY_PATH, encoding="utf-8-sig")
        self.intents = self.dataframe["intent"].values.tolist()
        self.descriptions = self.dataframe["description"].values.tolist()
        self.labels = self.dataframe["label"].values.tolist()

    def execute_classify(self, model, name):
        predictions = [model.classify(intent, self.descriptions[index])
                       for index, intent in tqdm(enumerate(self.intents), total=len(self.intents))]
        self.dataframe[name + '_predict'] = predictions
        precision, recall, accuracy, f1 = self.calculate_metrics(self.labels, predictions)
        self.dataframe[name + '_accuracy'] = accuracy
        self.dataframe[name + '_precision'] = precision
        self.dataframe[name + '_recall'] = recall
        self.dataframe[name + '_f1'] = f1
        self.dataframe.to_csv(SENTENCE_CLASSIFY_PATH, index=False, encoding="utf-8-sig")

    def calculate_gpt(self):
        predictions = self.dataframe["gpt_predict"].values.tolist()
        precision, recall, accuracy, f1 = self.calculate_metrics(self.labels, predictions)
        self.dataframe['gpt_accuracy'] = accuracy
        self.dataframe['gpt_precision'] = precision
        self.dataframe['gpt_recall'] = recall
        self.dataframe['gpt_f1'] = f1
        self.dataframe.to_csv(SENTENCE_CLASSIFY_PATH, index=False, encoding="utf-8-sig")

    def statistics(self, columns):
        metrics = {
            'Name': [],
            "Accuracy": [],
            'Precision': [],
            'Recall': [],
            'F1': [],
        }
        for column in columns:
            metrics["Name"].append(column)
            metrics["Accuracy"].append(round(self.dataframe[column + '_accuracy'][1], 3))
            metrics["Precision"].append(round(self.dataframe[column + '_precision'][1], 3))
            metrics["Recall"].append(round(self.dataframe[column + '_recall'][1], 3))
            metrics["F1"].append(round(self.dataframe[column + '_f1'][1], 3))
        markdown_table = pd.DataFrame(metrics).to_markdown(index=False)
        with open(SENTENCE_CLASSIFY_METRICS_PATH, 'w') as file:
            file.write(markdown_table)

    @staticmethod
    def calculate_metrics(ground_truth, predictions):
        tp = sum(gt == 1 and pred == 1 for gt, pred in zip(ground_truth, predictions))
        fp = sum(gt == 0 and pred == 1 for gt, pred in zip(ground_truth, predictions))
        fn = sum(gt == 1 and pred == 0 for gt, pred in zip(ground_truth, predictions))
        tn = sum(gt == 0 and pred == 0 for gt, pred in zip(ground_truth, predictions))

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        accuracy = (tp + tn) / len(ground_truth)
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        return precision, recall, accuracy, f1


if __name__ == '__main__':
    from src.config import SIMILARITY_MODEL, SIAMESE_MODEL_STATE_DICT, CROSS_MODEL_STATE_DICT
    from src.cross.model import CrossFineTuneModel
    from src.siamese.model import SiameseFineTuneModel

    evaluator = Metrics()

    # Baseline模型
    # print("=============BaseLineModel=============")
    # fineTuneModel = BaseLineModel(SIMILARITY_MODEL)
    # evaluator.execute_classify(fineTuneModel, "baseline")

    # Cross模型
    print("===========CrossFineTuneModel===========")
    fineTuneModel = CrossFineTuneModel(model_path=SIMILARITY_MODEL, state_dict_path=CROSS_MODEL_STATE_DICT)
    evaluator.execute_classify(fineTuneModel, "cross")

    # Siamese模型
    print("==========SiameseFineTuneModel==========")
    model = SiameseFineTuneModel(model_path=SIMILARITY_MODEL, state_dict_path=SIAMESE_MODEL_STATE_DICT)
    evaluator.execute_classify(model, "siamese")

    evaluator.calculate_gpt()

    evaluator.statistics(
        [
            "gpt",
            # "baseline",
            "cross",
            "siamese"
        ]
    )
