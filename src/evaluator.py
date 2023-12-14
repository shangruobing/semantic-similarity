import pandas as pd
from src.metrics import calculate_precision_f1_recall
from src.config import API_RETRIEVE_PATH, API_RETRIEVE_METRICS_PATH
from tqdm import tqdm
from src.cross.model import CrossFineTuneModel
from src.siamese.model import SiameseFineTuneModel


class Evaluator:
    def __init__(self):
        self.dataframe = None
        self.intents = []
        self.actions = []
        self.__load_real_data()

    def __load_real_data(self):
        self.dataframe = pd.read_csv(API_RETRIEVE_PATH, encoding="utf-8-sig")
        self.intents = self.dataframe["intent"].values.tolist()
        self.actions = self.dataframe["actions"].values.tolist()

    def execute_similarity_match(self, model, name):
        select_list = [model.similarity_match(item, only_name=True, top=3) for item in tqdm(self.intents)]
        self.dataframe[name + '_actions'] = select_list
        precisions = []
        recalls = []
        f1s = []
        for index, select in enumerate(select_list):
            precision, recall, f1 = calculate_precision_f1_recall(eval(self.actions[index]), eval(str(select)))
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        self.dataframe[name + '_precision'] = precisions
        self.dataframe[name + '_recall'] = recalls
        self.dataframe[name + '_f1'] = f1s
        self.dataframe.to_csv(API_RETRIEVE_PATH, index=False, encoding="utf-8-sig")

    def statistics(self, columns):
        metrics = {
            'Name': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
        }
        for column in columns:
            metrics["Name"].append(column)
            metrics["Precision"].append(round(self.dataframe[column + '_precision'].sum(), 3))
            metrics["Recall"].append(round(self.dataframe[column + '_recall'].sum(), 3))
            metrics["F1"].append(round(self.dataframe[column + '_f1'].sum(), 3))
        markdown_table = pd.DataFrame(metrics).to_markdown(index=False)
        with open(API_RETRIEVE_METRICS_PATH, 'w') as file:
            file.write(markdown_table)


if __name__ == '__main__':
    from src.config import SIMILARITY_MODEL, SIAMESE_MODEL_STATE_DICT, CROSS_MODEL_STATE_DICT

    evaluator = Evaluator()
    # Baseline模型
    # print("=============BaseLineModel=============")
    # fineTuneModel = BaseLineModel(SIMILARITY_MODEL)
    # evaluator.execute_similarity_match(fineTuneModel, "baseline")

    # Cross模型
    print("===========CrossFineTuneModel===========")
    fineTuneModel = CrossFineTuneModel(model_path=SIMILARITY_MODEL, state_dict_path=CROSS_MODEL_STATE_DICT)
    evaluator.execute_similarity_match(fineTuneModel, "cross")

    # Siamese模型
    print("==========SiameseFineTuneModel==========")
    model = SiameseFineTuneModel(model_path=SIMILARITY_MODEL, state_dict_path=SIAMESE_MODEL_STATE_DICT)
    evaluator.execute_similarity_match(model, "siamese")

    evaluator.statistics(
        [
            "baseline",
            "cross",
            "siamese",
        ]
    )
