import os

import pandas as pd
from tqdm import tqdm
from src.baseline.model import BaselineModel
from src.cross.model import CrossFineTuneModel
from src.siamese.model import SiameseFineTuneModel
from src.config import DATA_PATH, FOLDER_PATH


class Evaluator:
    @staticmethod
    def evaluate(model):
        df = pd.read_csv(DATA_PATH / "dataset/test2.csv")
        name = df['name']
        rewrite = df['rewrite']
        label = df['label']
        predict = []
        for index in tqdm(range(len(df))):
            pred = model.classify(*[name[index], rewrite[index]])
            predict.append(pred)
        if not os.path.exists(FOLDER_PATH):
            os.mkdir(FOLDER_PATH)
        FILE_PATH = f"{FOLDER_PATH}/result.csv"
        if os.path.exists(FILE_PATH):
            df = pd.read_csv(FILE_PATH, encoding="utf-8-sig")
        else:
            print(f"The {FILE_PATH} does not exist, an {FILE_PATH} file has been created.")
            df = pd.DataFrame(columns=["name", "rewrite", "label", "predict"])
        for index in range(len(predict)):
            new_row = {'name': name[index], 'rewrite': rewrite[index], 'label': label[index], 'predict': predict[index]}
            df.loc[len(df)] = new_row
        df.to_csv(FILE_PATH, index=False, encoding="utf-8-sig")


if __name__ == '__main__':
    from src.config import SIMILARITY_MODEL, SIAMESE_MODEL_STATE_DICT, CROSS_MODEL_STATE_DICT

    evaluator = Evaluator()
    # Baseline模型
    print("=============BaseLineModel=============")
    fineTuneModel = BaselineModel(SIMILARITY_MODEL)
    evaluator.evaluate(fineTuneModel)

    # Cross模型
    print("===========CrossFineTuneModel===========")
    fineTuneModel = CrossFineTuneModel(model_path=SIMILARITY_MODEL, state_dict_path=CROSS_MODEL_STATE_DICT)
    evaluator.evaluate(fineTuneModel)

    # Siamese模型
    print("==========SiameseFineTuneModel==========")
    model = SiameseFineTuneModel(model_path=SIMILARITY_MODEL, state_dict_path=SIAMESE_MODEL_STATE_DICT)
    evaluator.evaluate(model)
