from core.evaluator import Evaluator
from baseline.model import BaselineModel
from cross.model import CrossFineTuneModel
from siamese.model import SiameseFineTuneModel
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
