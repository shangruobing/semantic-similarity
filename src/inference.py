from config import SIMILARITY_MODEL, SIAMESE_MODEL_STATE_DICT, CROSS_MODEL_STATE_DICT
from baseline.model import BaselineModel
from cross.model import CrossFineTuneModel
from siamese.model import SiameseFineTuneModel

corpus = ("下雨就打车去苏州大学", "将ABC添加到我的歌单")
print(corpus)

print("=============BaseLineModel=============")
model = BaselineModel(model_path=SIMILARITY_MODEL)
print(model.classify(*corpus))

print("===========CrossFineTuneModel===========")
model = SiameseFineTuneModel(model_path=SIMILARITY_MODEL, state_dict_path=SIAMESE_MODEL_STATE_DICT)
print(model.classify(*corpus))

print("==========SiameseFineTuneModel==========")
model = CrossFineTuneModel(model_path=SIMILARITY_MODEL, state_dict_path=CROSS_MODEL_STATE_DICT)
print(model.classify(*corpus))
