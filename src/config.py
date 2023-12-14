import os
from pathlib import Path
from src.utils import get_now_date, get_now_datetime

# 项目根目录
ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent
# 运行结果所在目录
FOLDER_PATH = ROOT_PATH / f"result/{get_now_date()}/"

# API选择文件路径
API_RETRIEVE_PATH = FOLDER_PATH / "api_retrieve.csv"
API_RETRIEVE_METRICS_PATH = FOLDER_PATH / "api_metrics.md"
# 语句相似度分类文件路径
SENTENCE_CLASSIFY_PATH = FOLDER_PATH / "sentence_classify.csv"
SENTENCE_CLASSIFY_METRICS_PATH = FOLDER_PATH / "sentence_classify.md"

# 数据文件路径
DATA_PATH = ROOT_PATH / "data"
ALL_REWRITE_INTENT_JSON = DATA_PATH / Path("json/rewrite_intent.json")
ALL_API_JSON = str(DATA_PATH / Path("json/all_api.json"))
ALL_INTENT_API_CSV = DATA_PATH / Path("csv/intent_api.csv")

# 本地模型路径
SIMILARITY_MODEL = str(ROOT_PATH / Path("models/similarity/"))
FINE_TUNE = ROOT_PATH / Path("models/" + get_now_datetime())
CROSS_MODEL_STATE_DICT = str(ROOT_PATH / Path("models/cross/pytorch_model.pth"))
SIAMESE_MODEL_STATE_DICT = str(ROOT_PATH / Path("models/siamese/pytorch_model.pth"))
