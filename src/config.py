import os
from pathlib import Path
from core.utils import get_now_date, get_now_datetime

# 项目根目录
ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent
# 运行结果所在目录
FOLDER_PATH = ROOT_PATH / f"result/{get_now_date()}/"

# 数据文件路径
DATA_PATH = ROOT_PATH / "data"
ALL_REWRITE_INTENT_JSON = DATA_PATH / Path("json/rewrite_intent.json")

# 本地模型路径
SIMILARITY_MODEL = str(ROOT_PATH / Path("models/similarity/"))
CROSS_MODEL_STATE_DICT = str(ROOT_PATH / Path("models/cross/pytorch_model.pth"))
SIAMESE_MODEL_STATE_DICT = str(ROOT_PATH / Path("models/siamese/pytorch_model.pth"))
