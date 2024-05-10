import time
from datetime import datetime
import torch


def get_now_date():
    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d")
    return formatted_time


def get_now_datetime():
    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M")
    return formatted_time


def running_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        run_time = round(end_time - start_time, 4)
        wrapper.run_time = run_time
        print("Function {} Running Time：{} seconds".format(func.__name__, run_time))
        return result

    return wrapper


def get_device(name=None):
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    if name:
        device = torch.device(name)
    return device
