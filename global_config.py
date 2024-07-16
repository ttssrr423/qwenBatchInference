import json
import os
import torch
from multiprocessing import Manager

script_path = os.path.dirname(__file__)
GLOBAL_CONFIG = json.load(open(os.path.join(script_path, "configuration.json"), encoding="utf8"))
PORT = GLOBAL_CONFIG["server_port"]

NUM_GPUS = int(torch.cuda.device_count())
assert NUM_GPUS >= 1
PIPELINE_PARALLEL_SIZE = min(GLOBAL_CONFIG["pipeline_parallel_size"], NUM_GPUS)
WORLD_SIZE = int(NUM_GPUS // PIPELINE_PARALLEL_SIZE) * PIPELINE_PARALLEL_SIZE
DATA_PARALLEL_SIZE = WORLD_SIZE / PIPELINE_PARALLEL_SIZE

USE_LORA = bool(GLOBAL_CONFIG["use_lora"])

class DDP_PoolCls():
    def __init__(self):
        self._manager = Manager()
        self._dict = self._manager.dict() # 输出结果的信息
        self._daemon_dict = self._manager.dict() # 清理字典的守护进程信息
        self._queue = self._manager.Queue() # 输入queue的信息
        self._extra = self._manager.dict() # 额外进程间通信用于加载。
        self._buffer_info = None
    @property
    def manager(self):
        return self._manager
    @property
    def dict(self):
        return self._dict
    @property
    def daemon_dict(self):
        return self._daemon_dict
    @property
    def queue(self):
        return self._queue
    @property
    def extra(self):
        return self._extra
    @property
    def buffer_info(self):
        return self._buffer_info

    def set_buffer(self, buffer_instance):
        self._buffer_info = buffer_instance

DDP_POOL = DDP_PoolCls()