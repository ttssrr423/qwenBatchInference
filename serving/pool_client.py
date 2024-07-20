import asyncio
import queue
import hashlib
import random
import datetime
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED
from global_config import DATA_PARALLEL_SIZE, GLOBAL_CONFIG, DDP_POOL
from serving.stream_pool import StreamPool, RetState, GenState

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
    level="INFO"
)
logger = logging.getLogger(__name__)
script_path = os.path.dirname(__file__)

MAX_QUEUE_LEN = int(DATA_PARALLEL_SIZE * GLOBAL_CONFIG["max_dynamic_bsz"] * 5)

def get_hash(_text):
    hasher = hashlib.md5()
    tm = datetime.datetime.now()
    rand_num = str(random.random())[-4:]
    text_with_time = _text + tm.strftime("%Y-%m-%d %H:%M:%S.%f") + rand_num
    hasher.update(text_with_time.encode(encoding="utf-8"))
    return hasher.hexdigest(), tm

def release_resource(request_id, buffer_rid=None, stream_pool=None):
    if request_id in DDP_POOL.dict:
        if isinstance(DDP_POOL.dict[request_id], int):
            del DDP_POOL.dict[request_id]
    if buffer_rid is not None and buffer_rid >= 0:
        stream_pool.set_ret_state(buffer_rid, RetState.END)
        stream_pool.set_timeout(buffer_rid)

class ThredPoolExecutWithLimitedQueue(ThreadPoolExecutor):
    def __init__(self, max_queue_size=MAX_QUEUE_LEN, *args, **kwargs):
        super(ThredPoolExecutWithLimitedQueue, self).__init__(*args, **kwargs)
        self._work_queue = queue.Queue(maxsize=max_queue_size)

async def simple_chat(buffer_info, text_info, hist, gen_kwargs, timeout=120, request_id=None):
    if request_id is None:
        request_id, start_tm = get_hash(text_info[0])
    else:
        start_tm = datetime.datetime.now()
    input_info = [request_id, False, text_info, hist, gen_kwargs]
    DDP_POOL.dict[request_id] = "waiting"
    dict_item = "waiting"
    DDP_POOL.queue.put(input_info, timeout=timeout)
    logger.info(f"CLIENT: added request {request_id}, queue={DDP_POOL.queue.qsize()}, dict={len(DDP_POOL.dict)}")
    stream_pool = StreamPool(buffer_info)
    buffer_record_id = -1

    while True:
        time_cost = (datetime.datetime.now() - start_tm).total_seconds()
        if time_cost > timeout:
            if buffer_record_id >= 0:
                stream_pool.set_request_to_stop(request_id, buffer_record_id)
            DDP_POOL.dict[request_id] = "expired"
            release_resource(request_id)
            logger.info(f"CLIENT RUNTIME ERROR: req={request_id} 推理超时，无法获取结果。")
            return "RUNTIME ERROR: 推理超时，无法获取结果。"

        if buffer_record_id < 0 or dict_item == "waiting":
            dict_item = DDP_POOL.dict[request_id]

        if isinstance(dict_item, str):
            if dict_item != "waiting" and dict_item.startswith("RUNTIME ERROR"):
                release_resource(request_id)
                logger.info(f"CLIENT RUNTIME ERROR: req={request_id}, msg={dict_item}")
                return dict_item

        if isinstance(dict_item, int):
            # 提交成功后，dict_item变成ddp协程获取到的record_id，之后可以不占据manager锁，直接循环查询reply。
            if dict_item < 0:
                release_resource(request_id)
                logger.info(f"CLIENT RUNTIME ERROR: req={request_id} 无法获取缓存buffer_id")
                return "RUNTIME ERROR: 无法获取缓存buffer_id"
            else:
                start_tm = datetime.datetime.now() # 当开始生成第一帧后，timeout重新开始计时。
                buffer_record_id = dict_item
                _, record_tuple = stream_pool.view_record(rid=buffer_record_id)
                if record_tuple[1] == GenState.END:
                    result = record_tuple[4]
                    release_resource(request_id, buffer_rid=buffer_record_id, stream_pool=stream_pool)
                    logger.info(f"CLIENT: reached eos for {request_id}")
                    break
        else:
            pass
        await asyncio.sleep(0.2)

    logger.info(f"CLIENT: cleaned queue={DDP_POOL.queue.qsize()}, dict={len(DDP_POOL.dict)}")
    return result

async def stream_chat(buffer_info, text_info, hist, gen_kwargs, timeout=120, request_id=None):
    if request_id is None:
        request_id, start_tm = get_hash(text_info[0])
    else:
        start_tm = datetime.datetime.now()
    input_info = [request_id, True, text_info, hist, gen_kwargs]
    DDP_POOL.dict[request_id] = "waiting"
    dict_item = "waiting"
    DDP_POOL.queue.put(input_info, timeout=timeout)
    logger.info(f"CLIENT: added request {request_id}, queue={DDP_POOL.queue.qsize()}, dict={len(DDP_POOL.dict)}")
    stream_pool = StreamPool(buffer_info)
    first_started = False
    prev_res = ""
    buffer_record_id = -1
    while True:
        time_cost = (datetime.datetime.now() - start_tm).total_seconds()
        if time_cost > timeout:
            logger.info(f"CLIENT RUNTIME ERROR: req={request_id} 推理超时，无法获取结果。")
            if buffer_record_id >= 0:
                stream_pool.set_request_to_stop(request_id, buffer_record_id)
            DDP_POOL.dict[request_id] = "expired"
            release_resource(request_id)
            yield ("RUNTIME ERROR: 推理超时，无法获取结果。", "end")
            break

        if buffer_record_id == -1 or dict_item == "waiting":
            dict_item = DDP_POOL.dict[request_id]
        else:
            dict_item = buffer_record_id

        if isinstance(dict_item, str):
            if dict_item != "waiting" and dict_item.startswith("RUNTIME ERROR"):
                release_resource(request_id)
                logger.info(f"CLIENT RUNTIME ERROR: req={request_id}, msg={dict_item}")
                yield (dict_item, "end")
                break

        if isinstance(dict_item, int):
            if dict_item < 0:
                release_resource(request_id)
                logger.info(f"CLIENT RUNTIME ERROR: req={request_id} 无法获取缓存buffer_id")
                yield("RUNTIME ERROR: 无法获取缓存buffer_id", "end")
                break
            else:
                if not first_started:
                    start_tm = datetime.datetime.now() # 当开始生成第一帧后，timeout重新开始计时。
                    first_started = True

                buffer_record_id = dict_item
                _, record_tuple = stream_pool.view_record(rid=buffer_record_id)

                if record_tuple[1] == GenState.END:
                    result = record_tuple[4]
                    current_req_id = record_tuple[3]
                    # 当buffer record检测到不匹配的request_id时，可能是生成loop已经将资源分配给其他请求，那么尝试从DDPWorkers.dict中获取合法结果，之后停止生成。
                    if current_req_id != request_id:
                        logger.info(f"CLIENT WARNING: mismatched request_id {current_req_id}!={request_id}, stop generation.")
                        result = prev_res
                        if request_id in DDP_POOL.dict:
                            maybe_last_res = DDP_POOL.dict[request_id]
                            if isinstance(maybe_last_res, str) and maybe_last_res != "waiting" and not maybe_last_res.startswith("RUNTIME ERROR"):
                                result = maybe_last_res
                    else:
                        release_resource(request_id, buffer_rid=buffer_record_id, stream_pool=stream_pool)
                    logger.info(f"CLIENT: reached eos for {request_id}")
                    yield (result, "end")
                    break
                elif record_tuple[1] == GenState.GENERATING:
                    result = record_tuple[4]
                    current_req_id = record_tuple[3]

                    # 当buffer record检测到不匹配的request_id时，可能是生成loop已经将资源分配给其他请求，那么尝试从DDPWorkers.dict中获取合法结果，之后停止生成。
                    if current_req_id != request_id:
                        logger.info(f"CLIENT WARNING: mismatched request_id {current_req_id}!={request_id}, stop generation.")
                        if request_id in DDP_POOL.dict:
                            maybe_last_res = DDP_POOL.dict[request_id]
                            if isinstance(maybe_last_res, str) and maybe_last_res!="waiting" and not maybe_last_res.startswith("RUNTIME ERROR"):
                                prev_res = maybe_last_res
                        yield (prev_res, "end")
                        break

                    if result != prev_res:
                        yield (result, "generating")
                    prev_res = result
        else:
            pass
        await asyncio.sleep(0.2)

    logger.info(f"CLIENT: cleaned queue={DDP_POOL.queue.qsize()}, dict={len(DDP_POOL.dict)}")

class PooledClient():
    def __init__(self, n_gpus):
        self.n_gpus = n_gpus
        self.pool = ThredPoolExecutWithLimitedQueue(max_workers=n_gpus)
        self.loop = asyncio.get_event_loop()
        self.buffer_info = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.loop.close()

    def set_buffer(self, buffer_info):
        self.buffer_info = buffer_info # StreamPool(buffer_info)
        return

    async def get_reply(self, text, history, gen_kwargs, request_id=None, timeout=120):
        if self.buffer_info is None:
            self.buffer_info = DDP_POOL.extra["buffer_info"]
        t = [self.pool.submit(stream_chat, self.buffer_info, text, history, gen_kwargs, request_id=request_id, timeout=timeout)]
        wait(t, return_when=FIRST_COMPLETED)
        res = t[0].result()
        async for item in res:
            yield item

    async def get_chat(self, text, history, gen_kwargs, request_id=None, timeout=120):
        if self.buffer_info is None:
            self.buffer_info = DDP_POOL.extra["buffer_info"]
        res = await simple_chat(self.buffer_info, text, history, gen_kwargs, request_id=request_id, timeout=timeout)
        return res

    async def stop_generate(self, request_id):
        if self.buffer_info is None:
            self.buffer_info = DDP_POOL.extra["buffer_info"]

        maybe_last_res = "None"
        if request_id in DDP_POOL.dict:
            stream_pool = StreamPool(self.buffer_info)
            maybe_last_res = DDP_POOL.dict[request_id]
            if isinstance(maybe_last_res, int) and maybe_last_res>= 0:
                suc = stream_pool.set_request_to_stop(request_id, int(maybe_last_res))
            else:
                suc = False
        else:
            suc = False
        logger.info(f"CLIENT: Trying to stop request_id={request_id} rid={maybe_last_res}, suc={suc}")
        return "success" if suc else "fail"

pool_client = PooledClient(MAX_QUEUE_LEN)