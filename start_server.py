import uvicorn
from serving.app import app
import multiprocessing
import time
import requests
from global_config import DATA_PARALLEL_SIZE, PIPELINE_PARALLEL_SIZE, NUM_GPUS, PORT, DDP_POOL, GLOBAL_CONFIG
import logging
import datetime
import sys
import asyncio
import multiprocessing as mp
from serving.ddp_worker import infer_worker_loop
from serving.stream_pool import StreamPool
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
    level="INFO"
)
logger = logging.getLogger(__name__)

def start_server_process():
    def run_server():
        host = "0.0.0.0"
        port = PORT
        uvicorn.run(app, host=host, port=port)

    p = multiprocessing.Process(target=run_server, args=())
    p.start()
    return p

def manager_cleaner_start():
    clean_interval = 10.0
    timeout_for_clean = 200

    def clean_by_timeout():
        prev_queue_len = -1
        prev_queue_countdown = 3
        prev_qd_sum = -1
        while True:
            cur_queue_len = DDP_POOL.queue.qsize()
            if not cur_queue_len < prev_queue_len and cur_queue_len-len(DDP_POOL.dict) > 0:
                # when processing queue is always longer than returning dict, maybe the queue got stuck.
                prev_queue_countdown -= 1
                if prev_queue_countdown <= 0:
                    logger.info("processes not generating for more than 3 clean periods, emptying queue.")
                    queue_looped = {}
                    while not DDP_POOL.queue.empty():
                        inp_info = DDP_POOL.queue.get()
                        req_id = inp_info[0]
                        if len(inp_info) > 0 and req_id in DDP_POOL.daemon_dict and req_id in DDP_POOL.dict:
                            if req_id not in queue_looped:
                                logger.info(f"putting back to queue: {req_id}")
                                DDP_POOL.queue.put(inp_info)
                                queue_looped[req_id] = True
                            else:
                                break
                        else:
                            logger.info(f"removing request from queue: {req_id if len(inp_info) > 0 else 'unk'}")
                            del inp_info
                    prev_queue_countdown = 3
                    prev_queue_len = -1
            prev_queue_len = int(cur_queue_len)

            for _k, _v in list(DDP_POOL.daemon_dict.items()):
                td = (datetime.datetime.now() - _v).total_seconds()
                if td > timeout_for_clean:
                    del DDP_POOL.daemon_dict[_k]
                    if _k in DDP_POOL.dict:
                        logger.info(f"removing dict due to timeout: {_k}")
                        del DDP_POOL.dict[_k]
                elif _k not in DDP_POOL.dict:
                    del DDP_POOL.daemon_dict[_k]

            existed_running = list(DDP_POOL.dict.keys())
            for _k2 in existed_running:
                if _k2 not in DDP_POOL.daemon_dict:
                    DDP_POOL.daemon_dict[_k2] = datetime.datetime.now()

            q_size = DDP_POOL.queue.qsize()
            d_size = len(DDP_POOL.dict)
            if (q_size > 0 or d_size > 0) or prev_qd_sum > 0:
                logger.info(f"after daemon cleaner queue size: {q_size}, dict size: {d_size}")
            time.sleep(clean_interval)
            prev_qd_sum = q_size + d_size
    p = multiprocessing.Process(target=clean_by_timeout, args=())
    p.start()
    logger.info("cleaning process started...")
    return p

def start_ddp_inferer(async_loops, queue, mgr_dict, extra_dict, buffer_info):
    # mp.spawn(main, nprocs=DATA_PARALLEL_SIZE, args=(queue, mgr_dict, extra_dict, buffer_info))
    for i in range(int(DATA_PARALLEL_SIZE)):
        p = mp.Process(target=infer_worker_loop, args=(i, PIPELINE_PARALLEL_SIZE, NUM_GPUS, async_loops[i], queue, mgr_dict, extra_dict, buffer_info))
        p.start()

if __name__ == "__main__":
    server_process = start_server_process()
    clean_process = manager_cleaner_start()

    # 所有卡同时处理的样本总量翻8倍，因为StreamPool不是线程安全的，大一些有助于避免两个线程同时给一个record_id添加占用标记。
    max_processing_examples = int(DATA_PARALLEL_SIZE * GLOBAL_CONFIG["max_dynamic_bsz"] * 8)
    buffer = StreamPool.create_buffer(max_processing_examples, GLOBAL_CONFIG["max_length"], 256)
    DDP_POOL.set_buffer(buffer)
    DDP_POOL.extra["buffer_info"] = buffer

    process_loops = [asyncio.new_event_loop() for _ in range(int(DATA_PARALLEL_SIZE))]
    start_ddp_inferer(process_loops, DDP_POOL.queue, DDP_POOL.dict, DDP_POOL.extra, DDP_POOL.buffer_info)
    server_process.join()
    clean_process.join()