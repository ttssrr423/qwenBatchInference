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
import os
import multiprocessing as mp
from serving.ddp_worker import infer_worker_loop
from serving.stream_pool import StreamPool

workspace = os.path.dirname(__file__)
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
    timeout_for_clean = 250

    def clean_by_timeout():
        prev_queue_len = -1
        prev_queue_countdown = 3
        prev_qd_sum = -1
        while True:
            # ======== data_id 健康检测 ================
            if "ddp_processes" in DDP_POOL.extra:
                processes_info = [x for x in DDP_POOL.extra["ddp_processes"].split("|") if len(x)>0]
            else:
                processes_info = []
            dead_processes = []
            alive_processes = []
            for process_info_str in processes_info:
                data_id, pid, gpid = process_info_str.split(",")
                cmd = f'ps -aux | grep "{pid}" | grep python'
                str_res = os.popen(cmd).read()
                is_alive = len([x for x in str_res.split("\n") if len(x)>0 and "defunct" not in x]) == 2
                if not is_alive:
                    data_key = f"ddp_process_log{data_id}"
                    if data_key in DDP_POOL.extra:
                        latest_log = DDP_POOL.extra[data_key]
                    else:
                        latest_log = ""
                    logger.error(f"ALIVE KEEPER: DDP{data_id} CRASHED, pid={pid}, latest_action={latest_log}")
                    dead_processes.append((data_id, pid, gpid))

                    core_del_cmd = f"rm {workspace}/core.*"
                    core_rm_res = os.popen(core_del_cmd).read()
                    logger.warning(f"ALIVE KEEPER: executing {core_del_cmd}. res={core_rm_res}")
                else:
                    alive_processes.append(process_info_str)

            DDP_POOL.extra["ddp_processes"] = "|".join(alive_processes)
            # data_id进程重启
            if len(dead_processes) > 0:
                while DDP_POOL.extra.get("ddp_restarting", False):
                    time.sleep(0.2)
                if "tobe_amended" not in DDP_POOL.extra:
                    DDP_POOL.extra["tobe_amended"] = []
                DDP_POOL.extra["tobe_amended"] = DDP_POOL.extra["tobe_amended"] + [int(x[0]) for x in dead_processes]

            cur_queue_len = DDP_POOL.queue.qsize()
            if not cur_queue_len < prev_queue_len and cur_queue_len-len(DDP_POOL.dict) > 0:
                # 当多次dict与queue存在长度差异时，可能存在一些需要被清理的垃圾。
                prev_queue_countdown -= 1
                if prev_queue_countdown <= 0:
                    logger.info("processes not generating for more than 3 clean periods, emptying queue.")
                    queue_looped = {}

                    # 清理存在于queue的，但dict中不存在的
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

            # daemon_dict存储dict中任意样本首次出现的时间，如果dict中存在超过timeout_for_clean的样本，则应该属于漏清理样本，可以删除。
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

            # 打印清理后长度
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
        p = mp.Process(target=infer_worker_loop, args=(i, PIPELINE_PARALLEL_SIZE, NUM_GPUS, async_loops[i], queue, mgr_dict, extra_dict, buffer_info, False, -10))
        p.start()

def amend_ddp_inferer_loop(async_loops, queue, mgr_dict, extra_dict, buffer_info):
    amend_interval = 10.0
    while True:
        if "tobe_amended" not in extra_dict or len(extra_dict["tobe_amended"])==0:
            time.sleep(amend_interval)
            continue
        extra_dict["ddp_restarting"] = True
        amending_data_ids = list(set([x for x in extra_dict["tobe_amended"]]))
        next_loading_seq = amending_data_ids[1:] + [-1]
        seq_indices = list(range(len(amending_data_ids)))
        logger.info(f"ALIVE KEEPER: restarting {len(amending_data_ids)} processes: {amending_data_ids}")
        for seq_id, data_id, next_load_id in zip(seq_indices, amending_data_ids, next_loading_seq):
            p = mp.Process(target=infer_worker_loop, args=(data_id, PIPELINE_PARALLEL_SIZE, NUM_GPUS, async_loops[data_id], queue, mgr_dict, extra_dict, buffer_info, seq_id==0, next_load_id))
            p.start()
        extra_dict["tobe_amended"] = []
        extra_dict["ddp_restarting"] = False
        time.sleep(2.0)

# def raise_io_limit():
#     cmd = "ulimit -SHn 65535"
#     str_res = os.popen(cmd).read()
#     logger.info(f"io limit raised: {str_res}")
#     return

global GLOB_RES_BUFFER
global GLOB_TOKEN_BUFFER

if __name__ == "__main__":
    logger.info('main pid is %s' % os.getpid())
    # raise_io_limit()

    server_process = start_server_process()
    clean_process = manager_cleaner_start()

    # 所有卡同时处理的样本总量翻32倍，因为StreamPool不是线程安全的，大一些有助于避免两个线程同时给一个record_id添加占用标记。
    max_processing_examples = int(DATA_PARALLEL_SIZE * GLOBAL_CONFIG["max_dynamic_bsz"] * 32)
    buffer, glob_res_buffer, glob_token_buffer = StreamPool.create_buffer(max_processing_examples, GLOBAL_CONFIG["max_length"], 256)
    global GLOB_RES_BUFFER
    GLOB_RES_BUFFER = glob_res_buffer
    global GLOB_TOKEN_BUFFER
    GLOB_TOKEN_BUFFER = glob_token_buffer
    DDP_POOL.set_buffer(buffer)
    DDP_POOL.extra["buffer_info"] = buffer

    process_loops = [asyncio.new_event_loop() for _ in range(int(DATA_PARALLEL_SIZE))]
    start_ddp_inferer(process_loops, DDP_POOL.queue, DDP_POOL.dict, DDP_POOL.extra, DDP_POOL.buffer_info)
    amend_ddp_inferer_loop(process_loops, DDP_POOL.queue, DDP_POOL.dict, DDP_POOL.extra, DDP_POOL.buffer_info)

    server_process.join()
    clean_process.join()

    GLOB_RES_BUFFER.unlink()
    GLOB_TOKEN_BUFFER.unlink()
