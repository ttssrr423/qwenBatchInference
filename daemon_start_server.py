import os
import torch
import requests
import time
from global_config import PORT, GLOBAL_CONFIG
import logging
import datetime
import sys

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
    level="INFO"
)
logger = logging.getLogger(__name__)

daemon_file_path = "daemon.log"
log_file_path = GLOBAL_CONFIG["log_file"] # "/var/log/backend.log"

def if_port_occupied():
    cmd = 'netstat -anp | grep ' + str(PORT)
    str_res = os.popen(cmd).read()
    if (":"+str(PORT)) in str_res and "LISTEN" in str_res:
        return True
    return False

def if_gpus_occupied(mem_threshold=4000, num_threshold=0.5):
    """
    不超过50%的总gpu的显存占用小于4000Mib时，判定为backend推理服务失效，需要重启服务。
    """

    cmd = 'nvidia-smi | grep "MiB /"'
    str_res = os.popen(cmd).read()
    usage_lines = [x.split("|")[2] for x in str_res.split("\n") if len(x.split("|")) > 2]
    gpu_ct = int(torch.cuda.device_count())
    occupied = [0 for _ in range(gpu_ct)]
    if len(usage_lines) == gpu_ct:
        for dev_id, usage_info in enumerate(usage_lines):
            mem_alloc = int(usage_info.split("MiB /")[0].strip())
            if mem_alloc > mem_threshold:
                occupied[dev_id] = 1
    if sum(occupied) <= (gpu_ct * num_threshold):
        return False
    return True

def write_daemon(string_info, is_info=True):
    if is_info:
        logger.info(string_info)
    else:
        logger.error(string_info)

    with open(daemon_file_path, mode="a", encoding="utf8") as fw:
        fw.write(str(datetime.datetime.now())+"\t" + string_info + "\n")
    return


def run_clean():
    show_script = "ps -ef | grep python | grep -v grep | grep -v daemon_start_server"
    killing_lines = os.popen(show_script).read()
    lines = killing_lines.split("\n")
    print(lines)
    if len(lines) > 0 and len(lines[0]) > 0:
        clean_script = "ps -ef | grep python | grep -v grep | grep -v daemon_start_server | awk '{print $2}' | xargs kill -9"
        str_res = os.popen(clean_script).read()
    return

def run_start(log_path):
    run_script = f"nohup python start_server.py > {log_path} 2>&1 &"
    str_res = os.popen(run_script).read()
    print(str_res)
    str_res2 = os.popen("\n").read()
    return

def loop():
    interval = 30.0
    write_daemon("starting service daemon loop.")
    is_first = True
    while True:
        if not if_gpus_occupied():
            write_daemon("no running service detected, restarting service.", is_first)
            run_clean()
            time.sleep(0.5)
            run_start(log_file_path)
            write_daemon("run stript started...")
            time.sleep(5.0)
            write_daemon(f"port_occupied={if_port_occupied()}")
            write_daemon(f"gpu_occupied={if_gpus_occupied()}")

            while (not if_port_occupied()):
                time.sleep(1.0)
            while (not if_gpus_occupied()):
                time.sleep(interval)
            write_daemon("restart service finished...")
            is_first=False

        time.sleep(interval)

if __name__ == "__main__":
    loop()