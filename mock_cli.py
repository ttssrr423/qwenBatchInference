import json
# import websocket    # pip install websocket-client
from multiprocessing import Process
import datetime
import random

LOCAL_URL = "http://127.0.0.1:8081"
PARALLEL_NUM = 16
ROUND_NUM = 5
USE_STREAM = True
OUT_FILE = None #"test_results.txt" # None # "test_results.txt"

def single_id_request(_id):
    import requests
    headers = {"Content-Type": "application/json", "cache-control": "no-cache"}
    data = {"request_id": "REQ"+str(_id)}
    res = requests.post("http://127.0.0.1:8081/get_from_buffer", data=json.dumps(data), headers=headers)
    buffer_res = res.json()
    print("late_fetching: ", buffer_res)
    return

def chat_request():
    import requests
    headers = {"Content-Type":"application/json", "cache-control":"no-cache"}
    data = {"query":"你可以做什么？", "history":[["你好", "你好，我是小PAI，是人工智能机器人。"]]}
    res = requests.post(LOCAL_URL+"/chat", data=json.dumps(data), headers=headers)
    print(res.json())

import requests
import random
def stream_post(inps):
    req_id, out_list, res_dict = inps[0], inps[1], inps[2]

    headers = {"Content-Type":"application/json", "cache-control":"no-cache"}
    data = {"query":"你可以做什么？", "history":[["你好", "你好，我是小PAI，是人工智能机器人。"]],"request_id":"REQ"+str(req_id)} # "request_id":"REQ"+str(req_id)
    # data = {"query": "那你明年多大啊？", "history": [["你好，你多大了", f"我{req_id+1}岁啦！"]]}  # "request_id":"REQ"+str(req_id)
    # adapter_name = "skip" if random.random() < 0.5 else "default"
    adapter_name = "skip"
    return_lgt = False # True if random.random() > 0.5 else False
    data.update({"gen_kwargs":{"max_length":256, "temperature":0.01, "adapter_name": adapter_name, "return_logits":return_lgt}})
    print(f"submitted: {req_id}")
    stream_res = requests.post(LOCAL_URL+"/stream_chat_post", data=json.dumps(data), stream=True) # headers=headers,
    msg_ct = 0
    for res_raw in stream_res.iter_lines():
        # print(res_raw)
        # ============== json_str yield ===============
        if res_raw.startswith(b'data:'):
            res_ent = json.loads(res_raw[5:])
        else:
            res_ent = {}
        if req_id not in res_dict: res_dict[req_id] = []
        if len(res_ent) > 0 and "text" in res_ent:
            res_dict[req_id] = res_dict[req_id] + [res_ent["text"]]

        if "flag" in res_ent and res_ent["flag"] == "end":
            len_text = len(res_ent["text"])
            flat_text = res_ent["text"].replace("\n", "\\n")
            print(f"Reply Text(len={len_text}): {flat_text}")
            out_list.append(len_text)
        msg_ct += 1
        continue
    return 1

def chat_post(inps):
    req_id, out_list, res_dict = inps[0], inps[1], inps[2]
    headers = {"Content-Type":"application/json", "cache-control":"no-cache"}
    data0 = {"query":"你可以做什么？", "history":[["你好", "你好，我是小PAI，是人工智能机器人。"]], "request_id":"REQ"+str(req_id)} # "request_id":"REQ"+str(req_id)
    data1 = {"query": f"今天是星期三，{req_id}天之后是星期几？", "history":[], "request_id": "REQ"+str(req_id)} # "request_id":"REQ"+str(req_id)
    data2 = {"query": "随便讲个笑话。", "history": [],
            "request_id": "REQ" + str(req_id)}  # "request_id":"REQ"+str(req_id)
    data3 = {"query": "为什么地球是圆的？", "history": [],
            "request_id": "REQ" + str(req_id)}  # "request_id":"REQ"+str(req_id)
    task_map = {0:data0, 1:data1, 2:data2, 3:data3}
    # data = task_map[int(req_id) % 4]
    data = task_map[0]
    lora_name = "skip"
    return_lgt = False # True if random.random() > 0.5 else False
    # max_length = random.randint(1000, 1200)
    max_length = 256
    data.update({"gen_kwargs":{"max_length":max_length, "temperature":0.01, "adapter_name": lora_name}})
    
    req_res = requests.post(LOCAL_URL+"/chat", data=json.dumps(data))
    try:
        ent = json.loads(req_res.text)
        resp_text = ent["response"]
        print(req_id, resp_text.replace("\n", "<s>"))
        len_text = len(resp_text)
        out_list.append(len_text)
    
        if OUT_FILE is not None:
            with open(OUT_FILE, mode="a") as fw:
                fw.write(json.dumps({"id":req_id, "text":resp_text}, ensure_ascii=False)+"\n")
    except Exception as ex:
        print(ex)
    return 1

import time
def dummy_method(id):
    print(id)
    time.sleep(1.0)
    return 2

from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED
def stream_pool_request(round_num=1, parallel_num=1):

    total_reqs = round_num * parallel_num
    from multiprocessing import Manager
    manager = Manager()
    out_list = manager.list([])
    stream_res_counter = manager.dict({})

    overall_textlen = 0
    overall_timesum = 0.0

    # executor = ThreadPoolExecutor(max_workers=parallel_num)
    print(f"total_requests={total_reqs}")
    with ThreadPoolExecutor(max_workers=parallel_num) as executor:
        t0 = datetime.datetime.now()
        # tasks = [executor.submit(stream_post, (i, out_list, stream_res_counter)) for i in range(total_reqs)]
        if USE_STREAM:
            tasks = [executor.submit(stream_post, tuple((i, out_list, stream_res_counter))) for i in range(total_reqs)]
        else:
            tasks = [executor.submit(chat_post, tuple((i, out_list, stream_res_counter))) for i in range(total_reqs)]
        for future in as_completed(tasks):
            pass

        batch_time_sum = (datetime.datetime.now()-t0).total_seconds()
        textlens = out_list
        batch_textlen = sum(textlens)
        overall_textlen += batch_textlen
        overall_timesum += batch_time_sum
        empty_res_num = 0
        fail_res_num = 0
        for _id in range(parallel_num):
            if _id in stream_res_counter:
                frame_num = len(stream_res_counter[_id])
                frame_textlens = [len(x) for x in stream_res_counter[_id]]
                # print("frames and their textlen:", _id, frame_num, frame_textlens)
                if (frame_num == 1 and frame_textlens[0] == 0):
                    empty_res_num += 1
                if (frame_num >= 1 and stream_res_counter[_id][-1].startswith("RUNTIME ERROR")):
                    fail_res_num +=1
            else:
                # print("frames:", _id, 0)
                empty_res_num += 1
            # if len(stream_res_counter[_id]) < 50:
            # if len(stream_res_counter[_id]) < 1:
            #     single_id_request(_id)
        round_num -= 1
        print(f"req_nums={parallel_num}, empty_result={empty_res_num}, failed_result={fail_res_num}")
        print(f"batch example num:{len(textlens)}, in {batch_time_sum} secs. lens={textlens}")
        print("char per sec: ", batch_textlen/batch_time_sum)

        print("=======final char per sec stat ========")
        print(overall_textlen/overall_timesum)

if __name__ == "__main__":  # confirms that the code is under main function
    stream_pool_request(parallel_num=PARALLEL_NUM, round_num=ROUND_NUM)
