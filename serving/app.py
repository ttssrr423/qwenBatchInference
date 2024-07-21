import sys
import asyncio
import logging
import hashlib
import datetime
import random
import json
import time
import threading
from global_config import PORT
from fastapi import Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
from fastapi import Body, FastAPI, File, Form, Query, UploadFile, WebSocket
from serving.liteqwen_inferer import get_inferer


import nest_asyncio
nest_asyncio.apply()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
    level="INFO"
)
logger = logging.getLogger(__name__)

app = FastAPI()
setattr(app, "fail_ct", 0)
setattr(app, "model_init_finished", False)

def get_hash(_text):
    hasher = hashlib.md5()
    tm = datetime.datetime.now()
    rand_num = str(random.random())[-4:]
    text_with_time = _text + tm.strftime("%Y-%m-%d %H:%M:%S.%f") + rand_num
    hasher.update(text_with_time.encode(encoding="utf-8"))
    return hasher.hexdigest(), tm

global_inferer = None
inferer_loading_lock = False
def get_global_inferer():
    global inferer_loading_lock
    while (inferer_loading_lock):
        time.sleep(1.0)

    global global_inferer
    if global_inferer is None:
        if not inferer_loading_lock:
            inferer_loading_lock = True
        global_inferer = get_inferer()
        print("liteqwen loading finished, unblock python requests...")
        inferer_loading_lock = False
    return global_inferer

@app.get("/_init")
async def init_inferer():
    thread = threading.Thread(target=get_global_inferer, daemon=True)
    thread.start()
    # get_global_inferer()
    return 1

@app.post("/stream_chat_post")
async def stream_chat_post(request: Request):
    try:
        # json_post_raw = request.json()
        # json_str_list = loop.run_until_complete(asyncio.gather(json_post_raw, loop=loop))
        # logger.info((f"raw request: {json_str_list}").replace("\n", "\\n"))
        # json_post = json.dumps(json_str_list[0])

        json_post_raw = await request.json()
        json_post = json.dumps(json_post_raw)
        json_post_list = json.loads(json_post)
        logger.info("running until complete request json:" + str(json_post_list))
        prompt = json_post_list.get('query')
        history = json_post_list.get('history')
        role = json_post_list.get('role', "user")
        metadata = json_post_list.get('metadata', "")
        gen_kwargs = json_post_list.get("gen_kwargs", {})
        request_id = "UNK" if "request_id" not in json_post_list else str(json_post_list["request_id"])
    except Exception as ex:
        logger.error(f"error during request decoding: {ex}, body={request.body()}")
        resp_entity = {"text": "",
                       "delta": "",
                       "code": -1,
                       "flag": "error"}
        def err_gen():
            for item in [resp_entity]:
                yield item

        async def err_decorate(_generator):
            for partial_sent_entity in _generator:
                partial_sent_json = json.dumps(partial_sent_entity, ensure_ascii=False)
                yield ServerSentEvent(partial_sent_json, event="delta")
            logger.info(f"generation finished.")

        return EventSourceResponse(err_decorate(err_gen()))

    if request_id == "UNK":
        request_id, start_tm = get_hash(prompt)
    else:
        _, start_tm = get_hash(prompt)
    
    inferer = get_global_inferer()
    tuple_generator = inferer.get_stream_reply([prompt, metadata, role], history, gen_kwargs, request_id=request_id)

    async def decorate():
        prev_res = ""
        async for rep_text, logits_info, flag in tuple_generator:
            if rep_text.startswith("RUNTIME ERROR:"):
                code = -1
                text = ""
                delta = rep_text
                app.fail_ct += 1
            else:
                code = 200
                delta = rep_text[len(prev_res):]
                text = rep_text
                app.fail_ct = 0
            resp_entity = {"text": text,
                           "delta": delta,
                           "logits": logits_info,
                           "code": code,
                           "flag": flag}
            resp_json = json.dumps(resp_entity)
            yield ServerSentEvent(resp_json, event="delta")
            prev_res = rep_text

    return EventSourceResponse(decorate(), headers={"Access-Control-Allow-Origin":"*"})


@app.post("/chat")
async def normal_chat(request: Request):
    try:
        json_post_raw = await request.json()
        logger.info(f"RAW REQUEST: {json_post_raw}")
        json_post = json.dumps(json_post_raw)
        json_post_list = json.loads(json_post)
        prompt = json_post_list.get('query')
        history = json_post_list.get('history')
        role = json_post_list.get('role', "user")
        metadata = json_post_list.get('metadata', "")
        gen_kwargs = json_post_list.get("gen_kwargs", {})
        request_id = "UNK" if "request_id" not in json_post_list else str(json_post_list["request_id"])
    except Exception as ex:
        logger.error(f"error during request decoding: {ex}, body={request.body()}")
        answer = {
            "response": "",
            "history": [],
            "code": -1,
            "flag": "invalid inputs"
        }
        return answer

    if request_id == "UNK":
        request_id, start_tm = get_hash(prompt)
    else:
        _, start_tm = get_hash(prompt)

    inferer = get_global_inferer()

    result, logits_info = await inferer.get_chat([prompt, metadata, role], history, gen_kwargs, request_id=request_id)
    if result.startswith("RUNTIME ERROR:"):
        answer = {
            "response": "",
            "history": [],
            "code": -1,
            "logits": None,
            "flag": result
        }
        app.fail_ct += 1
    else:
        answer = {
            "response": result,
            "history": [],
            "code": 200,
            "logits": logits_info,
            "flag": "end"
        }
        app.fail_ct = 0
    return answer

@app.get("/health")
async def health(request: Request):
    if app.fail_ct < 5:
        data = {"status": "UP"}
        return data
    raise Exception("consecutive fail count exceeding threshold, possibly caused by inference loop failure.")