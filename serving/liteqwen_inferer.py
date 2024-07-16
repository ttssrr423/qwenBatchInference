import asyncio

import transformers

from global_config import GLOBAL_CONFIG, WORLD_SIZE, PIPELINE_PARALLEL_SIZE, DATA_PARALLEL_SIZE
from qwen2 import Qwen2ForCausalLM, Qwen2Config, Qwen2TokenizerFast
from transformers.generation.utils import LogitsProcessorList
import time
import json
import sys
import random
import traceback
import datetime
from liteqwen_py import connector
import logging
import re
import os
os.environ.setdefault("LITEQWEN_MASK_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "mask_templates"))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
    level="INFO"
)
logger = logging.getLogger(__name__)


def prepare_kwargs_and_lora(self, req_id, _inferer, gen_kwargs, use_lora, input_ids_len):

    adapter_name_result = "skip"

    should_skip_lora = gen_kwargs.pop("skip_lora", False)
    using_adapter_name = gen_kwargs.pop("adapter_name", "skip")

    if use_lora and should_skip_lora:
        if hasattr(_inferer, "current_lora_name") and _inferer.current_lora_name == "skip":
            pass
        else:
            if logger is not None:
                logger.info(f"setting lora to skip after {req_id}")
            adapter_name_result = "skip"
    elif use_lora:
        if not hasattr(_inferer, "current_lora_name"):
            _inferer.current_lora_name = "skip"
        if _inferer.current_lora_name != using_adapter_name:
            if logger is not None:
                logger.info(f"setting lora to {using_adapter_name} after {req_id}")
        adapter_name_result = using_adapter_name

    top_p = GLOBAL_CONFIG["top_p"]
    temperature = GLOBAL_CONFIG["temperature"]


    no_english = gen_kwargs.pop("no_english", False)
    if no_english:
        logits_mask = self.english_mask
    else:
        logits_mask = self.null_mask
    return_logits = gen_kwargs.pop("return_logits", False)

    default_gen_kwargs = {"top_p": top_p,
                          "temperature": temperature,
                          "logits_mask": logits_mask,
                          "return_logits": return_logits}

    default_gen_kwargs.update(gen_kwargs)

    max_length_default = GLOBAL_CONFIG["max_length"]
    if "max_new_tokens" in gen_kwargs:
        default_gen_kwargs.update({"max_new_tokens": gen_kwargs["max_new_tokens"]})
    elif "max_length" in gen_kwargs:
        default_gen_kwargs.update({"max_length": min(max_length_default, gen_kwargs["max_length"])})
    else:
        default_gen_kwargs.update({"max_length":max_length_default})

    return adapter_name_result, default_gen_kwargs

def convert_to_chatglm3_history(v2_history):
    new_hist = []
    for conv_round in v2_history:
        new_hist.append({"role": "user", "content": conv_round[0]})
        new_hist.append({"role": "assistant", "metadata": "", "content": conv_round[1]})
    return new_hist

# valid_v3_keys = {"role":None, "content":"", "metadata":"", "tools":""}
def check_v3_history(v3_history):
    for item in v3_history:
        if not isinstance(item, dict):
            return False
        if "role" not in item or "content" not in item:
            return False
    return True

# def glm3_postprocess(output):
#     content = ""
#     for response in output.split("<|assistant|>"):
#         metadata, content = response.split("\n", maxsplit=1)
#         if not metadata.strip():
#             content = content.strip()
#             # new_hist = {"role": "assistant", "metadata": metadata, "content": content}
#             content = content.replace("[[训练时间]]", "2023年")
#         else:
#             # new_hist = {"role": "assistant", "metadata": metadata, "content": content}
#             content = json.dumps({"name": metadata.strip(), "content": content}, ensure_ascii=False)
#     return content

class LiteqwenInferer():
    def __init__(self, world_size, data_parallel_size, pipeline_parallel_size, model_path, lora_paths,
                max_dynamic_bsz, max_sequence_length, top_p, top_k, default_temperature, use_lora, thread_data_id=-1, token_smem_info=None):
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(model_path)
        use_lora = use_lora and len(lora_paths) > 0
        self.use_lora = use_lora
        self.top_p = top_p
        self.top_k = int(top_k)
        self.temperature = default_temperature
        self.max_dynamic_bsz = int(max_dynamic_bsz)
        self.max_sequence_length = int(max_sequence_length)
        self.data_parallel_size = int(data_parallel_size)
        self.pipeline_parallel_size = int(pipeline_parallel_size)
        self.world_size = int(world_size)
        self.current_lora_name = "skip"
        self.data_id = thread_data_id
        if thread_data_id < 0:
            logger.info("PYTHON ERROR: DDP should start a data_parallel thread with data_id>=0")

        self.english_mask = {"default":0.01, "ids":[-1], "value":0.0} # 加载scripts/mask_templates/logits_masks.json里的template_id=-1的模板。生成过程如下：
        """
        all_tokens = [(i, self.tokenizer.convert_ids_to_tokens([i])[0]) for i in range(self.tokenizer.vocab_size)]
        english_ids = []
        for tk_id, tk in all_tokens:
            if len(tk) > 0 and "[" not in tk:
                remaining = re.sub('[▁|a-z|A-Z]+', "", tk)
                if remaining == "" and tk != "▁":
                    english_ids.append(tk_id)
        json.dumps([{"name": "no_english", "template_id": -1, "default": 0.0, "value": -4096.0, "ids": english_ids}])
        """
        self.null_mask = {"default":0.0, "ids":[], "value":0.0}

        self.liteqwen_connector = connector.Qwen2Inferer(model_path, self.world_size, self.data_parallel_size, self.pipeline_parallel_size, self.max_dynamic_bsz,
                                            self.max_sequence_length, top_p=top_p, top_k=self.top_k, temperature=default_temperature, use_lora=use_lora, token_smem_info=token_smem_info)
        self.liteqwen_connector.load_base_model_params(model_path)
        for item in lora_paths:
            if self.use_lora:
                self.liteqwen_connector.load_lora_params(item["name"], item["path"])
        # self.liteqwen_connector.start_loops()
        self.liteqwen_connector.start_single_thread_loop(thread_data_id)

        return

    async def get_stream_reply(self, input_info, history, gen_kwargs, request_id=None):
        text, inp_metadata, inp_role = input_info
        if request_id is None:
            request_id = "rand_request" + str(random.randint(0, 9999))

        valid_request = True
        if inp_role is None or inp_role not in ["user", "assistant", "system", "observation"]:
            inp_role = "user"
        if history is None:
            history = []
        elif len(history) > 0:
            if isinstance(history[0], list) and len(history[0])==2 and isinstance(history[0][0], str):
                history = convert_to_chatglm3_history(history)
            elif isinstance(history[0], dict):
                valid_history = check_v3_history(history)
                if not valid_history:
                    logger.warning(f"invalid V3 history for req_id={request_id}, skipping request. hist={history}")
                    yield "RUNTIME ERROR: 输入history格式有误。", None, "end"
                    valid_request = False
            else:
                logger.warning(f"invalid V3 history for req_id={request_id}, skipping request. hist={history}")
                yield "RUNTIME ERROR: 输入history格式有误。", None, "end"
                valid_request = False

        return_raw = True # bool(gen_kwargs.pop("return_raw", True))
        seed = int(gen_kwargs.pop("seed", -1)) # 默认随机
        top_k = max(min(gen_kwargs.pop("topk", self.top_k), 64), 1)
        try:
            # inputs = self.tokenizer.build_chat_input(text, history=history, role=inp_role)
            message_list = history + [{"role":inp_role, "content":text}]
            completed_text = self.tokenizer.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True)

            if "prefix_text" in gen_kwargs:
                completed_text = completed_text + str(gen_kwargs["prefix_text"])
            inputs = self.tokenizer([completed_text], return_tensors="pt").to("cpu")
            
            inp_len = int(inputs["input_ids"].shape[1])
            if inp_len > (GLOBAL_CONFIG["max_length"] - 20):
                yield "RUNTIME ERROR: 输入太长，距离max_length不足20个token，无法生成。", None, "end"
                valid_request = False

            input_id_list = [int(x) for x in list(inputs["input_ids"][0].cpu().numpy())]
            lora_name_using, hf_gen_kwargs = prepare_kwargs_and_lora(self, request_id, self, gen_kwargs, self.use_lora, inp_len)
            flat_text = text.replace("\n", "<s>")
            logger.info(f"START GENERATION stream=True, req_id={request_id}, role={inp_role}, query={flat_text}, hist_len={len(history)}, gen_kwargs={hf_gen_kwargs}")
        except Exception as ex:
            traceback.print_exc()
            logger.error(f"ERROR on req_id={request_id} during prompt build, gen_kwargs preparation or lora selection: {ex}")
            yield "RUNTIME ERROR: "+str(ex), None, "end"
            valid_request = False

        if valid_request:
            hf_gen_kwargs.update({"seed": seed, "top_k": top_k, "adapter_name": lora_name_using})
            suc = self.liteqwen_connector.submit(request_id, input_id_list, gen_config=hf_gen_kwargs)
            if suc == 0:
                yield "RUNTIME ERROR: " + "排队超时，请稍后再提交请求。", None, "end"
            else:
                submitted_tm = datetime.datetime.now()
                first_frame_cost = -1.0
                success_frame_ct = 0
                accumulating_ids = []
                await asyncio.sleep(0.5)
                while True:
                    resp = self.liteqwen_connector.get_generated(request_id, return_logits=hf_gen_kwargs["return_logits"])
                    logits_info = resp["logits"] if "logits" in resp else None
                    if resp["status"] == 0:
                        await asyncio.sleep(0.2)
                        continue
                    elif resp["status"] == 2:
                        res_id_list = list(resp["token_ids"])
                        # final_full_text = self.tokenizer.decode(res_id_list)
                        final_full_text = self.tokenizer.decode(res_id_list, skip_spetial_tokens=True)
                        if len(final_full_text) > 10 and final_full_text[-10:] == "<|im_end|>":
                            final_full_text = final_full_text[:-10]
                        success_frame_ct += 1
                        metadata_splited = final_full_text.split("\n")
                        # if not return_raw and len(metadata_splited) > 1:
                        #     final_full_text = final_full_text[len(metadata_splited[0])+1:] # 去除metadata
                        flat_reply = final_full_text.replace("\n", "<s>")
                        time_cost = (datetime.datetime.now() - submitted_tm).total_seconds()
                        logger.info(f"REPLY for request_id={request_id} first/total=({first_frame_cost}/{time_cost}) secs, inp/rep={len(input_id_list)}/{len(res_id_list)} tokens; text={flat_reply}")
                        yield final_full_text, logits_info, "end"
                        break
                    elif resp["status"] == -1:
                        yield "RUNTIME ERROR: liteqwen error with status -1", None, "end"
                        break
                    else:
                        delta_tokens = list(resp["token_ids"])
                        accumulating_ids.extend(delta_tokens)
                        frame_text = self.tokenizer.decode(accumulating_ids, skip_spetial_tokens=True)
                        if len(frame_text) > 10 and frame_text[-10:] == "<|im_end|>":
                            frame_text = frame_text[:-10]
                        success_frame_ct += 1
                        metadata_splited = frame_text.split("\n")
                        if not return_raw and len(metadata_splited) > 1:
                            frame_text = frame_text[len(metadata_splited[0])+1:] # 去除metadata
                        if first_frame_cost < 0.0:
                            first_frame_cost = (datetime.datetime.now() - submitted_tm).total_seconds()
                        yield frame_text, logits_info, "generating"

                    await asyncio.sleep(0.1)
                    if success_frame_ct > self.max_sequence_length:
                        print("max success frame received, exiting receiving loop.")
                        break

    async def get_chat(self, input_info, history, gen_kwargs, request_id=None):
        text, inp_metadata, inp_role = input_info
        if request_id is None:
            request_id = "rand_request" + str(random.randint(0, 9999))

        if inp_role is None or inp_role not in ["user", "assistant", "system", "observation"]:
            inp_role = "user"
        if history is None:
            history = []
        elif len(history) > 0:
            if isinstance(history[0], list) and len(history[0])==2 and isinstance(history[0][0], str):
                history = convert_to_chatglm3_history(history)
            elif isinstance(history[0], dict):
                valid_history = check_v3_history(history)
                if not valid_history:
                    logger.warning(f"invalid V3 history for req_id={request_id}, skipping request. hist={history}")
                    return "RUNTIME ERROR: 输入history格式有误。", None
            else:
                logger.warning(f"invalid V3 history for req_id={request_id}, skipping request. hist={history}")
                return "RUNTIME ERROR: 输入history格式有误。", None

        return_raw = True # bool(gen_kwargs.pop("return_raw", True))
        seed = int(gen_kwargs.pop("seed", -1)) # 默认随机
        top_k = max(min(gen_kwargs.pop("topk", self.top_k), 64), 1)

        try:
            message_list = history + [{"role":inp_role, "content":text}]
            completed_text = self.tokenizer.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True)

            if "prefix_text" in gen_kwargs:
                completed_text = completed_text + str(gen_kwargs["prefix_text"])
            inputs = self.tokenizer([completed_text], return_tensors="pt").to("cpu")
            inp_len = int(inputs["input_ids"].shape[1])
            if inp_len > (GLOBAL_CONFIG["max_length"] - 20):
                return "RUNTIME ERROR: 输入太长，距离max_length不足20个token，无法生成。", None

            input_id_list = [int(x) for x in list(inputs["input_ids"][0].cpu().numpy())]
            lora_name_using, hf_gen_kwargs = prepare_kwargs_and_lora(self, request_id, self, gen_kwargs, self.use_lora, inp_len)
            flat_text = text.replace("\n", "<s>")
            logger.info(f"START GENERATION stream=True, req_id={request_id}, role={inp_role}, query={flat_text}, hist_len={len(history)}, gen_kwargs={hf_gen_kwargs}")
        except Exception as ex:
            traceback.print_exc()
            logger.error(f"ERROR on req_id={request_id} during prompt build, gen_kwargs preparation or lora selection: {ex}")
            return "RUNTIME ERROR: "+str(ex), None

        hf_gen_kwargs.update({"seed": seed, "top_k": top_k, "adapter_name": lora_name_using})
        suc = self.liteqwen_connector.submit(request_id, input_id_list, gen_config=hf_gen_kwargs)
        if suc == 0:
            return "RUNTIME ERROR: 排队超时，请稍后再提交请求。", None

        success_frame_ct = 0
        accumulating_ids = []
        unfinished_text = ""
        await asyncio.sleep(0.2) # 这里延迟不能去除，需要等c++队列添加或移动的锁，太快请求get_generate可能导致status=-1，之后请求才加入队列成功。
        while True:
            print("connector getting...")
            resp = self.liteqwen_connector.get_generated(request_id, return_logits=hf_gen_kwargs["return_logits"])
            logits_info = resp["logits"] if "logits" in resp else None
            print("connector resp=", resp)
            if resp["status"] == 0:
                await asyncio.sleep(0.2)
                continue
            elif resp["status"] == 2:
                # final_full_text = self.tokenizer.decode(list(resp["token_ids"]))
                final_full_text = self.tokenizer.decode(list(resp["token_ids"]), skip_spetial_tokens=True)
                if len(final_full_text) > 10 and final_full_text[-10:] == "<|im_end|>":
                    final_full_text = final_full_text[:-10]
                success_frame_ct += 1
                metadata_splited = final_full_text.split("\n")
                # if not return_raw and len(metadata_splited) > 1:
                #     final_full_text = final_full_text[len(metadata_splited[0]) + 1:]  # 去除metadata
                flat_reply = final_full_text.replace("\n", "<s>")
                print(f"REPLY for request_id={request_id} is: {flat_reply}")
                logger.info(f"REPLY for request_id={request_id} is: {flat_reply}")

                return final_full_text, logits_info
            elif resp["status"] == -1:
                return "RUNTIME ERROR: liteqwen error with status -1", None
            else:
                delta_tokens = list(resp["token_ids"])
                accumulating_ids.extend(delta_tokens)
                frame_text = self.tokenizer.decode(accumulating_ids, skip_spetial_tokens=True)
                if len(frame_text) > 10 and frame_text[-10:] == "<|im_end|>":
                    frame_text = frame_text[:-10]
                success_frame_ct += 1
                # metadata_splited = frame_text.split("\n")
                # if not return_raw and len(metadata_splited) > 1:
                #     frame_text = frame_text[len(metadata_splited[0]) + 1:]  # 去除metadata
                unfinished_text = frame_text

            await asyncio.sleep(0.1)
            if success_frame_ct > self.max_sequence_length:
                flat_reply = unfinished_text.replace("\n", "<s>")
                logger.warning(f"Unfinished REPLY for request_id={request_id} is: {flat_reply}")
                return unfinished_text, logits_info

        metadata_splited = unfinished_text.split("\n")
        if not return_raw and len(metadata_splited) > 1:
            unfinished_text = unfinished_text[len(metadata_splited[0]) + 1:]  # 去除metadata
        flat_reply = unfinished_text.replace("\n", "<s>")
        logger.warning(f"Unfinished REPLY for request_id={request_id} is: {flat_reply}")
        return unfinished_text, logits_info

def get_inferer(data_id=-1, token_smem_info=None):
    return LiteqwenInferer(WORLD_SIZE, DATA_PARALLEL_SIZE, PIPELINE_PARALLEL_SIZE, GLOBAL_CONFIG["model_name_or_path"],
                         GLOBAL_CONFIG["adapters"], GLOBAL_CONFIG["max_dynamic_bsz"], GLOBAL_CONFIG["max_length"], GLOBAL_CONFIG["top_p"],
                         GLOBAL_CONFIG["top_k"], GLOBAL_CONFIG["temperature"], GLOBAL_CONFIG["use_lora"], thread_data_id=data_id, token_smem_info=token_smem_info)