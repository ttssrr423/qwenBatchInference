from global_config import DATA_PARALLEL_SIZE, GLOBAL_CONFIG, USE_LORA, NUM_GPUS
from serving.stream_pool import StreamPool, RetState, GenState
import sys
import logging
import time
import datetime
import asyncio
import numpy as np
import json
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
    level="INFO"
)
logger = logging.getLogger(__name__)

def convert_to_openai_history(v2_history):
    new_hist = []
    for conv_round in v2_history:
        new_hist.append({"role": "user", "content": conv_round[0]})
        new_hist.append({"role": "assistant", "content": conv_round[1]})
    return new_hist

def check_history(v3_history):
    for item in v3_history:
        if not isinstance(item, dict):
            return False
        if "role" not in item or "content" not in item:
            return False
    return True

def prepare_kwargs_and_lora(self, req_id, _inferer, gen_kwargs, use_lora, data_id):
    adapter_name_result = "skip"
    using_adapter_name = gen_kwargs.pop("adapter_name", "skip")

    if use_lora:
        if not hasattr(_inferer, "current_lora_name"):
            _inferer.current_lora_name = "skip"
        if _inferer.current_lora_name != using_adapter_name:
            if logger is not None:
                logger.info(f"DDP{data_id}: setting lora to {using_adapter_name} after {req_id}")
        adapter_name_result = using_adapter_name

    top_p = GLOBAL_CONFIG["top_p"]
    temperature = GLOBAL_CONFIG["temperature"]

    # no_english = gen_kwargs.pop("no_english", False)
    # if no_english:
    #     logits_mask = self.english_mask
    # else:
    #     logits_mask = self.null_mask

    return_logits = gen_kwargs.pop("return_logits", False)
    default_gen_kwargs = {"top_p": top_p,
                          "temperature": temperature,
                          # "logits_mask": logits_mask,
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

global inferer
def infer_worker_loop(data_id, pipeline_parallel_size, world_size, process_loop, queue, mgr_dict, extra_dict, buffer_info):
    asyncio.set_event_loop(process_loop)
    buffer_pool = StreamPool(buffer_info)
    buffer_pool.set_dp(data_id, int(world_size/pipeline_parallel_size))
    max_bsz = int(GLOBAL_CONFIG["max_dynamic_bsz"])

    from serving.liteqwen_inferer import get_inferer
    logger.info(f"DDP: python loop: data_id={data_id}/{world_size//pipeline_parallel_size} thread started.")
    global inferer
    inferer = None
    token_smem_info = {"name": buffer_pool.token_buf_name, "size_t": buffer_pool.max_qsize*buffer_pool.token_record_stride*4, "record_maxlen": buffer_pool.token_record_stride}
    token_pool = np.ndarray(shape=(buffer_pool.max_qsize, buffer_pool.token_record_stride), dtype=np.int32, buffer=buffer_pool.token_buffer.buf)

    if data_id == 0:
        inferer = get_inferer(data_id=data_id, token_smem_info=token_smem_info)
        extra_dict["loading_id"] = 1
    else:
        while ("loading_id" not in extra_dict or not extra_dict["loading_id"] == data_id):
            time.sleep(0.5)
        inferer = get_inferer(data_id=data_id, token_smem_info=token_smem_info)
        extra_dict["loading_id"] += 1
    logger.info(f"DDP: data_id={data_id} model loaded...")

    async def try_submit():
        while True:
            try:
                # 不同协程共用一个进程，而manager.Queue是进程安全的，所以如果不用get_nowait()，而是用get()，则会被一个协程阻塞整个进程下的所有协程，导致死锁。
                inp_info = queue.get_nowait()
            except:
                await asyncio.sleep(0.1)
                continue
            if inp_info is None:
                del inp_info
                await asyncio.sleep(0.1)
                continue

            if len(inp_info) != 5:
                logger.warning(f"python receiving invalid inputs, make sure fields are provided correctly.")
                if len(inp_info) > 5:
                    snapshot_inp = inp_info[:5]
                else:
                    del inp_info
                    continue
            else:
                snapshot_inp = [x for x in inp_info]

            req_id, is_stream, text_info, history, gen_kwargs = snapshot_inp[0], snapshot_inp[1], snapshot_inp[2], \
                                                                snapshot_inp[3], snapshot_inp[4]
            is_expired_flag = mgr_dict.get(req_id, "None")
            if is_expired_flag == "expired":
                del inp_info
                del mgr_dict[req_id]
                await asyncio.sleep(0.05)
                continue

            inp_text, inp_metadata, inp_role = text_info[0], text_info[1], text_info[2]
            if inp_role is None or inp_role not in ["user", "assistant", "system", "observation"]:
                inp_role = "user"
            if history is None:
                history = []
            elif len(history) > 0:
                if isinstance(history[0], list) and len(history[0]) == 2 and isinstance(history[0][0], str):
                    history = convert_to_openai_history(history)
                elif isinstance(history[0], dict):
                    valid_history = check_history(history)
                    if not valid_history:
                        logger.warning(f"invalid history for req_id={req_id}, skipping request. hist={history}")
                        mgr_dict[req_id] = "RUNTIME ERROR: 输入history格式有误。"
                        del inp_info
                        continue
                else:
                    logger.warning(f"invalid V3 history for req_id={req_id}, skipping request. hist={history}")
                    mgr_dict[req_id] = "RUNTIME ERROR: 输入history格式有误。"
                    del inp_info
                    continue

            buf_record_id = buffer_pool.wait_for_start(req_id)

            seed = int(gen_kwargs.pop("seed", -1)) # 默认随机
            top_k = 32 # max(min(gen_kwargs.pop("topk", inferer.top_k), 64), 1)

            message_list = history + [{"role": inp_role, "content": inp_text}]
            completed_text = inferer.tokenizer.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True)
            if "prefix_text" in gen_kwargs:
                completed_text = completed_text + str(gen_kwargs["prefix_text"])
            inputs = inferer.tokenizer([completed_text], return_tensors="pt").to("cpu")
            inp_len = int(inputs["input_ids"].shape[1])
            if inp_len > (GLOBAL_CONFIG["max_length"] - 20):
                logger.warning(f"DDP{data_id}: skipping req_id={req_id}, 输入太长，距离max_length不足20个token，无法生成。")
                mgr_dict[req_id] = "RUNTIME ERROR: 输入太长，距离max_length不足20个token，无法生成。"
                del inp_info
                continue

            input_id_list = [int(x) for x in list(inputs["input_ids"][0].cpu().numpy())]
            lora_name_using, hf_gen_kwargs = prepare_kwargs_and_lora(inferer, req_id, inferer, gen_kwargs, inferer.use_lora, data_id)

            flat_text = inp_text.replace("\n", "\\n")
            logger.info(f"DDP{data_id}: submitting stream={is_stream}, record_id={buf_record_id}, req_id={req_id}, gen_kwargs={hf_gen_kwargs}, role={inp_role}, hist_len={len(history)}, query={flat_text}")

            hf_gen_kwargs.update({"seed": seed, "top_k": top_k, "adapter_name": lora_name_using})
            suc = inferer.liteqwen_connector.submit(buf_record_id, req_id, input_id_list, gen_config=hf_gen_kwargs)
            if suc == 0:
                # c++后端队列最大长是 data_parallel_size * max_dynamic_bsz * 5，比python的pool容量小。排队太严重时，这一行continue会开始丢弃queue中获取的请求。
                mgr_dict[req_id] = "RUNTIME ERROR: " + "排队已满，请稍后再提交请求。"
                del inp_info
                continue

            buffer_pool.refresh_time(buf_record_id)
            buffer_pool.set_gen_state(buf_record_id, GenState.GENERATING)
            token_pool[buf_record_id][0] = 0 # 重置python eos状态
            token_pool[buf_record_id][1] = 0 # 重置c++ eos状态
            token_pool[buf_record_id][2] = 0 # 重置c++ reply长度
            mgr_dict[req_id] = buf_record_id # 成功提交后，manager字典中的record_id正式生效。
            submitted_tm = datetime.datetime.now()
            await asyncio.sleep(0.2)

            timeout = inferer.liteqwen_connector.timeout_in_secs
            first_frame_cost = -1.0
            success_frame_ct = 0
            prev_token_len = 0
            while True:
                terminated, is_eos, gen_len = token_pool[buf_record_id][0], token_pool[buf_record_id][1], token_pool[buf_record_id][2]
                time_cost = (datetime.datetime.now() - submitted_tm).total_seconds()
                if (time_cost >= timeout) and (time_cost-first_frame_cost >= timeout):
                    buffer_pool.write_result(buf_record_id, f"RUNTIME ERROR: data_id={data_id} timeout during generation of rec={buf_record_id}, req={req_id}", is_first_frame=False)
                    break
                if terminated:
                    direct_replied_ids = list(token_pool[buf_record_id][3:3+gen_len])
                    final_full_text = inferer.tokenizer.decode(direct_replied_ids, skip_spetial_tokens=True)
                    buffer_pool.write_result(buf_record_id, final_full_text, is_first_frame=(success_frame_ct == 0))
                    break
                if gen_len == 0:
                    await asyncio.sleep(0.1)
                    continue
                if gen_len == prev_token_len:
                    await asyncio.sleep(0.1)
                    continue
                else:
                    prev_token_len = gen_len

                if first_frame_cost < 0.0:
                    first_frame_cost = (datetime.datetime.now() - submitted_tm).total_seconds()

                if is_stream or (is_eos and not is_stream):
                    direct_replied_ids = list(token_pool[buf_record_id][3:3+gen_len])
                    final_full_text = inferer.tokenizer.decode(direct_replied_ids, skip_spetial_tokens=True)
                    if len(final_full_text) > 10 and final_full_text[-10:] == "<|im_end|>":
                        final_full_text = final_full_text[:-10]
                    flat_reply = final_full_text.replace("\n", "\\n")

                    if not final_full_text.startswith("RUNTIME ERROR:") and "return_logits" in hf_gen_kwargs and hf_gen_kwargs["return_logits"]:
                        resp = inferer.liteqwen_connector.get_generated(req_id, return_logits=hf_gen_kwargs["return_logits"])
                        logits_info = resp["logits"] # if "logits" in resp else None
                        lgt_json = json.dumps(logits_info, ensure_ascii=False)
                        lgt_str = "LOGITS_LEN:"+str(len(lgt_json))+":"+lgt_json
                        if len(lgt_str) + len(final_full_text) < inferer.max_sequence_length * 4:
                            # max_seq_len * 4 is the estimated token number can be transfered.
                            final_full_text = lgt_str + final_full_text
                    buffer_pool.write_result(buf_record_id, final_full_text, is_first_frame=(success_frame_ct == 0))
                else:
                    # no_stream and not eos
                    await asyncio.sleep(0.05)

                if is_eos:
                    logger.info(f"DDP{data_id}: REPLY for record_id={buf_record_id}, request_id={req_id} first/total=({first_frame_cost}/{time_cost}) secs, inp/rep={len(input_id_list)}/{gen_len} tokens; text={flat_reply}")
                    break

                success_frame_ct += 1
                if max(success_frame_ct, gen_len) > inferer.max_sequence_length:
                    direct_replied_ids = list(token_pool[buf_record_id][3:3+gen_len])
                    final_full_text = inferer.tokenizer.decode(direct_replied_ids, skip_spetial_tokens=True)
                    flat_text = final_full_text.replace("\n", "\\n")
                    logger.info(f"DDP{data_id}: max length reached for record_id={buf_record_id}, request_id={req_id}, exiting receiving loop with reply={flat_text}")
                    break

            inferer.liteqwen_connector.delete_req(req_id)
            buffer_pool.set_gen_state(buf_record_id, GenState.END)
            try:
                del inp_info
            except:
                pass
            continue # 抓取下一条样本

    infinite_tasks = []
    # 每个data_id的进程下最多max_bsz个协程不断尝试submit请求。
    for bi in range(max_bsz):
        task = process_loop.create_task(try_submit())
        infinite_tasks.append(task)

    process_loop.run_until_complete(asyncio.gather(*infinite_tasks))