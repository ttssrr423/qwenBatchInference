from global_config import DATA_PARALLEL_SIZE, GLOBAL_CONFIG, USE_LORA, NUM_GPUS
from serving.stream_pool import StreamPool, RetState, GenState
import sys
import logging
import time
import datetime
import asyncio
import numpy as np
import json
import traceback
import signal
import os

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

    return_logits = gen_kwargs.pop("return_logits", False)
    default_gen_kwargs = {"top_p": top_p,
                          "temperature": temperature,
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

global enlable_last_word
enlable_last_word = False # debug用，使用update_log输出进程crash前最后更新的打印。
global inferer

def infer_worker_loop(data_id, pipeline_parallel_size, world_size, process_loop, queue, mgr_dict, extra_dict, buffer_info, is_first_load, next_loading):
    logger.info('ALIVE KEEPER: DDP%s: registing ddp pid=%s, group id=%s.' % (data_id, os.getpid(), os.getpgrp()))
    if "ddp_processes" not in extra_dict:
        extra_dict["ddp_processes"] = f"{data_id},{os.getpid()},{os.getpgrp()}"
    else:
        extra_dict["ddp_processes"] += f"|{data_id},{os.getpid()},{os.getpgrp()}"

    extra_log_key = f"ddp_process_log{data_id}"
    def update_log(content):
        # 更新debug信息，上报给清理进程的，crash前最后的信息。
        global enlable_last_word
        if enlable_last_word:
            extra_dict[extra_log_key] = content

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

    model_load_start_tm = datetime.datetime.now()
    if next_loading < -1:
        # 按照data_id顺序加载。
        if data_id == 0:
            inferer = get_inferer(data_id=data_id, token_smem_info=token_smem_info)
            extra_dict["loading_id"] = 1
        else:
            while ("loading_id" not in extra_dict or not extra_dict["loading_id"] == data_id):
                time.sleep(0.5)
                if (datetime.datetime.now() - model_load_start_tm).total_seconds() >= 300:
                    break
            inferer = get_inferer(data_id=data_id, token_smem_info=token_smem_info)
            extra_dict["loading_id"] += 1
    else:
        time.sleep(12.0) # 重新拉起时，防止进程终止后cuda来不及清空，sleep确保cuda清零。
        # 按照is_first_load以及next_loading的指定顺序加载。
        if is_first_load:
            inferer = get_inferer(data_id=data_id, token_smem_info=token_smem_info)
            extra_dict["loading_id"] = next_loading
        else:
            while ("loading_id" not in extra_dict or not extra_dict["loading_id"] == data_id):
                if (datetime.datetime.now() - model_load_start_tm).total_seconds() >= 300:
                    break
                time.sleep(0.5)
            inferer = get_inferer(data_id=data_id, token_smem_info=token_smem_info)
            extra_dict["loading_id"] += next_loading
    logger.info(f"DDP: data_id={data_id} model loaded...")

    # 协程锁，以及每个协程占用中的record_id记录。
    async_lock = asyncio.Lock()
    occupying_coro_rids = [-1 for _ in range(max_bsz)]
    async def try_submit(coro_id, alock):
        first_frame_stat = []
        first_frame_mean_cost = 0.5
        prev_rec_id = data_id
        def regist_first_frame_stat(existed_first_frames, new_ff_cost):
            # 统计first frame用来预估c++排队时间。
            if len(existed_first_frames) >= 10:
                existed_first_frames.pop(0)
            existed_first_frames.append(new_ff_cost)
            return sum(existed_first_frames)/len(existed_first_frames)

        try:
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

                if len(inp_info) != 6:
                    logger.warning(f"python receiving invalid inputs, make sure fields are provided correctly.")
                    if len(inp_info) > 6:
                        snapshot_inp = inp_info[:6]
                    else:
                        del inp_info
                        continue
                else:
                    snapshot_inp = [x for x in inp_info]

                timeout = inferer.liteqwen_connector.timeout_in_secs
                req_id, cli_submit_tm, is_stream, text_info, history, gen_kwargs = snapshot_inp[0], snapshot_inp[1], snapshot_inp[2], \
                                                                    snapshot_inp[3], snapshot_inp[4], snapshot_inp[5]
                is_expired_flag = mgr_dict.get(req_id, "None")
                waited_tm = (datetime.datetime.now() - cli_submit_tm).total_seconds()

                # client计时timeout后set flag，没必要接着推理。
                if is_expired_flag == "expired":
                    del inp_info
                    if req_id in mgr_dict:
                        del mgr_dict[req_id]
                    logger.info(f"DDP{data_id}: skipping req_id={req_id}, queued for {waited_tm} secs, client informing, timeout expired.")
                    await asyncio.sleep(0.02)
                    continue

                # 如果client没有set timeout，worker自行根据传入的client接收请求时间，判定是否超时。
                if (datetime.datetime.now() - cli_submit_tm).total_seconds() > (timeout + 2.0):
                    del inp_info
                    if req_id in mgr_dict:
                        del mgr_dict[req_id]
                    logger.info(f"DDP{data_id}: skipping req_id={req_id}, queued for {waited_tm} secs, since client submit tm={cli_submit_tm}, timeout expired.")
                    await asyncio.sleep(0.02)
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
                async with alock:
                    # 加锁后分配并占用record_id，从prev_rec_id为起点继续寻找。
                    # record_id必须确保线程安全，不同进程的record_id对data_parallel_size取余不同，所以进程间也安全。
                    # 当生命周期完成时，需要释放record_id.
                    buf_record_id = buffer_pool.coro_wait_for_start(req_id, occupying_coro_rids, prev_rec_id)
                    occupying_coro_rids[coro_id] = buf_record_id
                    prev_rec_id = buf_record_id

                logger.info(f"DDP{data_id}: submitting stream={is_stream}, record_id/coro_id={buf_record_id}/{coro_id}, req_id={req_id}, queued for {waited_tm} secs, gen_kwargs={hf_gen_kwargs}, role={inp_role}, hist_len={len(history)}, query={flat_text}")
                hf_gen_kwargs.update({"seed": seed, "top_k": top_k, "adapter_name": lora_name_using})
                logger.info(f"input_id_list={input_id_list}")
                suc = 0
                if suc == 0:
                    # c++队列满，最多等5秒，如果还没有推理EOS，则放弃该样本。
                    retry_countdown = 5
                    while retry_countdown > 0:
                        # 基于历史first_frame耗时的统计，估算能否在超时之前完成c++排队，并获取首帧结果。
                        if (datetime.datetime.now() - cli_submit_tm).total_seconds() + first_frame_mean_cost * 1.5 > timeout:
                            # 预估没有足够时间的话，直接放弃样本。
                            suc = 0
                            first_frame_mean_cost = min(0.9*first_frame_mean_cost, 50.0) # 衰减防止一直等待
                            break
                        suc = inferer.liteqwen_connector.submit(buf_record_id, req_id, input_id_list, gen_config=hf_gen_kwargs)
                        if suc > 0:
                            break
                        retry_countdown -= 1
                        await asyncio.sleep(1.0)

                if suc == 0:
                    # c++后端队列满时，会丢弃queue中获取的请求。
                    # 尝试submit过程中，buf_record_id会被占用，如果尝试失败，需要把它释放，否则后面样本无法复用。
                    del inp_info
                    if req_id in mgr_dict:
                        del mgr_dict[req_id]
                    waited_tm = (datetime.datetime.now() - cli_submit_tm).total_seconds()
                    logger.info(f"DDP{data_id}: skipping req_id={req_id}, after {waited_tm} secs since client submit tm={cli_submit_tm}, expected cpp wait tm={first_frame_mean_cost} timeout.")
                    async with alock:
                        # 加锁后释放record_id，同时设置occupying_coro_rids[coro_id]=-1
                        buffer_pool.coro_set_timeout(buf_record_id, coro_id, occupying_coro_rids)
                    # update_log(f"released without generating rec={buf_record_id}, req={req_id}")
                    await asyncio.sleep(1.0)
                    continue

                # 成功提交c++，开始新一轮计时
                mgr_dict[req_id] = buf_record_id # 成功提交后，manager字典中的record_id正式生效。之后的mgr_dict清理由client或清理进程来处理。
                buffer_pool.refresh_time(buf_record_id)
                buffer_pool.set_gen_state(buf_record_id, GenState.GENERATING)
                token_pool[buf_record_id][0] = 0 # 重置python eos状态
                token_pool[buf_record_id][1] = 0 # 重置c++ eos状态
                token_pool[buf_record_id][2] = 0 # 重置c++ reply长度
                cqueue_start_tm = datetime.datetime.now() # 成功提交后刷新的timeout计时起点，也是c++batch池的timeout计时起点。
                # update_log(f"submit suc, generating rec={buf_record_id}, req={req_id}")
                await asyncio.sleep(0.2)

                first_frame_cost = -1.0
                success_frame_ct = 0
                prev_token_len = 0
                release_dely = 3.0  # 释放record_id时预留3秒record倒计时，给c++和client端进行收尾后再正式释放。
                while True:
                    terminated, is_eos, gen_len = token_pool[buf_record_id][0], token_pool[buf_record_id][1], token_pool[buf_record_id][2]
                    # update_log(f"gen content read, len={gen_len}, rec={buf_record_id}, req={req_id}")
                    if first_frame_cost < 0.0 and gen_len > 0:
                        first_frame_cost = (datetime.datetime.now() - cqueue_start_tm).total_seconds()
                        first_frame_mean_cost = regist_first_frame_stat(first_frame_stat, first_frame_cost)

                    # 生成过程中超时判定
                    if not is_stream:
                        is_expired = (datetime.datetime.now() - cli_submit_tm).total_seconds() > timeout
                    else:
                        if first_frame_cost > 0:
                            cli_waited = (datetime.datetime.now()-cqueue_start_tm).total_seconds() - first_frame_cost
                        else:
                            cli_waited = (datetime.datetime.now() - cli_submit_tm).total_seconds()
                        is_expired = cli_waited > timeout
                    cpp_time_cost = (datetime.datetime.now() - cqueue_start_tm).total_seconds()

                    if is_expired:
                        # 如果c++还在batch队列中，使用liteqwen_connector.delete_waiting_req移除。如果c++已经开始推理，set_expire会安全处理推理中的样本
                        inferer.liteqwen_connector.delete_waiting_req(req_id)
                        inferer.liteqwen_connector.set_expire(req_id) # 这里不能主动delete_req清理c++的资源，可能会让ctx_ref->Append变成nullptr->Append导致崩溃。
                        if not is_stream:
                            waited_tm = (datetime.datetime.now() - cli_submit_tm).total_seconds()
                            err_msg = f"RUNTIME ERROR: DDP{data_id}: skipping rec={buf_record_id}, req_id={req_id}, after {waited_tm} secs since client submit tm={cli_submit_tm}, timeout expired during generation."
                            logger.info(err_msg)
                            if is_eos and gen_len>0:
                                # 尝试在超时时获取eos结果。
                                direct_replied_ids = list(token_pool[buf_record_id][3:3 + gen_len])
                                final_full_text = inferer.tokenizer.decode(direct_replied_ids, skip_spetial_tokens=True)
                                if len(final_full_text) > 10 and final_full_text[-10:] == "<|im_end|>":
                                    final_full_text = final_full_text[:-10]
                                flat_reply = final_full_text.replace("\n", "\\n")
                                buffer_pool.write_result(buf_record_id, final_full_text, is_first_frame=(success_frame_ct == 0))
                                logger.info(f"DDP{data_id}: REPLY for record_id={buf_record_id}, request_id={req_id} first/total=({first_frame_cost}/{cpp_time_cost}) secs, inp/rep={len(input_id_list)}/{gen_len} tokens; text={flat_reply}")
                            else:
                                buffer_pool.write_result(buf_record_id, "RUNTIME ERROR: 生成等待超时。", is_first_frame=False)
                        else:
                            direct_replied_ids = list(token_pool[buf_record_id][3:3+gen_len])
                            final_full_text = inferer.tokenizer.decode(direct_replied_ids, skip_spetial_tokens=True)
                            logger.info(f"RUNTIME ERROR: DDP{data_id}: forced stream generation terminate due to timeout, req_id={req_id}, rec={buf_record_id}.")
                            buffer_pool.write_result(buf_record_id, final_full_text, is_first_frame=(success_frame_ct == 0))

                        # update_log(f"should expire, setting c++ stop for rec={buf_record_id}, req={req_id}")
                        break

                    if terminated:
                        # 通过stop_generation api调用终止流式生成。
                        direct_replied_ids = list(token_pool[buf_record_id][3:3+gen_len])
                        final_full_text = inferer.tokenizer.decode(direct_replied_ids, skip_spetial_tokens=True)
                        buffer_pool.write_result(buf_record_id, final_full_text, is_first_frame=(success_frame_ct == 0))
                        flat_reply = final_full_text.replace("\n", "\\n")
                        logger.info(f"DDP{data_id}: client active terminating with REPLY for record_id={buf_record_id}, request_id={req_id} first/total=({first_frame_cost}/{cpp_time_cost}) secs, inp/rep={len(input_id_list)}/{gen_len} tokens; text={flat_reply}")

                        inferer.liteqwen_connector.delete_waiting_req(req_id)
                        inferer.liteqwen_connector.set_expire(req_id)
                        break

                    if gen_len == 0:
                        await asyncio.sleep(0.1)
                        continue
                    if gen_len == prev_token_len:
                        await asyncio.sleep(0.1)
                        continue
                    else:
                        # 只有存在增量非空内容时继续处理
                        prev_token_len = gen_len

                    if is_stream or (is_eos and not is_stream):
                        # update_log(f"regular eos for rec={buf_record_id}, req={req_id}")
                        direct_replied_ids = list(token_pool[buf_record_id][3:3+gen_len])
                        final_full_text = inferer.tokenizer.decode(direct_replied_ids, skip_spetial_tokens=True)
                        if len(final_full_text) > 10 and final_full_text[-10:] == "<|im_end|>":
                            final_full_text = final_full_text[:-10]
                        flat_reply = final_full_text.replace("\n", "\\n")

                        # 使用get_entity_frame的方式，而非直接读取shared_memory的方式获取logits，速度会比直接读shared_memory慢一些。
                        if not final_full_text.startswith("RUNTIME ERROR:") and "return_logits" in hf_gen_kwargs and hf_gen_kwargs["return_logits"]:
                            resp = inferer.liteqwen_connector.get_generated(req_id, return_logits=hf_gen_kwargs["return_logits"])
                            if "logits" in resp:
                                logits_info = resp["logits"] # if "logits" in resp else None
                                lgt_json = json.dumps(logits_info, ensure_ascii=False)
                                lgt_str = "LOGITS_LEN:"+str(len(lgt_json))+":"+lgt_json
                                if len(lgt_str) + len(final_full_text) < inferer.max_sequence_length * 4:
                                    # max_seq_len * 4 is the estimated token number can be transfered.
                                    final_full_text = lgt_str + final_full_text
                        # token_ids转化完成后的文本，写入text shared_memory
                        buffer_pool.write_result(buf_record_id, final_full_text, is_first_frame=(success_frame_ct == 0))
                    else:
                        # no_stream and not eos
                        await asyncio.sleep(0.1)

                    if is_eos:
                        logger.info(f"DDP{data_id}: REPLY for record_id={buf_record_id}, request_id={req_id} first/total=({first_frame_cost}/{cpp_time_cost}) secs, inp/rep={len(input_id_list)}/{gen_len} tokens; text={flat_reply}")
                        break

                    success_frame_ct += 1
                    # 最大帧数或生成token长度超过限制，兜底退出，一般不触发。
                    if max(success_frame_ct, gen_len) > inferer.max_sequence_length:
                        direct_replied_ids = list(token_pool[buf_record_id][3:3+gen_len])
                        final_full_text = inferer.tokenizer.decode(direct_replied_ids, skip_spetial_tokens=True)
                        flat_text = final_full_text.replace("\n", "\\n")
                        logger.info(f"DDP{data_id}: max length reached for record_id={buf_record_id}, request_id={req_id}, exiting receiving loop with reply={flat_text}")
                        break

                # update_log(f"server ddp cleanup for rec={buf_record_id}, req={req_id}")
                if not is_expired and not terminated:
                    # 正常eos或达到最大长度后，c++推理会正常终止，这时主动delete才不会被c++后续生成的步骤触发crash。
                    # 主动终止或超时后，c++可能没有及时收到通知，继续产生新token，所以使用set_expire延迟清理。
                    inferer.liteqwen_connector.delete_req(req_id)

                buffer_pool.set_gen_state(buf_record_id, GenState.END)
                buffer_pool.set_ret_state(buf_record_id, RetState.END)
                async with alock:
                    # update_log(f"eos releasing rec={buf_record_id}, req={req_id}")
                    # 释放record_id
                    buffer_pool.coro_set_timeout(buf_record_id, coro_id, occupying_coro_rids, delay=release_dely)
                try:
                    del inp_info
                except:
                    pass
                continue # 抓取下一条样本
        except Exception as ex:
            err_str = traceback.format_exc()
            logger.error(f"DDP{data_id}: DDP ERROR: {ex}\n{err_str}, infinite loop exit, one dp is dead.")

    infinite_tasks = []
    # 每个data_id的进程下最多max_bsz个协程不断尝试submit请求。
    for bi in range(max_bsz):
        task = process_loop.create_task(try_submit(bi, async_lock))
        infinite_tasks.append(task)

    process_loop.run_until_complete(asyncio.gather(*infinite_tasks))