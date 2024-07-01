import ctypes
from ctypes import POINTER
import math
import os
import numpy as np
from enum import IntEnum
import threading
import torch
import json
from tqdm import tqdm
from collections import OrderedDict
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
from safetensors import safe_open

import platform
if platform.system() == 'Windows':
    liteqwen_lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.split(os.path.realpath(__file__))[0], "liteqwen_conn.dll"))
else:
    liteqwen_lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.split(os.path.realpath(__file__))[0], "libliteqwen_conn.so"))

# (char* request_id, int input_length, int* input_ids, float top_p, int top_k, float temperature, int max_length, int max_new_tokens, 
#    char* adapter_name, int seed, float mask_base_val, float mask_except_val, int except_ids_len, int* logit_except_ids, bool return_logits)
liteqwen_lib.submit_request.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int, 
    ctypes.c_char_p, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_void_p, ctypes.c_bool]

# (char *key, int shape_length, int *shape, int dtype, int oriDataType, void *oriData)
liteqwen_lib.store_tensor.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

# (char* adapter_name, bool fan_in_fan_out, float lora_alpha, int r, char* target_modules_joined)
liteqwen_lib.add_lora_adapter_config.argtypes = [ctypes.c_char_p, ctypes.c_bool, ctypes.c_float, ctypes.c_int, ctypes.c_char_p]

# (int world_size, int data_parallel_size, int pipeline_parallel, char* json_config_path, int layer_num, int* block2device_list, int max_dynamic_bsz, int max_sequence_length, int max_queue_size, int timeout)
liteqwen_lib.initialize_empty_qwen2.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

liteqwen_lib.start_loops.argtypes = []

liteqwen_lib.get_frame.argtypes = [ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
liteqwen_lib.get_frame.restype = ctypes.POINTER(ctypes.c_int)
liteqwen_lib.free_frame.argtypes = [ctypes.POINTER(ctypes.c_int)]

liteqwen_lib.get_frame_entity.argtypes = [ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
liteqwen_lib.get_frame_entity.restype = ctypes.POINTER(ctypes.c_ubyte)
liteqwen_lib.free_frame_entity.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]

liteqwen_lib.make_q4_meta.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool]


class DataType(IntEnum):
    FLOAT16 = 0
    INT32 = 1
    INT64 = 2
    INT16 = 3
    INT8 = 4
    INT4 = 5
    INT2 = 6
    BIT = 7
    BFLOAT16=8
    FLOAT32 = 9
    INT4_NOZERO = 10 # 不用zeroPoint的int4, floatValue = min + uint4Value * scale
    INT32PARAM = 100 # int32的参数，这种类型的数据永远存在CPU上


class QLinearParamBuffer:
    layer_params = {}

class GPTQLinearParamSet(object):
    def __init__(self, prefix, group_size):
        self.prefix = prefix
        self.qweight = None
        self.qzeros = None
        self.scales = None
        self.g_idx = None
        self.bias = None
        self.has_bias = False
        self.in_feature = -1
        self.out_feature = -1
        self.group_size = group_size

    def check_validity(self):
        if self.qzeros is not None and self.qweight is not None and self.scales is not None and self.g_idx is not None:
            self.has_bias = self.bias is not None
            in_feature = int(self.g_idx.shape[0])
            out_feature = int(self.qweight.shape[1])
            pack_num = 8
            if int(self.qzeros.shape[1]) * pack_num != out_feature:
                raise Exception(f"error on quantization parameter load for {self.prefix} only 4bit gptq supported. please check that qzeros.shape[1] * 32 // 4 == qweight.shape[1]")
            self.in_feature = in_feature
            self.out_feature = out_feature
            return True
        return False

    def upload(self, device_id):
        weight_name = self.prefix + ".qweight"
        w_shape = [int(x) for x in self.qweight.shape]
        liteqwen_lib.store_tensor(weight_name.encode(), len(w_shape), (ctypes.c_int * len(w_shape))(*w_shape),
                                  int(DataType.INT32), int(DataType.INT32),
                                  self.qweight.cpu().numpy().astype(np.int32).ctypes.data_as(ctypes.c_void_p))
        
        zero_name = self.prefix + ".qzeros"
        z_shape = [int(x) for x in self.qzeros.shape]
        liteqwen_lib.store_tensor(zero_name.encode(), len(z_shape), (ctypes.c_int * len(z_shape))(*z_shape),
                                  int(DataType.INT32), int(DataType.INT32),
                                  self.qzeros.cpu().numpy().astype(np.int32).ctypes.data_as(ctypes.c_void_p))

        scale_name = self.prefix + ".scales"
        s_shape = [int(x) for x in self.scales.shape]
        liteqwen_lib.store_tensor(scale_name.encode(), len(s_shape), (ctypes.c_int * len(s_shape))(*s_shape),
                                  int(DataType.FLOAT16), int(DataType.FLOAT16),
                                  self.scales.cpu().numpy().astype(np.float16).ctypes.data_as(ctypes.c_void_p))

        g_name = self.prefix + ".g_idx"
        g_shape =  [int(x) for x in self.g_idx.shape]
        liteqwen_lib.store_tensor(g_name.encode(), len(g_shape), (ctypes.c_int * len(g_shape))(*g_shape),
                            int(DataType.INT32), int(DataType.INT32),
                            self.g_idx.cpu().numpy().astype(np.int32).ctypes.data_as(ctypes.c_void_p))
        
        if self.bias is not None:
            b_name = self.prefix + ".bias"
            b_shape =  [int(x) for x in self.bias.shape]
            liteqwen_lib.store_tensor(b_name.encode(), len(b_shape), (ctypes.c_int * len(b_shape))(*b_shape),
                                int(DataType.FLOAT16), int(DataType.FLOAT16),
                                self.bias.cpu().numpy().astype(np.float16).ctypes.data_as(ctypes.c_void_p))

        # print(f"uploading quant linear params for {self.prefix} on device={device_id}")
        liteqwen_lib.make_q4_meta(self.prefix.encode(), self.in_feature, self.out_feature, self.group_size, self.has_bias)
        return

class Qwen2Inferer:
    def __init__(self, model_path, world_size, data_parallel_size, pipeline_parallel_size, max_dynamic_bsz, max_length, top_p=0.8, top_k=50, temperature=0.8, use_lora=False, layer_to_device_list=None):
        self.world_size = world_size
        self.data_parallel_size = data_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        assert (self.world_size == self.data_parallel_size * self.pipeline_parallel_size)
        self.max_length = max_length # max dynamic length, BL
        self.max_dynamic_bsz = max_dynamic_bsz
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.use_lora = use_lora
        self.lora_name_insert_map = {}
        self.dp_layer_device_map = []
        self.timeout_in_secs = 120.0

        self.min_len = 32
        # self.maxlen_choices = [32]
        # _tmp_len = 32
        # while (_tmp_len < self.max_length):
        #     self.maxlen_choices.append(min(_tmp_len * 2, self.max_length))
        #     _tmp_len = _tmp_len * 2
        # print(f"maxlen choices: {self.maxlen_choices}")

        json_path = os.path.join(model_path, "config.json")
        layer_num = json.load(open(json_path, encoding="utf8"))["num_hidden_layers"]
        if pipeline_parallel_size == 1:
            layer_to_device_list = [0] * layer_num
        elif layer_to_device_list is None and pipeline_parallel_size>1:
            embed_equivalen = 3
            vocab_equivalent = 3
            per_stage_layers = max(int((layer_num + embed_equivalen + vocab_equivalent)  / pipeline_parallel_size), 1)
            last_stage_layers = int((layer_num - per_stage_layers * (pipeline_parallel_size-2)) / 2)
            first_stage_layers = layer_num - last_stage_layers - per_stage_layers * (pipeline_parallel_size-2)
            middle_stages = []
            for stg_i in range(pipeline_parallel_size-2):
                middle_stages.extend([stg_i+1]*per_stage_layers)
            layer_to_device_list = ([0]*first_stage_layers) + middle_stages + ([pipeline_parallel_size-1]*last_stage_layers)
        else:
            assert isinstance(layer_to_device_list, list)
            for i in range(pipeline_parallel_size):
                assert i in layer_to_device_list
        print("initializing with layer-device setup: "+str(layer_to_device_list))
        # # (int world_size, int data_parallel_size, int pipeline_parallel, char* json_config_path, int layer_num, int* block2device_list, int max_dynamic_bsz, int max_sequence_length, int max_queue_size, int timeout)
        # print(f"python parsing model params to c++: layer_num={layer_num}, dynamic_bsz={max_dynamic_bsz}, BL={max_length}")
        liteqwen_lib.initialize_empty_qwen2(world_size, data_parallel_size, pipeline_parallel_size, json_path.encode(), layer_num, (ctypes.c_int * layer_num)(*layer_to_device_list), max_dynamic_bsz, max_length, data_parallel_size*max_dynamic_bsz*5, int(self.timeout_in_secs*1000))

        for di in range(data_parallel_size):
            dp_layer_devices = [di*pipeline_parallel_size + x for x in layer_to_device_list]
            self.dp_layer_device_map.append(dp_layer_devices)


    def add_lora_cfg(self, lora_config_path, adapter_name="default"):
        config = json.load(open(lora_config_path, encoding="utf8"))
        liteqwen_lib.add_lora_adapter_config(adapter_name.encode(), bool(config["fan_in_fan_out"]), float(config["lora_alpha"]), int(config["r"]), (",".join(config["target_modules"])).encode())
        self.lora_name_insert_map[adapter_name] = False

    def submit(self, request_id, input_ids, gen_config=None, logits_mask_base_val=0.0, logits_mask_except_val=0.0, logits_mask_except_ids=[], return_logits=False):
        top_p = self.top_p
        top_k = self.top_k
        temperature = self.temperature
        maxlen = -1
        max_new_tokens = -1
        seed = -1
        if len(input_ids) > self.max_length - 32:
            print("input is too long, leaving less than 32 tokens before max_length. Please shorten the input.")
            return -1
        
        inp_len = len(input_ids)
        if isinstance(gen_config, dict):
            if "top_p" in gen_config: top_p = gen_config["top_p"]
            if "top_k" in gen_config: top_k = gen_config["top_k"]
            if "temperature" in gen_config: temperature = gen_config["temperature"]
            if "max_new_tokens" in gen_config and gen_config["max_new_tokens"] > 0:
                max_new_tokens = gen_config["max_new_tokens"]
            else:
                max_new_tokens = -1
            
            if "max_length" in gen_config and gen_config["max_length"] > 0:
                # 调整maxlen到min_len整数倍
                maxlen = min(self.max_length, gen_config["max_length"])
                min_len = 32
                if maxlen < min_len:
                    maxlen = min_len
                else:
                    maxlen_rounded = ((maxlen - 1) // self.min_len + 1) * self.min_len
                    maxlen = min(self.max_length, maxlen_rounded)
            else:
                maxlen = -1
            
            if "max_new_tokens" in gen_config and gen_config["max_new_tokens"] > 0:
                maxlen2 = ((inp_len + gen_config["max_new_tokens"] - 1) // self.min_len + 1) * self.min_len
            else:
                maxlen2 = -1
            
            if maxlen > 0 and maxlen2 > 0:
                maxlen = min(maxlen, maxlen2)
            elif maxlen > 0:
                pass
            elif maxlen2 > 0:
                maxlen = maxlen2
            else:
                pass # 允许推理框架根据max_new_tokens自动调整
            
            if "seed" in gen_config: seed = gen_config["seed"]
        else:
            gen_config = {}
        
        skip_lora = gen_config.get("skip_lora", False)
        if "adapter_name" in gen_config:
            if not skip_lora:
                adapter_name = gen_config["adapter_name"]
            else:
                adapter_name = "skip"
        else:
            adapter_name = "default" if (self.use_lora and not skip_lora) else "skip"

        if "logits_mask" in gen_config:
            logits_mask_base_val = gen_config["logits_mask"]["default"]
            logits_mask_except_val = gen_config["logits_mask"]["value"]
            logits_mask_except_ids = gen_config["logits_mask"]["ids"]

        if "return_logits" in gen_config:
            return_logits = gen_config["return_logits"]

        if len(logits_mask_except_ids) == 0:
            logits_mask_except_ids.append(0)
        logits_mask_ids_len = len(logits_mask_except_ids)
        if logits_mask_base_val != 0.0 or logits_mask_except_val != 0.0:
            assert logits_mask_ids_len > 1

        # (char* request_id, int input_length, int* input_ids, float top_p, int top_k, float temperature, int max_length, int max_new_tokens, 
        #    char* adapter_name, int seed, float mask_base_val, float mask_except_val, int except_ids_len, int* logit_except_ids, bool return_logits)
        # print("maxlen=", maxlen, "max_new_tokens=", max_new_tokens)
        success = liteqwen_lib.submit_request(request_id.encode(), inp_len, (ctypes.c_int * inp_len)(*input_ids), top_p, top_k, temperature, int(maxlen), int(max_new_tokens), 
            adapter_name.encode(), seed, logits_mask_base_val, logits_mask_except_val, logits_mask_ids_len, (ctypes.c_int * (logits_mask_ids_len+1))(*logits_mask_except_ids), return_logits)
        return success

    def add_weight(self, device_id, param_name, tensor, qlinear_prefix=None, quant_group_size=None):
        #  int gpu_id, char *key, int shape_length, int *shape, int dtype, int oriDataType, void *oriData
        device_id = -1
        shape = [int(x) for x in list(tensor.shape)]

        if qlinear_prefix is not None and len(qlinear_prefix) > 0:
            # 将所有qlinear层的prefix进行缓存初始化
            for prefix in qlinear_prefix:
                if prefix not in QLinearParamBuffer.layer_params:
                    QLinearParamBuffer.layer_params[prefix] = GPTQLinearParamSet(prefix, quant_group_size)
                if param_name.startswith(prefix):
                    is_quant_param = True
                    if param_name.endswith(".qweight"):
                        QLinearParamBuffer.layer_params[prefix].qweight = tensor
                    elif param_name.endswith(".qzeros"):
                        QLinearParamBuffer.layer_params[prefix].qzeros = tensor
                    elif param_name.endswith(".scales"):
                        QLinearParamBuffer.layer_params[prefix].scales = tensor
                    elif param_name.endswith(".g_idx"):
                        QLinearParamBuffer.layer_params[prefix].g_idx = tensor
                    elif param_name.endswith(".bias"):
                        QLinearParamBuffer.layer_params[prefix].bias = tensor
                    else:
                        is_quant_param = False
                    if is_quant_param:
                        is_completed = QLinearParamBuffer.layer_params[prefix].check_validity()
                        if is_completed:
                            (QLinearParamBuffer.layer_params[prefix]).upload(device_id)
                        return

        parsing_type = DataType.FLOAT16
        np_type = np.float16
        if tensor.dtype == torch.float16:
            parsing_type = DataType.FLOAT16
            np_type = np.float16
        elif tensor.dtype == torch.float32:
            parsing_type = DataType.FLOAT32
            np_type = np.float32
        else:
            raise ("权重需要是fp16或fp32，或GPTQ格式的量化权重。内部推理统一使用half格式，暂不兼容其他格式的tensor")
        
        liteqwen_lib.store_tensor(param_name.encode(), len(shape), (ctypes.c_int * len(shape))(*shape), int(parsing_type), int(parsing_type), tensor.cpu().numpy().astype(np_type).ctypes.data_as(ctypes.c_void_p))
        return

    def load_base_model_params(self, model_path):
        # 先加载到cpu上
        device_id = -1
        # if os.path.exists(open(os.path.join(model_path, "pytorch_model.bin.index.json"))):
        #     weight_info = json.load(open(os.path.join(model_path, "pytorch_model.bin.index.json"), mode="r", encoding="utf8"))
        
        #     file_names = list(sorted(set(weight_info["weight_map"].values())))
        #     for file in file_names:
        #         state_dict = torch.load(open(os.path.join(model_path, file), mode="rb"), map_location="cpu")
        #         for nm, param in tqdm(state_dict.items(), desc=f"device {device_id} loading file {file}", total=len(state_dict)):
        #             self.add_weight(device_id, nm, param.half())

        if os.path.exists(os.path.join(model_path, "model.safetensors.index.json")):
            weight_info = json.load(open(os.path.join(model_path, "model.safetensors.index.json")))
        else:
            weight_info = json.load(
                open(os.path.join(model_path, "pytorch_model.bin.index.json"), mode="r", encoding="utf8"))
        full_name_list = list(weight_info["weight_map"].values())
        total_param_ct = len(full_name_list)

        quant_prefix_list = []
        full_key_list = list(weight_info["weight_map"].keys())
        for k in full_key_list:
            if k.endswith(".qweight"):
                prefix = k[:-8]
                quant_prefix_list.append(prefix)
        
        model_json_path = os.path.join(model_path, "config.json")
        group_size = None
        if os.path.exists(model_json_path):
            config_ent = json.load(open(model_json_path, encoding="utf8"))
            if "quantization_config" in config_ent and "group_size" in config_ent["quantization_config"]:
                group_size = config_ent["quantization_config"]["group_size"] # group size should be specified in json.
            elif "hidden_size" in config_ent:
                hidden_size = config_ent["hidden_size"]

        file_names = list(sorted(set(full_name_list)))
        param_it = 1
        for file in file_names:
            if str(os.path.join(model_path, file)).__contains__("safetensor"):
                with safe_open(os.path.join(model_path, file), framework="pt", device='cpu') as f:
                    for nm in f.keys():
                        print(f'\rloading parameters {param_it}/{total_param_ct}',end='')
                        param = f.get_tensor(nm)
                        self.add_weight(device_id, nm, param, qlinear_prefix=quant_prefix_list, quant_group_size=group_size)
                        param_it += 1
            else:
                state_dict = torch.load(open(os.path.join(model_path, file), mode="rb"), map_location="cpu")
                for nm, param in tqdm(state_dict.items(), desc=f"device {device_id} loading file {file}", total=len(state_dict)):
                    self.add_weight(device_id, nm, param, qlinear_prefix=quant_prefix_list, quant_group_size=group_size)
        return
    
    def load_lora_params(self, adapter_name, lora_path):
        self.add_lora_cfg(os.path.join(lora_path, "adapter_config.json"), adapter_name=adapter_name)
        device_id = -1
        def add_branch_name(_nm, _added):
            part_list = list(_nm.split("."))
            for _pos, part in enumerate(part_list):
                if part.startswith("lora_"):
                    part_list.insert(_pos+1, _added.strip())
                    return ".".join(part_list)
            return _nm
        if os.path.exists(os.path.join(lora_path, "adapter_model.bin")):
            state_dict = torch.load(open(os.path.join(lora_path, "adapter_model.bin"), mode="rb"), map_location="cpu")
        else:
            state_dict = {}
            with safe_open(os.path.join(lora_path, "adapter_model.safetensors"), framework="pt", device='cpu') as f:
                for nm in f.keys():
                    param = f.get_tensor(nm)
                    state_dict[nm] = param
        for nm, param in tqdm(state_dict.items(), desc=f"device {device_id} loading lora {adapter_name}",
                            total=len(state_dict)):
            branched_name = add_branch_name(nm, adapter_name)
            self.add_weight(device_id, branched_name, param.float())
        return
    
    def start_loops(self):
        liteqwen_lib.start_loops()
    
    def get_generated(self, request_id, is_incremental=True, force_interupt=False, no_stream=False, return_logits=False):

        if not return_logits:
            result_info = liteqwen_lib.get_frame(request_id.encode(), is_incremental, force_interupt, no_stream)
            status = result_info[0]
            seq_length = result_info[1]
            incremental_len = result_info[2]
            
            result_ids = []
            for i in range(incremental_len):
                result_ids.append(result_info[i+3])
            
            liteqwen_lib.free_frame(result_info)
            result = {"status":status, "current_length":seq_length, "token_ids":result_ids}
        else:
            result_ptr = liteqwen_lib.get_frame_entity(request_id.encode(), is_incremental, force_interupt, no_stream)
            data_len = int.from_bytes(ctypes.string_at(result_ptr, 4), "big", signed=False)
            result_raw = bytes(result_ptr[4:4+data_len])
            liteqwen_lib.free_frame_entity(result_ptr)
            try:
                result_str = result_raw.decode('utf-8')
                frame_ent = json.loads(result_str)
            except Exception as ex:
                print(ex)
                print(result_raw)
                raise ex
            
            result =  {"status": frame_ent["status"], "current_length":frame_ent["cur_len"], "token_ids":frame_ent["token_ids"]}

        if return_logits and "logits" in frame_ent:
            top_dict = OrderedDict()
            for i in range(len(frame_ent["top_pos"])):
                pos = frame_ent["top_pos"][i]
                if pos not in top_dict:
                    top_dict[pos] = []
                top_dict[pos].append({"lgt": frame_ent["logits"][i], "tid": frame_ent["top_ids"][i]})
            dict_list = []
            for _k, _v in top_dict.items():
                dict_list.append({"pos": _k, "tops": _v})
            result["logits"] = dict_list     
        return result