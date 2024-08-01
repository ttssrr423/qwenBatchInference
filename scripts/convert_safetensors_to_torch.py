dst_path = "/mnt/e/UbuntuFiles/models_saved/qwen2_7b_int4_gptq_bin"
src_path = "/mnt/e/UbuntuFiles/models_saved/qwen2_7b_int4_gptq"

torch_name = "pytorch_model-0000{0}-of-0000{1}.bin"
safetensor_name = "model-0000{0}-of-0000{1}.safetensors"

import torch
import os
import json
from safetensors import safe_open

def start_convert():
    if os.path.exists(os.path.join(src_path, "model.safetensors.index.json")):
        weight_info = json.load(open(os.path.join(src_path, "model.safetensors.index.json")))
        w_map = weight_info["weight_map"]
        new_map = {}
        for _k, _v in w_map.items():
            segs = _v.split("-")
            f1, f2 = segs[1].replace("0", ""), (segs[3].split(".")[0]).replace("0", "")
            new_name = torch_name.format(f1, f2)
            new_map[_k] = new_name

        weight_info["weight_map"] = new_map
        json.dump(weight_info, open(os.path.join(dst_path, "pytorch_model.bin.index.json"),  mode="w"), indent=4)

    # file_names = ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]
    file_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    file_ct = 0
    for file in file_names:
        safetensor_file = os.path.join(src_path, file)
        with safe_open(safetensor_file, framework="pt", device='cpu') as f:
            state_dict = {}
            file_ct += 1
            for nm in f.keys():
                # print(f'\rloading parameters {param_it}/{total_param_ct}', end='')
                print(f"converting {nm}")
                param = f.get_tensor(nm)
                state_dict[nm] = param
            par_len = len(state_dict)
            bin_file_name = torch_name.format(file_ct, len(file_names))
            print(f"saving torch bin {bin_file_name} with {par_len} params")
            torch.save(state_dict, open(os.path.join(dst_path, bin_file_name), mode="wb"))


start_convert()