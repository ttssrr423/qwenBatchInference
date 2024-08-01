try:
    from qwen2 import Qwen2ForCausalLM, Qwen2Config, Qwen2TokenizerFast
except:
    from qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
    from qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from qwen2.configuration_qwen2 import Qwen2Config

from peft import PeftModel
device = "cuda:0"
import json
from transformers import GenerationConfig

model_path = "/mnt/e/UbuntuFiles/models_saved/qwen15_14b_chat_int4_gptq_newbin"
# model_path = "/mnt/e/UbuntuFiles/models_saved/qwen2_7b_int4_gptq_bin"
lora_path = "/mnt/e/UbuntuFiles/models_saved/ppt_outline_epoch3"
default_gen_config = GenerationConfig()
default_gen_config.temperature = 0.01

from auto_gptq import AutoGPTQForCausalLM

default_messages = [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好，我是小PAI，是人工智能机器人。"},
                    {"role": "user", "content": "你可以做什么？"},
                ]

def inference():
    model = Qwen2ForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)

    tokenizer = Qwen2TokenizerFast.from_pretrained(model_path, trust_remote_code=True)

    inp_str = input("输入指令内容：\n")
    while inp_str != "q":
        if inp_str.startswith("FILE:"):
            read_str = open(inp_str[5:], encoding="utf8").read()
            try:
                messages = json.loads(read_str)
            except:
                try:
                    messages = eval(read_str)
                except:
                    print("not json serializable and not evaluable:")
                    print(read_str)
                    continue
        elif inp_str.startswith("#DEFAULT"):
            messages = default_messages
        else:
            prompt = inp_str.replace("\\\\n", "\n") # "�| 好�~U~J�~L�| �~X��~A�~_"
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]


        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(text)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        id_list = [int(x) for x in list(model_inputs["input_ids"][0].clone().detach().cpu().numpy())]
        print(id_list)
        token_list = tokenizer.convert_ids_to_tokens(id_list)
        print([(x,y) for x,y in zip(id_list, token_list)])
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            generation_config=default_gen_config
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_ids[0])
        print(response)

        inp_str = input("输入指令内容：\n")
    return

if __name__ == "__main__":
    inference()