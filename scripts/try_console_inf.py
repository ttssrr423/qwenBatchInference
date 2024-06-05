from qwen2 import Qwen2ForCausalLM, Qwen2Config, Qwen2TokenizerFast
from peft import PeftModel
device = "cuda:0"
model_path = "/mnt/e/UbuntuFiles/models_saved/qwen15_14b_chat_int4_gptq_bin"
# lora_path =  "/mnt/e/UbuntuFiles/models_saved/luban_instruction_dpo"

def inference():
    model = Qwen2ForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    # model = PeftModel.from_pretrained(model, lora_path)

    tokenizer = Qwen2TokenizerFast.from_pretrained(model_path, trust_remote_code=True)

    inp_str = input("输入指令内容：\n")
    while inp_str != "q":
        prompt = inp_str.replace("\\\\n", "\n") # "你好啊，你是谁？"
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
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
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)

        inp_str = input("输入指令内容：\n")
    return

if __name__ == "__main__":
    inference()