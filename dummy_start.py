import platform
import os
import ctypes
if platform.system() == 'Windows':
    liteqwen_lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.split(os.path.realpath(__file__))[0], "liteqwen_conn.dll"))
else:
    abs_path = "/home/st491/anaconda3/envs/server/lib/python3.9/site-packages/liteqwen_py-0.0.1-py3.9.egg/liteqwen_py"
    # liteqwen_lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.split(os.path.realpath(__file__))[0], "libliteqwen_conn.so"))
    liteqwen_lib = ctypes.cdll.LoadLibrary(os.path.join(abs_path, "libliteqwen_conn.so"))

liteqwen_lib.initialize_empty_qwen2.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

if __name__ == "__main__":
    layer_num = 40
    layer_to_device_list = [0] * 40
    max_dynamic_bsz = 16
    max_length = 4096
    timeout_in_secs = 200

    liteqwen_lib.initialize_empty_qwen2(1, 1, 1, "/mnt/e/UbuntuFiles/models_saved/qwen15_14b_chat_int4_gptq_bin".encode(),
                                        layer_num, (ctypes.c_int * layer_num)(*layer_to_device_list), max_dynamic_bsz,
                                        max_length, 1 * max_dynamic_bsz * 5,
                                        int(timeout_in_secs * 1000))