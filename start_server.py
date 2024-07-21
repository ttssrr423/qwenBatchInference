import uvicorn
from serving.app import app
import multiprocessing
import time
import requests
from global_config import NUM_GPUS, PORT, GLOBAL_CONFIG

def start_server_process():
    def run_server():
        host = "0.0.0.0"
        port = PORT
        uvicorn.run(app, host=host, port=port)

    p = multiprocessing.Process(target=run_server, args=())
    p.start()
    return p

if __name__ == "__main__":
    server_process = start_server_process()
    time.sleep(1.0)
    init_res = requests.get(f"http://127.0.0.1:{PORT}/_init")
    server_process.join()
