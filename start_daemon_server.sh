# 增加文件写入上限
ulimit -SHn 5120
# 清除被占用的shared memory
rm /dev/shm/*
nohup python daemon_start_server.py &