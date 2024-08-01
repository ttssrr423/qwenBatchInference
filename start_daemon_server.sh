ulimit -SHn 5120
rm /dev/shm/*
nohup python daemon_start_server.py &