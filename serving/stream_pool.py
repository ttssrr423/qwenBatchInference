from multiprocessing import shared_memory
import json
import struct
import datetime
import numpy as np
from enum import Enum
from multiprocessing.resource_tracker import unregister

class GenState(Enum):
    PREPARING = 1
    GENERATING = 2
    END = 3

class RetState(Enum):
    FETCHED = 1
    YIELDING = 2
    END = 3
    TERMINATE = 4

class StreamPool():
    @classmethod
    def create_buffer(cls, max_queue_size, max_sequence_length, max_request_id_len, as_json=True):
        generate_state_len = 4
        comsume_state_len = 4
        time_len = 4
        # 一个中文char占用3个bytes，往高估计，平均每个token包含4个中文char，一共12*max_sequence_length. 再考虑可能json序列化新增其他信息，长度再*4
        result_len = 12 * max_sequence_length * 4

        record_len = time_len + generate_state_len + comsume_state_len + 4 + max_request_id_len + 4 + result_len
        total_size = max_queue_size * record_len

        out_result_buffer = shared_memory.SharedMemory(create=True, size=total_size)
        buff_name = out_result_buffer.name
        unregister(out_result_buffer._name, 'shared_memory')

        # int占用4个bytes，矩阵shape=[max_queue_size, max_sequence_length+32], record_id=row_id
        token_size = max_queue_size * (max_sequence_length+32) * 4
        out_token_buffer = shared_memory.SharedMemory(create=True, size=token_size)
        token_buf_name = out_token_buffer.name
        unregister(out_token_buffer._name, 'shared_memory')

        buffer_meta = {
            "name": buff_name,
            "max_qsize": max_queue_size,
            "record_stride": record_len,
            "time_offset": 0,
            "gen_state_offset": 4,
            "consume_state_offset": 8,
            "req_id_offset": 12,
            "results_offset": 12+4+max_request_id_len,
            "max_request_id_len": max_request_id_len,
            "token_buf_name": token_buf_name,
            "token_record_stride": max_sequence_length+32
        }

        for i in range(max_queue_size):
            init_ts = float(datetime.datetime.now().timestamp()) - 500.0
            init_ts_bytes = struct.pack('<f', init_ts) # struct.unpack('<f', x)
            record_offset = i * record_len
            assert len(init_ts_bytes) == 4
            out_result_buffer.buf[record_offset:record_offset+4] = init_ts_bytes

        # (999999999).to_bytes(length=4, byteorder="little", signed=True)
        # bytes("123-abc".encode("ascii"))
        # float_value = struct.unpack('<f', struct.pack('<f', 3.14))
        if as_json:
            return json.dumps(buffer_meta), out_result_buffer, out_token_buffer
        return buffer_meta, out_result_buffer, out_token_buffer

    def __init__(self, meta_info, overwrite_timeout=120, as_json=True):
        if as_json:
            meta_info = json.loads(meta_info)

        self.buf_name = meta_info["name"]
        self.buffer = shared_memory.SharedMemory(name=self.buf_name)
        self.max_qsize = meta_info["max_qsize"]
        self.record_stride = meta_info["record_stride"]
        self.time_offset = meta_info["time_offset"]
        self.gen_state_offset = meta_info["gen_state_offset"]
        self.consume_state_offset = meta_info["consume_state_offset"]
        self.req_id_offset = meta_info["req_id_offset"]
        self.results_offset = meta_info["results_offset"]
        self.max_request_id_len = meta_info["max_request_id_len"]
        self.record_pt = 0
        self.overwrite_timeout = overwrite_timeout

        self.token_buf_name = meta_info["token_buf_name"]
        self.token_buffer = shared_memory.SharedMemory(name=self.token_buf_name)
        self.token_record_stride = meta_info["token_record_stride"]
        self.ptr_move_stride = 1
        self.token_pool = np.ndarray(shape=(self.max_qsize, self.token_record_stride), dtype=np.int32,
                                buffer=self.token_buffer.buf)

        unregister(self.buffer._name, 'shared_memory')
        unregister(self.token_buffer._name, 'shared_memory')


    def set_dp(self, start, stride):
        self.record_pt = start
        self.ptr_move_stride = stride

    def write_int_to_buffer(self, val, offset):
        val_bytes = val.to_bytes(length=4, byteorder="little", signed=True)
        self.buffer.buf[offset:offset+4] = val_bytes
        return

    def write_string_to_buffer(self, val, offset, is_utf8=True):
        if is_utf8:
            val_bytes = val.encode("utf8")
        else:
            val_bytes = val.encode("ascii")
        byteslen = len(val_bytes)
        len_bytes = byteslen.to_bytes(length=4, byteorder="little", signed=True)
        writing_bytes = len_bytes + val_bytes
        self.buffer.buf[offset:offset+4+byteslen] = writing_bytes
        return

    def read_string_from_buffer(self, offset, is_utf8=True):
        charlen_tmp = bytes(self.buffer.buf[offset:offset+4])
        char_len = int.from_bytes(charlen_tmp, byteorder='little', signed=True)
        str_bytes = bytes(self.buffer.buf[offset+4:offset+4+char_len])
        if is_utf8:
            return str_bytes.decode("utf8")
        else:
            return str_bytes.decode("ascii")

    def read_int_from_buffer(self, offset):
        tmp = bytes(self.buffer.buf[offset:offset + 4])
        return int.from_bytes(tmp, byteorder='little', signed=True)

    def read_record(self, rid):
        record_data = bytes(self.buffer.buf[rid * self.record_stride: (rid+1) * self.record_stride])

        ts_bytes = bytes(record_data[self.time_offset:self.time_offset+4])
        tfloat = float(struct.unpack('<f', ts_bytes)[0])

        gen_state = int.from_bytes(record_data[self.gen_state_offset:self.gen_state_offset+4], byteorder='little', signed=True)
        ret_state = int.from_bytes(record_data[self.consume_state_offset:self.consume_state_offset+4], byteorder='little', signed=True)

        req_id_charlen = int.from_bytes(record_data[self.req_id_offset:self.req_id_offset+4], byteorder='little', signed=True)
        req_id = (record_data[self.req_id_offset+4:self.req_id_offset+4+req_id_charlen]).decode("ascii")

        res_charlen = int.from_bytes(record_data[self.results_offset:self.results_offset+4], byteorder='little', signed=True)
        result = (record_data[self.results_offset+4:self.results_offset+4+res_charlen]).decode("utf8")

        return (tfloat, GenState(gen_state), RetState(ret_state), req_id, result)

    def view_record(self, rid=-1, request_id=None):
        if rid < 0 and request_id is None:
            return (0.0, GenState.END, RetState.END, "", "")
        if rid < 0 and request_id is not None:
            for i in range(self.max_qsize):
                record_request = self.read_string_from_buffer(i * self.record_stride + self.req_id_offset, is_utf8=False)
                if record_request == request_id:
                    rid = i
                    break
        if rid < 0:
            return (0.0, GenState.END, RetState.END, "", "")
        return rid, self.read_record(rid)

    def set_request_to_stop(self, request_id, rid):
        record_request = self.read_string_from_buffer(rid * self.record_stride + self.req_id_offset, is_utf8=False)
        record_generating = self.read_int_from_buffer(rid*self.record_stride + self.consume_state_offset)
        self.token_pool[rid][0] = 1
        if record_request == request_id and record_generating!=RetState.END:
            self.set_ret_state(rid, RetState.END)
            return True
        return False

    def set_gen_state(self, rid, gen_state):
        offset = self.record_stride * rid + self.gen_state_offset
        self.write_int_to_buffer(gen_state.value, offset)
        return

    def set_ret_state(self, rid, ret_state):
        offset = self.record_stride * rid + self.consume_state_offset
        self.write_int_to_buffer(ret_state.value, offset)
        return

    def set_timeout(self, rid, delay=0.0):
        time_offset = self.record_stride * rid + self.time_offset
        if delay > 0:
            # 重置上次更新时间延迟delay秒生效
            new_ts = float(datetime.datetime.now().timestamp()) - self.overwrite_timeout + delay
        else:
            # 重置上次更新时间立即生效
            new_ts = float(datetime.datetime.now().timestamp()) - self.overwrite_timeout - 20
        self.buffer.buf[time_offset:time_offset + 4] = struct.pack('<f', new_ts)
        return

    def coro_set_timeout(self, rid, coro_id, coro_occupied, delay=0.0):
        time_offset = self.record_stride * rid + self.time_offset
        if delay > 0:
            # 重置上次更新时间延迟delay秒生效
            new_ts = float(datetime.datetime.now().timestamp()) - self.overwrite_timeout + delay
        else:
            # 重置上次更新时间立即生效
            new_ts = float(datetime.datetime.now().timestamp()) - self.overwrite_timeout - 20
        self.buffer.buf[time_offset:time_offset + 4] = struct.pack('<f', new_ts)

        coro_occupied[coro_id] = -1
        return

    def write_result(self, rid, text, is_first_frame=False):
        offset = self.record_stride * rid + self.results_offset
        self.write_string_to_buffer(text, offset)
        if is_first_frame:
            self.set_gen_state(rid, GenState.GENERATING)
        return

    def read_return_state(self, rid):
        offset = self.record_stride * rid + self.consume_state_offset
        ret_state = int.from_bytes(self.buffer.buf[offset:offset + 4], byteorder='little', signed=True)
        return RetState(ret_state)

    def refresh_time(self, rid):
        record_offset = rid * self.record_stride
        time_offset = record_offset + self.time_offset
        cur_ts = float(datetime.datetime.now().timestamp())
        self.buffer.buf[time_offset:time_offset + 4] = struct.pack('<f', cur_ts)

    def wait_for_start(self, req_id):
        if len(req_id) > self.max_request_id_len:
            req_id = req_id[:self.max_request_id_len]

        trial_num = 0
        while True:
            cur_ts = float(datetime.datetime.now().timestamp())
            self.record_pt = (self.record_pt + self.ptr_move_stride) % self.max_qsize
            record_offset = self.record_pt * self.record_stride
            time_offset = record_offset + self.time_offset
            ts_bytes = bytes(self.buffer.buf[time_offset:time_offset + 4])
            prev_updated_ts = float(struct.unpack('<f', ts_bytes)[0])
            if (cur_ts - prev_updated_ts) > self.overwrite_timeout:
                # 通过刷新prev_update_time来占用record_id
                self.buffer.buf[time_offset:time_offset + 4] = struct.pack('<f', cur_ts)
                self.write_int_to_buffer(GenState.PREPARING.value, record_offset+self.gen_state_offset)
                self.write_int_to_buffer(RetState.FETCHED.value, record_offset + self.consume_state_offset)
                self.write_string_to_buffer(req_id, record_offset + self.req_id_offset, is_utf8=False)
                return self.record_pt

            # trial_num += 1
            # if trial_num >= self.max_qsize/self.ptr_move_stride: # 可能出于未知原因，资源已满，适当降低超时时间，并尝试中断其他生成。
            #     if (cur_ts - prev_updated_ts) > 120.0:
            #         print("PYTHON POOL WARNING: too much records occupied, maybe check for deletion after eos.")
            #         self.write_int_to_buffer(RetState.END.value, record_offset+self.consume_state_offset)
            #         trial_num = 0

    def coro_wait_for_start(self, req_id, coro_occupied_list, prev_rid):
        if len(req_id) > self.max_request_id_len:
            req_id = req_id[:self.max_request_id_len]
        try_rid = prev_rid
        while True:
            cur_ts = float(datetime.datetime.now().timestamp())
            try_rid = (try_rid + self.ptr_move_stride) % self.max_qsize
            record_offset = try_rid * self.record_stride
            time_offset = record_offset + self.time_offset
            ts_bytes = bytes(self.buffer.buf[time_offset:time_offset + 4])
            prev_updated_ts = float(struct.unpack('<f', ts_bytes)[0])
            if (cur_ts - prev_updated_ts) > self.overwrite_timeout:
                no_conflict = True
                for occupied_id in coro_occupied_list:
                    if occupied_id == try_rid:
                        no_conflict = False
                if not no_conflict:
                    continue

                # 通过刷新prev_update_time来占用record_id
                self.buffer.buf[time_offset:time_offset + 4] = struct.pack('<f', cur_ts)
                self.write_int_to_buffer(GenState.PREPARING.value, record_offset+self.gen_state_offset)
                self.write_int_to_buffer(RetState.FETCHED.value, record_offset + self.consume_state_offset)
                self.write_string_to_buffer(req_id, record_offset + self.req_id_offset, is_utf8=False)
                return try_rid
