import datetime
import json
import shutil
import tempfile
import urllib.parse
import random
import hashlib
import requests
from flask import Flask, render_template, request, session, send_file, make_response
import os
import uuid
from serving.LRU_cache import LRUCache
import threading
import pickle
import asyncio
from global_config import GLOBAL_CONFIG
# import yaml

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

LOCAL_SERVER_ADDR = "127.0.0.1:"+str(GLOBAL_CONFIG["server_port"])
PORT = str(GLOBAL_CONFIG["web_port"])
CHAT_CONTEXT_NUMBER_MAX = 7 # 连续对话模式下的上下文最大数量 n，即开启连续对话模式后，将上传本条消息以及之前你和GPT对话的n-1条消息
USER_SAVE_MAX = 12 # 设置最多存储n个用户，当用户过多时可适当调大

if os.getenv("DEPLOY_ON_RAILWAY") is not None or os.getenv("DEPLOY_ON_ZEABUR"):  # 如果是云部署，需要删除代理
    os.environ.pop('HTTPS_PROXY', None)

LOCAL_SERVER_ADDR = os.getenv("LOCAL_SERVER_ADDR", default=LOCAL_SERVER_ADDR)  # 如果环境变量中设置了LOCAL_SERVER_ADDR，则使用环境变量中的OPENAI_API_KEY
PORT = os.getenv("PORT", default=PORT)  # 如果环境变量中设置了PORT，则使用环境变量中的PORT

STREAM_FLAG = True  # 是否开启流式推送
USER_DICT_FILE = "all_user_dict_v3.pkl"  # 用户信息存储文件（包含版本）
lock = threading.Lock()  # 用于线程锁

project_info = "## 本地serving服务    \n" \
               "\n"

def get_hash(_text):
    hasher = hashlib.md5()
    tm = datetime.datetime.now()
    rand_num = str(random.random())[-4:]
    text_with_time = _text + tm.strftime("%Y-%m-%d %H:%M:%S.%f") + rand_num
    hasher.update(text_with_time.encode(encoding="utf-8"))
    return hasher.hexdigest(), tm


global user_defined_params
user_defined_params = dict()
if "gen_kwargs" not in user_defined_params:
    user_defined_params["gen_kwargs"] = {}

def get_response_stream_generate_from_local_api(message_history_complete):
    """
    从ChatGPT API获取回复
    :param apikey:
    :param message_context: 上下文
    :param message_history: 消息历史
    :param model: 模型
    :param temperature: 温度
    :param presence_penalty: 惩罚
    :param max_tokens: 最大token数量
    :return: 回复生成器
    """

    headers = {"Content-Type":"application/json", "cache-control":"no-cache"}
    message_history = [x for x in message_history_complete if x["role"] in ["assistant", "user"] and x["content"] not in ["""## 本地serving服务    
    
""", """创建新的用户id成功，请牢记该id"""]]
    query = message_history[-1]["content"]
    pair_buffer = ["", ""]
    history = []

    use_hist = user_defined_params.get("use_history", True)
    if use_hist:
        next_role = "user"
        for text_id in range(len(message_history)-2, -1, -1):
            hist_item = message_history[text_id]
            if hist_item["role"] == "assistant" and next_role == "user":
                pair_buffer[1] = hist_item["content"]
                next_role = "assistant"
            elif hist_item["role"] == "user" and next_role == "assistant":
                pair_buffer[0] = hist_item["content"]
                next_role = "user"
                history.insert(0, pair_buffer)
                pair_buffer = ["", ""]
            else:
                continue

    fixed_history = [{"role":"system", "metadata":"", "context":"你是一个乐于助人的AI助手。"}]
    for pair_item in history:
        fixed_history.append({"role":"user", "metadata":"", "content":pair_item[0]})
        fixed_history.append({"role":"assistant", "metadata":"", "content":pair_item[1]})

    request_id, start_tm = get_hash(query)

    temp = user_defined_params.get("temperature", GLOBAL_CONFIG["temperature"])
    kwargs = {}
    return_raw = False
    if "gen_kwargs" in user_defined_params:
        kwargs.update(user_defined_params["gen_kwargs"])
        if "adapter_name" in user_defined_params["gen_kwargs"]:
            return_raw = True
    kwargs.update({"temperature":temp, "return_raw":return_raw})
    data = {"query":query, "history":history, "request_id":request_id, "gen_kwargs":kwargs}

    print("开始流式请求")
    url = "http://" + LOCAL_SERVER_ADDR + "/stream_chat_post"
    # 请求接收流式数据 动态print
    try:
        response = requests.post(url, data=json.dumps(data),
                                   stream=True)

        def generate():
            stream_content = str()
            one_message = {"role": "assistant", "content": stream_content}
            message_history_complete.append(one_message)

            residual = b''
            msg_ct = 0
            content_buffer = ""
            incremental_text_recorder = ""
            for line in response.iter_lines():
                # print(line)
                # =============json decode =================
                res_raw = line
                if res_raw.startswith(b'data:'):
                    res_ent = json.loads(res_raw[5:])
                    res_type = "data"
                else:
                    res_ent = {}
                    res_type = b"delta, raw=" + res_raw
                if "text" in res_ent:
                    if len(res_ent["text"]) > len(incremental_text_recorder):
                        yield res_ent["text"][len(incremental_text_recorder):]
                        incremental_text_recorder = res_ent["text"]
                    else:
                        pass
                    one_message['content'] = res_ent["text"]
                continue

    except Exception as e:
        ee = e

        def generate():
            yield "request error:\n" + str(ee)

    return generate


# def handle_messages_get_response_stream(message, apikey, message_history, have_chat_context, chat_with_history):
#     message_history.append({"role": "user", "content": message})
#     asyncio.run(save_all_user_dict())
#     message_context = get_message_context(message_history, have_chat_context, chat_with_history)
#     generate = get_response_stream_generate_from_ChatGPT_API(message_context, apikey, message_history)
#     return generate


def check_session(current_session):
    """
    检查session，如果不存在则创建新的session
    :param current_session: 当前session
    :return: 当前session
    """
    if current_session.get('session_id') is not None:
        print("existing session, session_id:\t", current_session.get('session_id'))
    else:
        current_session['session_id'] = uuid.uuid1()
        print("new session, session_id:\t", current_session.get('session_id'))
    return current_session['session_id']


def check_user_bind(current_session):
    """
    检查用户是否绑定，如果没有绑定则重定向到index
    :param current_session: 当前session
    :return: 当前session
    """
    if current_session.get('user_id') is None:
        return False
    return True


def get_user_info(user_id):
    """
    获取用户信息
    :param user_id: 用户id
    :return: 用户信息
    """
    lock.acquire()
    user_info = all_user_dict.get(user_id)
    lock.release()
    return user_info


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    主页
    :return: 主页
    """
    check_session(session)
    return render_template('index.html')


@app.route('/loadHistory', methods=['GET', 'POST'])
def load_messages():
    """
    加载聊天记录
    :return: 聊天记录
    """
    check_session(session)
    success, message = auth(request.headers, session)
    code = 200  # 200表示云端存储了 node.js改写时若云端不存储则返回201
    if not success:
        return {"code": code, "data": [{"role": "web-system", "content": message}]}
    if session.get('user_id') is None:
        messages_history = [{"role": "assistant", "content": project_info},
                            {"role": "assistant", "content": "#### 当前浏览器会话为首次请求\n"
                                                             "#### 请输入已有用户`id`或创建新的用户`id`。\n"
                                                             "- 已有用户`id`请在输入框中直接输入\n"
                                                             "- 创建新的用户`id`请在输入框中输入`new:xxx`,其中`xxx`为你的自定义id，请牢记\n"
                                                             "- 输入`设置lora=default`或`设置温度=0.01`或`设置多轮=N`可以修改调用参数。\n"
                                                             "- 具体可以输入`帮助`以获取帮助提示"}]
    else:
        user_info = get_user_info(session.get('user_id'))
        chat_id = user_info['selected_chat_id']
        messages_history = user_info['chats'][chat_id]['messages_history']
        print(f"用户({session.get('user_id')})加载聊天记录，共{len(messages_history)}条记录")
    return {"code": code, "data": messages_history, "message": ""}


@app.route('/downloadUserDictFile', methods=['GET', 'POST'])
def download_user_dict_file():
    """
    下载用户字典文件
    :return: 用户字典文件
    """
    check_session(session)
    if request.headers.get("admin-password") is None:
        success, message = auth(request.headers, session)
        if not success:
            return "未授权，无法下载"
        user_id = request.headers.get("user-id")
        if user_id is None:
            return "未绑定用户，无法下载"
        select_user_dict = LRUCache(USER_SAVE_MAX)
        lock.acquire()
        select_user_dict.put(user_id, all_user_dict.get(user_id))
        lock.release()
        # 存储为临时文件再发送出去
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False, mode='wb') as temp_file:
            # 将 Python 对象使用 pickle 序列化保存到临时文件中
            pickle.dump(select_user_dict, temp_file)
        response = make_response(send_file(temp_file.name, as_attachment=True))
        response.headers["Content-Disposition"] = f"attachment; filename={user_id}_of_{USER_DICT_FILE}"
        response.call_on_close(lambda: os.remove(temp_file.name))
        return response

    else:
        response = make_response(send_file(USER_DICT_FILE, as_attachment=True))
        response.headers["Content-Disposition"] = f"attachment; filename={USER_DICT_FILE}"
        return response


def backup_user_dict_file():
    """
    备份用户字典文件
    :return:
    """
    backup_file_name = USER_DICT_FILE.replace(".pkl", f"_buckup_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.pkl")
    shutil.copy(USER_DICT_FILE, backup_file_name)
    print(f"备份用户字典文件{USER_DICT_FILE}为{backup_file_name}")


@app.route('/uploadUserDictFile', methods=['POST'])
def upload_user_dict_file():
    """
    上传用户字典文件 并合并记录
    :return:
    """
    check_session(session)
    file = request.files.get('file')        # 获取上传的文件
    if file:
        if request.headers.get("admin-password") is None:
            success, message = auth(request.headers, session)
            if not success:
                return "未授权，无法合并用户记录"
            user_id = request.headers.get("user-id")
            if user_id is None:
                return "未绑定用户，无法合并用户记录"
            if not file.filename.endswith(".pkl"):
                return "上传文件格式错误，无法合并用户记录"

            # 读取获取的文件
            upload_user_dict = ""
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False, mode='wb') as temp_file:
                file.save(temp_file.name)
            # 将 Python 对象使用 pickle 序列化保存到临时文件中
            try:
                with open(temp_file.name, 'rb') as temp_file:
                    upload_user_dict = pickle.load(temp_file)
            except:
                return "上传文件格式错误，无法解析以及合并用户记录"
            finally:
                os.remove(temp_file.name)
            # 判断是否为LRUCache对象
            if not isinstance(upload_user_dict, LRUCache):
                return "上传文件格式错误，无法合并用户记录"
            lock.acquire()
            user_info = all_user_dict.get(user_id)
            lock.release()
            upload_user_info = upload_user_dict.get(user_id)
            if user_info is None or upload_user_info is None:
                return "仅能合并相同用户id的记录，请确保所上传的记录与当前用户id一致"
            backup_user_dict_file()
            for chat_id in upload_user_info['chats'].keys():
                if user_info['chats'].get(chat_id) is None:
                    user_info['chats'][chat_id] = upload_user_info['chats'][chat_id]
                else:
                    new_chat_id = str(uuid.uuid1())
                    user_info['chats'][new_chat_id] = upload_user_info['chats'][chat_id]
            asyncio.run(save_all_user_dict())
            return '个人用户记录合并完成'
        else:
            if not file.filename.endswith(".pkl"):
                return "上传文件格式错误，无法上传用户记录"
            # 读取获取的文件
            upload_user_dict = ""
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False, mode='wb') as temp_file:
                file.save(temp_file.name)
            # 将 Python 对象使用 pickle 序列化保存到临时文件中
            try:
                with open(temp_file.name, 'rb') as temp_file:
                    upload_user_dict = pickle.load(temp_file)
            except:
                return "上传文件格式错误，无法解析以及合并用户记录"
            finally:
                os.remove(temp_file.name)
            # 判断是否为LRUCache对象
            if not isinstance(upload_user_dict, LRUCache):
                return "上传文件格式错误，无法合并用户记录"
            backup_user_dict_file()
            lock.acquire()
            for user_id in list(upload_user_dict.keys()):
                if all_user_dict.get(user_id) is None:
                    all_user_dict.put(user_id, upload_user_dict.get(user_id))
                else:
                    for chat_id in upload_user_dict.get(user_id)['chats'].keys():
                        if all_user_dict.get(user_id)['chats'].get(chat_id) is None:
                            all_user_dict.get(user_id)['chats'][chat_id] = upload_user_dict.get(user_id)['chats'][chat_id]
                        else:
                            new_chat_id = str(uuid.uuid1())
                            all_user_dict.get(user_id)['chats'][new_chat_id] = upload_user_dict.get(user_id)['chats'][chat_id]
            lock.release()
            asyncio.run(save_all_user_dict())
            return '所有用户记录合并完成'
    else:
        return '文件上传失败'


def auth(request_head, session):
    """
    验证用户身份
    :param request_head: 请求头
    :param session: session
    :return: 验证结果
    """
    user_id = request_head.get("user-id")
    user_id = urllib.parse.unquote(user_id)
    password = request_head.get("password")
    apikey = request_head.get("api-key")

    user_info = get_user_info(user_id)

    if user_info is not None:
        session['user_id'] = user_id
        if apikey is not None and len(apikey) > 1 and apikey != "null" and apikey != "undefined":
            user_info['apikey'] = apikey
        else:
            user_info['apikey'] = None
        return True, "success"
    else:
        if session.get('user_id') is not None:
            del session['user_id']
        return False, "用户不存在，请在设置中填写正确的用户id，或发送new:xxx创建新的用户，其中xxx为你的自定义id"


@app.route('/loadChats', methods=['GET', 'POST'])
def load_chats():
    """
    加载聊天联系人
    :return: 聊天联系人
    """
    check_session(session)
    success, message = auth(request.headers, session)

    if not check_user_bind(session) or not success:
        chats = []
    else:
        user_info = get_user_info(session.get('user_id'))
        chats = []
        for chat_id, chat_info in user_info['chats'].items():
            if chat_info['chat_with_history']:
                mode = "continuous"
            else:
                mode = "normal"
            chats.append(
                {"id": chat_id, "name": chat_info['name'], "selected": chat_id == user_info['selected_chat_id'],
                 "mode": mode, "messages_total": len(user_info['chats'][chat_id]['messages_history'])})
    code = 200  # 200表示云端存储了 node.js改写时若云端不存储则返回201
    return {"code": code, "data": chats, "message": ""}


def new_chat_dict(user_id, name, send_time):
    return {"chat_with_history": False,
            "have_chat_context": 0,  # 从每次重置聊天模式后开始重置一次之后累计
            "name": name,
            "messages_history": [{"role": "assistant", "content": project_info},
                                 {"role": "web-system", "content": f"当前对话的用户id为{user_id}"},
                                 {"role": "web-system", "content": send_time},
                                 {"role": "web-system", "content": f"你已添加了{name}，现在可以开始聊天了。"},
                                 ]}


def new_user_dict(user_id, send_time):
    chat_id = str(uuid.uuid1())
    user_dict = {"chats": {chat_id: new_chat_dict(user_id, "默认对话", send_time)},
                 "selected_chat_id": chat_id,
                 "default_chat_id": chat_id}

    user_dict['chats'][chat_id]['messages_history'].insert(1, {"role": "assistant",
                                                               "content": "创建新的用户id成功，请牢记该id"})
    return user_dict



@app.route('/returnMessage', methods=['GET', 'POST'])
def return_message():
    """
    获取用户发送的消息，调用get_chat_response()获取回复，返回回复，用于更新聊天框
    :return:
    """
    check_session(session)
    request_data = request.get_json()

    success, message = auth(request.headers, session)
    if not success:
        session.clear()

    messages = request_data.get("messages")
    max_tokens = request_data.get("max_tokens")
    model = request_data.get("model")
    temperature = request_data.get("temperature")
    stream = request_data.get("stream")
    continuous_chat = request_data.get("continuous_chat")
    save_message = request_data.get("save_message")

    send_message = messages[-1].get("content")
    send_time = messages[-1].get("send_time")
    display_time = bool(messages[-1].get("display_time"))
    url_redirect = {"url_redirect": "/", "user_id": None}
    if send_message == "帮助":
        return "### 帮助\n" \
               "1. 输入`new:xxx`创建新的用户id\n " \
               "2. 输入`id:your_id`切换到已有用户id，新会话时无需加`id:`进入已有用户\n" \
               "3. 输入`rename_id:xxx`可将当前用户id更改\n" \
               "4. 输入`设置温度=0.01`可以修改后续对话温度\n" \
               "5. 输入`设置lora=skip/default/xxx`可以修改后续使用的lora名称（skip=原版）\n" \
               "6. 输入`设置多轮=Y/N`可以修改是否多轮\n" \
               "7. 相关设置也可以在设置面板中进行设置\n" \
               "8. 输入`帮助`查看帮助信息"
    if session.get('user_id') is None:  # 如果当前session未绑定用户
        print("当前会话为首次请求，用户输入:\t", send_message)
        if send_message.startswith("new:"):
            user_id = send_message.split(":")[1]
            url_redirect["user_id"] = user_id
            if user_id in all_user_dict:
                session['user_id'] = user_id
                return url_redirect
            user_dict = new_user_dict(user_id, send_time)
            lock.acquire()
            all_user_dict.put(user_id, user_dict)  # 默认普通对话
            lock.release()
            print("创建新的用户id:\t", user_id)
            session['user_id'] = user_id
            url_redirect["user_id"] = user_id
            return url_redirect
        else:
            user_id = send_message
            user_info = get_user_info(user_id)
            if user_info is None:
                return "用户id不存在，请重新输入或创建新的用户id"
            else:
                session['user_id'] = user_id
                print("已有用户id:\t", user_id)
                # 重定向到index
                url_redirect["user_id"] = user_id
                return url_redirect
    else:  # 当存在用户id时
        if send_message.startswith("id:"):
            user_id = send_message.split(":")[1].strip()
            user_info = get_user_info(user_id)
            if user_info is None:
                return "用户id不存在，请重新输入或创建新的用户id"
            else:
                session['user_id'] = user_id
                url_redirect["user_id"] = user_id
                print("切换到已有用户id:\t", user_id)
                # 重定向到index
                return url_redirect
        elif send_message.startswith("new:"):
            user_id = send_message.split(":")[1]
            if user_id in all_user_dict:
                return "用户id已存在，请重新输入或切换到已有用户id"
            session['user_id'] = user_id
            url_redirect["user_id"] = user_id
            user_dict = new_user_dict(user_id, send_time)
            lock.acquire()
            all_user_dict.put(user_id, user_dict)
            lock.release()
            print("创建新的用户id:\t", user_id)
            return url_redirect
        elif send_message.startswith("delete:"):  # 删除用户
            user_id = send_message.split(":")[1]
            if user_id != session.get('user_id'):
                return "只能删除当前会话的用户id"
            else:
                lock.acquire()
                all_user_dict.delete(user_id)
                lock.release()
                session['user_id'] = None
                print("删除用户id:\t", user_id)
                # 异步存储all_user_dict
                asyncio.run(save_all_user_dict())
                return url_redirect
        elif send_message.startswith("set_apikey:"):
            apikey = send_message.split(":")[1]
            user_info = get_user_info(session.get('user_id'))
            user_info['apikey'] = apikey
            # TODO 前端未存储
            print("设置用户专属apikey:\t", apikey)
            return "设置用户专属apikey成功"
        elif send_message.startswith("rename_id:"):
            new_user_id = send_message.split(":")[1]
            user_info = get_user_info(session.get('user_id'))
            if new_user_id in all_user_dict:
                return "用户id已存在，请重新输入"
            else:
                lock.acquire()
                all_user_dict.delete(session['user_id'])
                all_user_dict.put(new_user_id, user_info)
                lock.release()
                session['user_id'] = new_user_id
                asyncio.run(save_all_user_dict())
                print("修改用户id:\t", new_user_id)
                url_redirect["user_id"] = new_user_id
                return url_redirect
        elif send_message.startswith("设置温度"):
            try:
                temperature = float(send_message[5:])
                user_defined_params["temperature"] = temperature
                return "开始使用温度"+str(temperature)
            except:
                temperature = GLOBAL_CONFIG["temperature"]
                user_defined_params["temperature"] = temperature
                return f"温度解析失败，使用默认{temperature}"
        elif len(send_message)>6 and (send_message[:6]).lower()=="设置lora":
            lora_name = send_message[7:]
            if lora_name == "skip":
                user_defined_params["gen_kwargs"]["skip_lora"] = True
            else:
                user_defined_params["gen_kwargs"]["adapter_name"] = lora_name
            return "lora设置完毕："+lora_name
        elif send_message.startswith("设置多轮"):
            try:
                use_history = (send_message[5]).upper() == "Y"
                user_defined_params["use_history"] = use_history
                rep = "多" if use_history else "单"
                return f"设置完毕，开始{rep}轮对话"
            except:
                use_history = True
                user_defined_params["use_history"] = use_history
                return "解析多轮设置失败，默认使用多轮。"
        else:  # 处理聊天数据
            user_id = session.get('user_id')
            print(f"用户({user_id})发送消息:{send_message}")
            user_info = get_user_info(user_id)
            chat_id = user_info['selected_chat_id']
            messages_history = user_info['chats'][chat_id]['messages_history']
            chat_with_history = user_info['chats'][chat_id]['chat_with_history']
            apikey = user_info.get('apikey')
            if chat_with_history:
                user_info['chats'][chat_id]['have_chat_context'] += 1
            if display_time:
                messages_history.append({'role': 'web-system', "content": send_time})
            for m in messages:
                keys = list(m.keys())
                for k in keys:
                    if k not in ['role', 'content']:
                        del m[k]

            if save_message:
                messages_history.append(messages[-1])
            asyncio.run(save_all_user_dict())
            if not save_message:
                messages_history = []
            generate = get_response_stream_generate_from_local_api(messages_history)
            return app.response_class(generate(), mimetype='application/json')


async def save_all_user_dict():
    """
    异步存储all_user_dict
    :return:
    """
    await asyncio.sleep(0)
    lock.acquire()
    with open(USER_DICT_FILE, "wb") as f:
        pickle.dump(all_user_dict, f)
    # print("all_user_dict.pkl存储成功")
    lock.release()


@app.route('/selectChat', methods=['GET'])
def select_chat():
    """
    选择聊天对象
    :return:
    """
    chat_id = request.args.get("id")
    check_session(session)
    if not check_user_bind(session):
        return {"code": -1, "msg": "请先创建或输入已有用户id"}
    user_id = session.get('user_id')
    user_info = get_user_info(user_id)
    user_info['selected_chat_id'] = chat_id
    return {"code": 200, "msg": "选择聊天对象成功"}


@app.route('/newChat', methods=['GET'])
def new_chat():
    """
    新建聊天对象
    :return:
    """
    name = request.args.get("name")
    time = request.args.get("time")
    new_chat_id = request.args.get("chat_id")
    check_session(session)
    if not check_user_bind(session):
        return {"code": -1, "msg": "请先创建或输入已有用户id"}
    user_id = session.get('user_id')
    user_info = get_user_info(user_id)
    # new_chat_id = str(uuid.uuid1())
    user_info['selected_chat_id'] = new_chat_id
    user_info['chats'][new_chat_id] = new_chat_dict(user_id, name, time)
    print("新建聊天对象")
    return {"code": 200, "data": {"name": name, "id": new_chat_id, "selected": True,
                                  "messages_total": len(user_info['chats'][new_chat_id]['messages_history'])}}


@app.route('/deleteHistory', methods=['GET'])
def delete_history():
    """
    清空上下文
    :return:
    """
    check_session(session)
    if not check_user_bind(session):
        print("请先创建或输入已有用户id")
        return {"code": -1, "msg": "请先创建或输入已有用户id"}
    user_info = get_user_info(session.get('user_id'))
    chat_id = user_info['selected_chat_id']
    default_chat_id = user_info['default_chat_id']
    if default_chat_id == chat_id:
        print("清空历史记录")
        user_info["chats"][chat_id]['messages_history'] = user_info["chats"][chat_id]['messages_history'][:5]
    else:
        print("删除聊天对话")
        del user_info["chats"][chat_id]
    user_info['selected_chat_id'] = default_chat_id
    return "2"


def check_load_pickle():
    global all_user_dict

    if os.path.exists(USER_DICT_FILE):
        with open(USER_DICT_FILE, "rb") as pickle_file:
            all_user_dict = pickle.load(pickle_file)
            all_user_dict.change_capacity(USER_SAVE_MAX)
        print(f"已加载上次存储的用户上下文，共有{len(all_user_dict)}用户, 分别是")
        for i, user_id in enumerate(list(all_user_dict.keys())):
            print(f"{i} 用户id:{user_id}\t对话统计:\t", end="")
            user_info = all_user_dict.get(user_id)
            for chat_id in user_info['chats'].keys():
                print(f"{user_info['chats'][chat_id]['name']}[{len(user_info['chats'][chat_id]['messages_history'])}] ",
                      end="")
            print()
    elif os.path.exists("all_user_dict_v2.pkl"):  # 适配V2
        print('检测到v2版本的上下文，将转换为v3版本')
        with open("all_user_dict_v2.pkl", "rb") as pickle_file:
            all_user_dict = pickle.load(pickle_file)
            all_user_dict.change_capacity(USER_SAVE_MAX)
        print("共有用户", len(all_user_dict), "个")
        for user_id in list(all_user_dict.keys()):
            user_info: dict = all_user_dict.get(user_id)
            for chat_id in user_info['chats'].keys():
                if "messages_history" in user_info['chats'][chat_id]:
                    for i in range(len(user_info['chats'][chat_id]['messages_history'])):
                        # 将system关键字改为 web-system
                        if "role" in user_info['chats'][chat_id]['messages_history'][i] and \
                                user_info['chats'][chat_id]['messages_history'][i]["role"] == "system":
                            user_info['chats'][chat_id]['messages_history'][i]["role"] = "web-system"

        asyncio.run(save_all_user_dict())

    elif os.path.exists("all_user_dict.pkl"):  # 适配当出现这个时
        print('检测到v1版本的上下文，将转换为v3版本')
        with open("all_user_dict.pkl", "rb") as pickle_file:
            all_user_dict = pickle.load(pickle_file)
            all_user_dict.change_capacity(USER_SAVE_MAX)
        print("共有用户", len(all_user_dict), "个")
        for user_id in list(all_user_dict.keys()):
            user_info: dict = all_user_dict.get(user_id)
            if "messages_history" in user_info:
                user_dict = new_user_dict(user_id, "")
                chat_id = user_dict['selected_chat_id']
                user_dict['chats'][chat_id]['messages_history'] = user_info['messages_history']
                user_dict['chats'][chat_id]['chat_with_history'] = user_info['chat_with_history']
                user_dict['chats'][chat_id]['have_chat_context'] = user_info['have_chat_context']
                all_user_dict.put(user_id, user_dict)  # 更新
        asyncio.run(save_all_user_dict())
    else:
        with open(USER_DICT_FILE, "wb") as pickle_file:
            pickle.dump(all_user_dict, pickle_file)
        print("未检测到上次存储的用户上下文，已创建新的用户上下文")

    # 判断all_user_dict是否为None且时LRUCache的对象
    if all_user_dict is None or not isinstance(all_user_dict, LRUCache):
        print("all_user_dict为空或不是LRUCache对象，已创建新的LRUCache对象")
        all_user_dict = LRUCache(USER_SAVE_MAX)


if __name__ == '__main__' or __name__ == 'main':
    print("持久化存储文件路径为:", os.path.join(os.getcwd(), USER_DICT_FILE))
    all_user_dict = LRUCache(USER_SAVE_MAX)
    check_load_pickle()

    if os.getenv("DEPLOY_ON_ZEABUR") is None:
        app.run(host="0.0.0.0", port=PORT, debug=True)
