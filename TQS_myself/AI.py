import sys
import requests
import os
import json

# 标准行缓冲输出
sys.stdout.reconfigure(line_buffering=True)
print("脚本启动成功")

# ===================== 配置区 =====================
# 方案A：原生直连DogAPI（你现在的方式）
API_KEY = "sk-eBUf7hVTemgB6MaXotjNvAeiy0LyplQvK7mz8KmBcCbwDF9J"
DOGAPI_URL = "https://www.dogapi.cc/v1/chat/completions"

# 全局会话上下文（实现连续对话核心）
conversation_history = [
    {"role": "system", "content": "你是量子计算、AI for Science科研专家，分析代码客观评估开发者水平，输出分条理清晰"}
]

# 普通网络代理（Clash，仅用于访问外网接口）
proxy = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890"
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
# ==================================================

def ask_ai_continue(user_input):
    # 追加用户提问到历史
    conversation_history.append({"role": "user", "content": user_input})
    payload = {
        "model": "claude-opus-4-8-thinking",
        "messages": conversation_history,  # 带上全部历史实现连续对话
        "temperature": 0.7
    }
    try:
        resp = requests.post(
            DOGAPI_URL,
            headers=headers,
            json=payload,
            proxies=proxy,
            timeout=120
        )
        print("\nHTTP状态码：", resp.status_code)
        if resp.status_code != 200:
            print("接口错误返回：", resp.text)
            return None

        data = resp.json()
        if "error" in data:
            print("接口报错：", data["error"])
            return None

        if "choices" not in data or len(data["choices"]) == 0:
            print("无有效回复")
            return None

        msg = data["choices"][0]["message"]
        think = msg.get("reasoning_content", "无独立思考内容")
        content = msg["content"]

        # 保存AI回答进上下文，下一轮对话能读取
        conversation_history.append({"role": "assistant", "content": content})

        print("===思考===")
        print(think)
        print("===回答===")
        print(content)
        return content

    except requests.exceptions.ConnectionError:
        print("网络错误：无法连接代理/接口，请检查Clash是否开启、端口是否7890")
    except requests.exceptions.Timeout:
        print("网络错误：请求超时")
    except Exception as err:
        print("未知请求错误：", err)
    return None

def read_project_info(root_path, max_file_size=20480):
    """读取项目目录+全部py源码，逻辑完全保留你的原版"""
    tree = []
    for root, dirs, files in os.walk(root_path):
        rel_path = root.replace(root_path, '').strip(os.sep)
        level = rel_path.count(os.sep) if rel_path else 0
        indent = '  ' * level
        tree.append(f"{indent}{os.path.basename(root)}/")
        subindent = '  ' * (level + 1)
        for f in files:
            tree.append(f"{subindent}{f}")
    dir_tree = "\n".join(tree)

    code_content = ""
    for root, _, files in os.walk(root_path):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            file_size = os.path.getsize(fpath)
            if file_size > max_file_size:
                code_content += f"\n====={fpath}=====\n[文件过大，跳过读取，大小：{file_size//1024}KB]\n"
                continue
            try:
                with open(fpath, "r", encoding="utf-8") as fp:
                    code_content += f"\n====={fpath}=====\n{fp.read()}\n"
            except UnicodeDecodeError:
                with open(fpath, "r", encoding="gbk") as fp:
                    code_content += f"\n====={fpath}=====\n{fp.read()}\n"
            except Exception as e:
                code_content += f"\n====={fpath}=====\n[读取失败：{str(e)}]\n"

    output = f"【项目目录结构】\n{dir_tree}\n\n【全部Python源码】\n{code_content}"
    if len(output) > 120000:
        output = output[:120000] + "\n\n...内容过长，已截断..."
    return output

# 持久化对话（可选，重启脚本不丢上下文）
def save_chat(file="chat_log.json"):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=2)
def load_chat(file="chat_log.json"):
    global conversation_history
    try:
        with open(file, "r", encoding="utf-8") as f:
            conversation_history = json.load(f)
        print("✅ 已加载历史对话上下文")
    except FileNotFoundError:
        print("无历史对话，新建会话")

if __name__ == "__main__":
    load_chat()
    project_root = r"E:\保研\中国科学技术大学\TQS\TQS_myself"
    print(f"正在读取项目目录：{project_root} ...")
    project_text = read_project_info(project_root)
    print(f"项目文本读取完成，总长度：{len(project_text)} 字符")
    # 第一轮自动发送项目分析请求
    first_prompt = f"请深度分析下面这个TQS_myself量子相关功能包，说明这个人的水平，与个人能力，在使用AI，只有一天的条件下：\n{project_text}"
    ask_ai_continue(first_prompt)

    # 循环多轮对话，持续追问
    while True:
        print("\n--------------------------------------")
        user_input = input("你继续提问(输入exit退出，save保存对话)：")
        cmd = user_input.strip().lower()
        if cmd == "exit":
            save_chat()
            print("已保存对话，程序退出")
            break
        if cmd == "save":
            save_chat()
            print("对话已保存")
            continue
        ask_ai_continue(user_input)