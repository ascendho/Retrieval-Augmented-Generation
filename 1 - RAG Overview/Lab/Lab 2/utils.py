import requests
import json
import os
from typing import List, Dict
from together import Together

def get_proxy_url():
    """
    获取 API 请求的基础 URL。
    逻辑：
    1. 如果在 Coursera 实验环境中（检测环境变量），返回课程专用的代理地址。
    2. 否则，尝试从环境变量获取 Together 基础地址，默认指向 Together 官方 API。
    """
    if 'IN_COURSERA_ENVIRON' in os.environ:
        # 课程专用的代理转发地址
        return 'https://proxy.dlai.link/coursera_proxy/together'
    # 默认 Together 官方 API 端点
    return os.environ.get('TOGETHER_BASE_URL', 'https://api.together.xyz/')

def get_proxy_headers():
    """
    获取 API 调用所需的请求头。
    通常包含从环境变量读取的 Authorization（授权）信息。
    """
    return {"Authorization": os.environ.get("TOGETHER_API_KEY", "")}

def get_together_key():
    """
    从环境变量中提取 Together API Key。
    """
    return os.environ.get("TOGETHER_API_KEY", "")

def generate_with_single_input(prompt: str,
                               role: str = 'user',
                               top_p: float = None,
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str ="Qwen/Qwen3.5-9B",
                               together_api_key = None,
                              **kwargs):
    """
    【单次输入生成函数】
    将单个字符串 prompt 封装成对话格式并调用 LLM。
    """

    # 清理参数：Together API 不接受字符串形式的 'none'，如果为 None 则不传递
    payload_top_p = top_p if top_p is not None else None
    payload_temperature = temperature if temperature is not None else None

    # 构建请求负载 (Payload)
    payload = {
        "model": model,
        # 将单次输入转为模型要求的列表字典格式
        "messages": [{'role': role, 'content': prompt}],
        "max_tokens": max_tokens,
        # 禁用推理过程（针对支持思考链的模型），以节省 Token
        "reasoning": {"enabled": False},
        **kwargs
    }
    
    # 仅当参数非空时才加入负载，避免 API 报错
    if payload_temperature is not None:
        payload["temperature"] = payload_temperature
    if payload_top_p is not None:
        payload["top_p"] = payload_top_p

    # --- 核心逻辑：判断是走【代理模式】还是【SDK直连模式】 ---
    if (not together_api_key) and ('TOGETHER_API_KEY' not in os.environ):
        # 情况 A：没有 API Key，认定为 Coursera 实验环境，走代理
        url = os.path.join(get_proxy_url(), 'v1/chat/completions')
        # verify=False 是因为代理环境可能使用自签名证书
        response = requests.post(url, json = payload, verify=False)
        
        if not response.ok:
            raise Exception(f"调用 LLM 出错: {response.text}")
        try:
            json_dict = json.loads(response.text)
        except Exception as e:
            raise Exception(f"解析 LLM 响应失败。\n异常: {e}\n响应内容: {response.text}")
    else:
        # 情况 B：提供了 API Key，使用 Together 官方 SDK 直连
        if together_api_key is None:
            together_api_key = os.environ['TOGETHER_API_KEY']
        client = Together(api_key =  together_api_key)
        # 调用 SDK 并将结果转为字典格式
        json_dict = client.chat.completions.create(**payload).model_dump()
        # 统一格式化角色名称（转为小写字符串）
        json_dict['choices'][-1]['message']['role'] = json_dict['choices'][-1]['message']['role'].name.lower()

    # 提取模型生成的最终角色和内容
    try:
        output_dict = {
            'role': json_dict['choices'][-1]['message']['role'], 
            'content': json_dict['choices'][-1]['message']['content']
        }
    except Exception as e:
        raise Exception(f"提取输出字典失败。错误: {e}")
    
    return output_dict


def generate_with_multiple_input(messages: List[Dict],
                               top_p: float = None,
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str ="Qwen/Qwen3.5-9B",
                                together_api_key = None,
                                **kwargs):
    """
    【多轮对话生成函数】
    直接接收一个消息列表（包含对话历史），逻辑与单次输入基本一致。
    """
    
    # 参数预处理
    payload_top_p = top_p if top_p is not None else None
    payload_temperature = temperature if temperature is not None else None

    # 构建请求负载
    payload = {
        "model": model,
        "messages": messages, # 直接使用传入的消息列表
        "max_tokens": max_tokens,
        "reasoning": {"enabled": False},
        **kwargs
    }
    
    if payload_temperature is not None:
        payload["temperature"] = payload_temperature
    if payload_top_p is not None:
        payload["top_p"] = payload_top_p

    # --- 代理 vs SDK 判断逻辑（与单次输入函数相同） ---
    if (not together_api_key) and ('TOGETHER_API_KEY' not in os.environ):
        # 代理访问模式（requests 库）
        url = os.path.join(get_proxy_url(), 'v1/chat/completions')
        response = requests.post(url, json = payload, verify=False)
        if not response.ok:
            raise Exception(f"调用 LLM 出错: {response.text}")
        try:
            json_dict = json.loads(response.text)
        except Exception as e:
            raise Exception(f"解析 LLM 响应失败: {e}")
    else:
        # 官方 SDK 直连模式
        if together_api_key is None:
            together_api_key = os.environ['TOGETHER_API_KEY']
        client = Together(api_key =  together_api_key)
        json_dict = client.chat.completions.create(**payload).model_dump()
        # 角色格式规范化
        json_dict['choices'][-1]['message']['role'] = json_dict['choices'][-1]['message']['role'].name.lower()

    # 封装统一的返回格式
    try:
        output_dict = {
            'role': json_dict['choices'][-1]['message']['role'], 
            'content': json_dict['choices'][-1]['message']['content']
        }
    except Exception as e:
        raise Exception(f"获取输出失败。错误: {e}")
    
    return output_dict