import json
import numpy as np
import pandas as pd
from pprint import pprint as original_pprint
from dateutil import parser
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os 
from together import Together

# ============================================================================
# 1. 模型与数据初始化
# ============================================================================

# 设置 Embedding 模型的本地路径（从环境变量获取）
model_name = os.path.join(os.environ['MODEL_PATH'], "BAAI/bge-base-en-v1.5")

# 加载 SentenceTransformer 模型用于将文本转换为向量（Embedding）
# cache_folder 指定了模型权重存放的目录
model = SentenceTransformer("BAAI/bge-base-en-v1.5", cache_folder = os.environ['MODEL_PATH'])

# 加载预先计算好的向量数据库（通常是之前对新闻数据集所有条目计算出的向量）
EMBEDDINGS = joblib.load("embeddings.joblib")

# ============================================================================
# 2. 代理与 API 配置函数
# ============================================================================

def get_proxy_url():
    """
    获取 API 代理 URL。
    如果在 Coursera 环境中，使用课程专用代理；否则尝试获取 Together.ai 官方地址。
    """
    if 'IN_COURSERA_ENVIRON' in os.environ:
        return 'https://proxy.dlai.link/coursera_proxy/together'
    return os.environ.get('TOGETHER_BASE_URL', 'https://api.together.xyz/')

def get_proxy_headers():
    """
    获取 API 请求头，主要包含 Authorization 认证信息。
    """
    return {"Authorization": os.environ.get("TOGETHER_API_KEY", "")}

def get_together_key():
    """
    从环境变量获取 Together.ai 的 API Key。
    """
    return os.environ.get("TOGETHER_API_KEY", "")

# ============================================================================
# 3. 数据处理与格式化工具
# ============================================================================

def pprint(*args, **kwargs):
    """
    自定义打印函数：将 Python 对象以美化的 JSON 格式输出。
    """
    print(json.dumps(*args, indent = 2))

def format_date(date_string):
    """
    标准化日期格式：将各种格式的日期字符串转换为 YYYY-MM-DD。
    """
    # 将输入字符串解析为 datetime 对象
    date_object = parser.parse(date_string)
    # 重新格式化为字符串
    formatted_date = date_object.strftime("%Y-%m-%d")
    return formatted_date

def read_dataframe(path):
    """
    读取 CSV 文件并转换为字典列表。
    同时对发布日期（published_at）和更新日期（updated_at）进行格式化。
    """
    df = pd.read_csv(path)

    # 对指定日期列应用格式化函数
    df['published_at'] = df['published_at'].apply(format_date)
    df['updated_at'] = df['updated_at'].apply(format_date)

    # 将 DataFrame 转换为字典记录格式（List of Dicts）
    df = df.to_dict(orient='records')
    return df

# ============================================================================
# 4. LLM 调用核心函数
# ============================================================================

def generate_with_single_input(prompt: str,
                               role: str = 'user',
                               top_p: float = None,
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str ="Qwen/Qwen3.5-9B",
                               together_api_key = None,
                              **kwargs):
    """
    调用大语言模型生成回复。支持通过代理或直接使用 Together SDK。
    """

    # 处理可选参数，避免向 API 发送 None 值
    payload_top_p = top_p if top_p is not None else None
    payload_temperature = temperature if temperature is not None else None

    # 构建基础请求载荷
    payload = {
        "model": model,
        "messages": [{'role': role, 'content': prompt}],
        "max_tokens": max_tokens,
        "reasoning": {"enabled": False}, # 禁用某些模型自带的思维链输出
        **kwargs
    }
    
    # 仅当参数有值时才添加到负载中
    if payload_temperature is not None:
        payload["temperature"] = payload_temperature
    if payload_top_p is not None:
        payload["top_p"] = payload_top_p

    # 判断调用模式：代理模式（Coursera 环境）还是直连模式（本地环境）
    if (not together_api_key) and ('TOGETHER_API_KEY' not in os.environ):
        # 使用 requests 发送 POST 请求到代理服务器
        url = os.path.join(get_proxy_url(), 'v1/chat/completions')
        response = requests.post(url, json = payload, verify=False) # 代理环境下通常跳过 SSL 验证
        if not response.ok:
            raise Exception(f"调用 LLM 报错: {response.text}")
        try:
            json_dict = json.loads(response.text)
        except Exception as e:
            raise Exception(f"解析相应 JSON 失败。异常: {e}\n响应内容: {response.text}")
    else:
        # 使用 Together 官方 SDK 直连
        if together_api_key is None:
            together_api_key = os.environ['TOGETHER_API_KEY']
        client = Together(api_key =  together_api_key)
        json_dict = client.chat.completions.create(**payload).model_dump()
        # 将枚举类型的 Role 转换为小写字符串以保持一致性
        json_dict['choices'][-1]['message']['role'] = json_dict['choices'][-1]['message']['role'].name.lower()

    # 封装返回结果
    try:
        output_dict = {
            'role': json_dict['choices'][-1]['message']['role'], 
            'content': json_dict['choices'][-1]['message']['content']
        }
    except Exception as e:
        raise Exception(f"获取输出字典失败。错误: {e}")
    return output_dict

# ============================================================================
# 5. RAG 检索相关逻辑
# ============================================================================

def concatenate_fields(dataset, fields):
    """
    将数据集中的多个字段合并为一个长字符串，用于生成 Embedding 或展示。
    """
    concatenated_data = [] 

    for data in dataset:
        text = "" 
        for field in fields: 
            # 如果字段不存在，使用空字符串
            context = data.get(field, '') 

            if context:
                # 拼接字段，并在中间加空格
                text += f"{context} " 

        # 去除前后空格，并截断长度（防止单条数据过长超过模型输入限制）
        text = text.strip()[:493]
        concatenated_data.append(text) 
    
    return concatenated_data

# 读取新闻数据集并转换为字典格式
NEWS_DATA = pd.read_csv("./news_data_dedup.csv").to_dict(orient = 'records')

def retrieve(query, top_k = 5):
    """
    检索逻辑：计算查询语句与数据库中向量的相似度，返回最相关的 K 个索引。
    """
    # 1. 将查询文本转换为向量
    query_embedding = model.encode(query)

    # 2. 计算查询向量与预存向量库之间的余弦相似度
    # reshape(1, -1) 将单向量转为 2D 数组以符合 API 要求
    similarity_scores = cosine_similarity(query_embedding.reshape(1,-1), EMBEDDINGS)[0]
    
    # 3. 对相似度分数进行降序排序，获取索引
    similarity_indices = np.argsort(-similarity_scores)

    # 4. 返回前 K 个最相似结果的索引
    top_k_indices = similarity_indices[:top_k]

    return top_k_indices

# ============================================================================
# 6. 交互式界面（ipywidgets）
# ============================================================================

import ipywidgets as widgets
from IPython.display import display, Markdown

def display_widget(llm_call_func):
    """
    创建并在 Jupyter 中显示一个交互式 UI，用于对比开启和关闭 RAG 的效果。
    """
    def on_button_click(b):
        # 清空之前的输出和状态
        output1.clear_output()
        output2.clear_output()
        status_output.clear_output()
        
        # 显示“生成中...”提示
        status_output.append_stdout("生成响应中，请稍候...\n")
        
        query = query_input.value
        top_k = slider.value
        # 处理自定义提示词布局
        prompt = prompt_input.value.strip() if prompt_input.value.strip() else None
        
        # 分别调用开启 RAG 和关闭 RAG 的逻辑（llm_call_func 是在作业主程序中定义的）
        response1 = llm_call_func(query, use_rag=True, top_k=top_k, prompt=prompt)
        response2 = llm_call_func(query, use_rag=False, top_k=top_k, prompt=prompt)
        
        # 在左右两个框中显示结果
        with output1:
            display(Markdown(response1))
        with output2:
            display(Markdown(response2))
        
        # 清除“生成中...”提示
        status_output.clear_output()

    # --- UI 组件定义 ---
    # 查询输入框
    query_input = widgets.Text(
        description='查询 (Query):',
        placeholder='在此输入你的问题',
        layout=widgets.Layout(width='100%')
    )

    # 增强提示词模板输入框
    prompt_input = widgets.Textarea(
        description='提示词布局:',
        placeholder=("在此输入提示词模板，必须包含 {query} 和 {documents} 占位符。"
                     "\n留空则使用系统默认模板。示例：\n问题：{query}\n参考资料：{documents}"),
        layout=widgets.Layout(width='100%', height='100px'),
        style={'description_width': 'initial'}
    )

    # Top K 滑动条
    slider = widgets.IntSlider(
        value=5, 
        min=1,
        max=20,
        step=1,
        description='检索条数 (Top K):',
        style={'description_width': 'initial'}
    )

    # 输出区域及状态显示
    output1 = widgets.Output(layout={'border': '1px solid #ccc', 'width': '45%'})
    output2 = widgets.Output(layout={'border': '1px solid #ccc', 'width': '45%'})
    status_output = widgets.Output()

    # 提交按钮
    submit_button = widgets.Button(
        description="获取模型回答",
        style={'button_color': '#f0f0f0', 'font_color': 'black'}
    )
    submit_button.on_click(on_button_click)

    # 左右对比标签
    label1 = widgets.Label(value="开启 RAG (基于检索数据)", layout={'width': '45%', 'text_align': 'center'})
    label2 = widgets.Label(value="未开启 RAG (仅凭模型知识)", layout={'width': '45%', 'text_align': 'center'})

    # --- 自定义 CSS 样式 ---
    display(widgets.HTML("""
    <style>
        .custom-output {
            background-color: #f9f9f9;
            color: black;
            border-radius: 5px;
        }
        .widget-textarea, .widget-button {
            background-color: #f0f0f0 !important;
            color: black !important;
            border: 1px solid #ccc !important;
        }
        textarea {
            background-color: #fff !important;
            color: black !important;
            border: 1px solid #ccc !important;
        }
    </style>
    """))

    # --- 布局与渲染 ---
    display(query_input, prompt_input, slider, submit_button, status_output)
    hbox_labels = widgets.HBox([label1, label2], layout={'justify_content': 'space-between'})
    hbox_outputs = widgets.HBox([output1, output2], layout={'justify_content': 'space-between'})

    def style_outputs(*outputs):
        for output in outputs:
            output.layout.margin = '5px'
            output.layout.height = '300px'
            output.layout.padding = '10px'
            output.layout.overflow = 'auto'
            output.add_class("custom-output")

    style_outputs(output1, output2)
    
    # 最后显示标签和输出区域
    display(hbox_labels)
    display(hbox_outputs)