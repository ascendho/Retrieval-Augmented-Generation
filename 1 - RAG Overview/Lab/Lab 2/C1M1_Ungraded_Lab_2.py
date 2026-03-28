# 从课程提供的工具包 utils 中导入封装好的函数
# 这些函数隐藏了复杂的 API 请求细节，方便初学者快速上手
from utils import (
    generate_with_single_input,    # 用于发送单条提示词（一问一答）
    generate_with_multiple_input,  # 用于发送对话历史（多轮对话）
    get_proxy_url,                 # 获取课程专用的代理服务器地址
    get_proxy_headers,             # 获取代理所需的认证头信息
    get_together_key,              # 获取 Together.ai 的 API 密钥
)

# 导入标准的 OpenAI 库和 HTTP 客户端
# 注意：Together.ai 的接口与 OpenAI 兼容，因此可以使用 OpenAI 的 SDK
from openai import OpenAI, DefaultHttpxClient
import httpx

# ==========================================================
# 1. 基础 LLM 调用示例
# ==========================================================

# 【示例 1：单轮对话】
# 就像在网页端直接输入一个问题。模型不记得你之前说过什么。
output = generate_with_single_input(
    prompt="What is the capital of France?"
)

print("--- 单轮对话结果 ---")
print("Role:", output["role"])       # 通常是 'assistant'
print("Content:", output["content"]) # 模型的回答内容

# 【示例 2：多轮对话】
# 通过传递一个列表来模拟“对话记忆”。
# 列表中包含了 user（用户）和 assistant（助手）交替说话的历史。
messages = [
    {"role": "user", "content": "Hello, who won the FIFA world cup in 2018?"},
    {"role": "assistant", "content": "France won the 2018 FIFA World Cup."},
    {"role": "user", "content": "Who was the captain?"}, # 这里的“他”指代上文的法国队
]

output = generate_with_multiple_input(
    messages=messages,
    max_tokens=100,
)

print("\n--- 多轮对话结果 ---")
print("Role:", output["role"])
print("Content:", output["content"]) # 模型会根据上下文回答：雨果·洛里斯（Hugo Lloris）


# ==========================================================
# 2. 使用标准 OpenAI SDK 进行高级配置
# ==========================================================

# 设置基础 URL。在课程环境中走代理，在本地环境通常填 Together.ai 的官方地址
base_url = get_proxy_url()

# 这一步是为了解决特定教学环境下的网络限制：
# 创建一个自定义的 HTTP 传输层，禁用 SSL 证书验证（仅限本实验安全环境使用）
transport = httpx.HTTPTransport(local_address="0.0.0.0", verify=False)

# 创建一个支持自定义 Header 和 SSL 绕过的 HTTP 客户端
http_client = DefaultHttpxClient(transport=transport, headers=get_proxy_headers())

# 初始化 OpenAI 客户端对象
client = OpenAI(
    api_key=get_together_key(),  # 身份验证令牌
    base_url=base_url,           # API 地址
    http_client=http_client,     # 注入上面配置好的 HTTP 客户端
)

# 使用 OpenAI 标准语法调用聊天接口
response = client.chat.completions.create(
    messages=messages,
    model="Qwen/Qwen3.5-9B",      # 指定使用的模型名称
    extra_body={"reasoning": False}, # 禁用某些模型的推理链输出，以节省 token
)

print("\n--- SDK 调用原生响应对象 ---")
print(response) # 打印完整的响应 JSON
print("提取的回复内容:", response.choices[0].message.content)


# ==========================================================
# 3. 提示词增强 (Prompt Augmentation) - 模拟 RAG 流程
# ==========================================================

# 模拟一个外部数据库：这里存储了两套房子的详细 JSON 数据
# LLM 本身并不知道这些私有数据，我们需要把它们“喂”给提示词
house_data = [
    {
        "address": "123 Maple Street",
        "city": "Springfield",
        "state": "IL",
        "zip": "62701",
        "bedrooms": 3,
        "bathrooms": 2,
        "square_feet": 1500,
        "price": 230000,
        "year_built": 1998,
    },
    {
        "address": "456 Elm Avenue",
        "city": "Shelbyville",
        "state": "TN",
        "zip": "37160",
        "bedrooms": 4,
        "bathrooms": 3,
        "square_feet": 2500,
        "price": 320000,
        "year_built": 2005,
    },
]

# 【函数：数据转文字】
# 将结构化的 JSON 字典列表转换成一段自然语言文本。
# 大模型处理自然语言描述的效果通常比直接读原始 JSON 更好。
def house_info_layout(houses):
    layout = ""
    for house in houses:
        # 使用 f-string 拼接每一项房屋数据
        layout += (
            f"House located at {house['address']}, {house['city']}, {house['state']} {house['zip']} with "
            f"{house['bedrooms']} bedrooms, {house['bathrooms']} bathrooms, "
            f"{house['square_feet']} sq ft area, priced at ${house['price']}, "
            f"built in {house['year_built']}.\n" 
        )
    return layout

# 【函数：构建增强提示词】
# 这个函数实现了 RAG 的核心：将指令 + 背景知识 + 用户问题 组合在一起
def generate_prompt(query, houses):
    # 1. 先把房屋数据转成一段话
    houses_layout = house_info_layout(houses)
    
    # 2. 构造一个包含上下文的复杂提示词模板
    # 三引号允许我们在字符串中保留换行，不需要写很多 \n
    prompt_text = f"""
Use the following houses information to answer users queries.
{houses_layout}
Query: {query}
             """
    return prompt_text

# ==========================================================
# 4. 最终对比测试：增强前 vs 增强后
# ==========================================================

query = "What is the most expensive house? And the bigger one?"

# --- 情况 A：不带任何房屋信息的提问 ---
# LLM 只能靠它训练时的通用知识回答，或者告诉你自己没有这些数据
query_without_house_info = generate_with_single_input(prompt=query, role="user")

# --- 情况 B：带增强提示词的提问 ---
# 1. 生成包含数据的提示词
enhanced_query = generate_prompt(query, houses=house_data)
# 2. 将这段包含了“标准答案背景”的文本发给 LLM
# 这里将 role 设为 assistant 是为了在特定提示词结构下模拟更稳健的响应逻辑
query_with_house_info = generate_with_single_input(prompt=enhanced_query, role="assistant")

print("\n--- 实验结果对比 ---")
print("【不带数据的回答】（可能会胡编或拒绝回答）：")
print(query_without_house_info["content"])

print("\n【增强数据后的回答】（能够准确从提供的资料中提取答案）：")
print(query_with_house_info["content"])