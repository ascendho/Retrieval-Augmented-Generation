"""
作业：RAG 系统入门

欢迎来到 RAG 课程的第一个作业！在本作业中，你将使用一个包含 BBC 新闻的数据集来构建一个 RAG（检索增强生成）管道。
目标是让大语言模型（LLM）能够从数据集中检索相关的详细新闻，并利用这些信息生成更准确的回答。
"""

# ============================================================================
# 1.2 导入必要的库
# ============================================================================
from utils import (
    retrieve,                    # 检索函数：根据查询寻找相关文档的索引
    pprint,                      # 格式化打印函数
    generate_with_single_input,  # LLM 调用函数
    read_dataframe,              # 读取数据集的函数
    display_widget               # 显示交互界面的函数
)
import unittests                 # 导入单元测试模块以验证你的代码


# ============================================================================
# 2 - 加载数据集
# ============================================================================
# 从 Kaggle 加载 BBC 新闻 2024 数据集
NEWS_DATA = read_dataframe("news_data_dedup.csv")

# 查看数据结构（打印第 10 到 11 条数据）
pprint(NEWS_DATA[9:11])


# ============================================================================
# 3.1 根据索引查询新闻函数
# ============================================================================
def query_news(indices):
    """
    根据指定的索引列表从数据集中检索元素。

    参数:
    indices (list of int): 包含所需元素在数据集中索引位置的列表。
    
    返回:
    list: 数据集中对应索引位置的元素列表。
    """
     
    # 使用列表推导式提取对应索引的新闻条目
    output = [NEWS_DATA[index] for index in indices]

    return output


# 测试：获取索引为 3, 6, 9 的新闻
indices = [3, 6, 9]
pprint(query_news(indices))


# ============================================================================
# 3.2 检索函数测试
# ============================================================================
# 测试检索功能：寻找关于“北美演唱会”的新闻，返回最相关的 1 条索引
indices = retrieve("Concerts in North America", top_k=1)
print(indices)

# 查询对应的具体新闻内容
retrieved_documents = query_news(indices)
pprint(retrieved_documents)


# ============================================================================
# 3.3 获取相关数据（练习 1）
# ============================================================================
def get_relevant_data(query: str, top_k: int = 5) -> list[dict]:
    """
    根据给定的查询语句检索并返回前 k 个最相关的数据项。

    该函数执行以下步骤：
    1. 根据查询语句从数据集中检索前 'k' 个相关项的索引。
    2. 根据这些索引从数据集中获取相应的数据内容。

    参数:
    - query (str): 用于寻找相关条目的搜索查询字符串。
    - top_k (int): 要检索的相关条目数量。默认为 5。

    返回:
    - list[dict]: 包含前 k 个相关项数据内容的字典列表。
    """
    # 第一步：检索与查询最相关的 top_k 个索引
    relevant_indices = retrieve(query=query, top_k=top_k)

    # 第二步：利用上一步获得的索引获取实际的新闻数据
    relevant_data = query_news(relevant_indices)

    return relevant_data


# 测试 get_relevant_data 函数
query = "Greatest storms in the US"
relevant_data = get_relevant_data(query, top_k=1)
pprint(relevant_data)

# 运行单元测试验证结果
unittests.test_get_relevant_data(get_relevant_data)


# ============================================================================
# 3.4 格式化相关数据（练习 2）
# ============================================================================
def format_relevant_data(relevant_data):
    """
    将相关文档列表格式化为结构化的字符串，以便在 RAG 系统中使用。

    参数:
    relevant_data (list): 包含相关数据的列表。

    返回:
    str: 格式化后的字符串，将多个文档整合在一起，为 RAG 提供上下文。
    """

    # 创建一个列表来存储格式化后的文档字符串
    formatted_documents = []
    
    # 遍历每一个相关文档
    for document in relevant_data:

        # 将每个文档格式化为特定的结构布局。
        # 注意每个文档应占一行，因此在末尾添加换行符。
        formatted_document = f"Title: {document['title']}, Description: {document['description']}, Published at: {document['published_at']}\nURL: {document['url']}"
        
        # 将格式化后的单个文档添加到列表中
        formatted_documents.append(formatted_document)
    
    # 将所有文档用换行符连接成一个最终的增强提示词背景字符串
    return "\n".join(formatted_documents)


# 使用示例数据进行测试
example_data = NEWS_DATA[4:8]
print(format_relevant_data(example_data))

# 运行单元测试验证结果
unittests.test_format_relevant_data(format_relevant_data)


# ============================================================================
# 3.5 生成最终提示词
# ============================================================================
def generate_final_prompt(query, top_k=5, use_rag=True, prompt=None):
    """
    根据用户查询生成最终提示词，可选择性地通过 RAG 整合相关数据。

    参数:
        query (str): 用户提出的问题查询。
        top_k (int): 需要检索并整合的相关数据条数。默认为 5。
        use_rag (bool): 是否使用检索增强生成（RAG）。如果为 True，则在提示词中包含相关数据。
        prompt (str): 可选的提示词模板，可包含 {query} 和 {documents} 占位符。

    返回:
        str: 生成的最终提示词。
    """
    # 如果不使用 RAG，则直接返回原始查询内容
    if not use_rag:
        return query

    # 获取与查询相关的 top_k 条数据
    relevant_data = get_relevant_data(query, top_k=top_k)

    # 将检索到的相关数据格式化为字符串
    retrieve_data_formatted = format_relevant_data(relevant_data)

    # 如果没有提供自定义模板，则使用默认的提示词模板
    if prompt is None:
        prompt = (
            f"请回答下方的用户提问。我们会为你提供额外的参考信息来辅助你构思答案。"
            f"这些相关信息源自 2024 年，你应该将其作为背景知识来回答问题。"
            f"不要仅仅依赖这些信息，而应将其融入你的整体知识储备中。\n"
            f"查询语句: {query}\n"
            f"2024 年新闻背景: {retrieve_data_formatted}"
        )
    else:
        # 如果提供了自定义模板，则将查询和文档填充到对应的占位符中
        prompt = prompt.format(query=query, documents=retrieve_data_formatted)

    return prompt


# 测试最终提示词生成效果
print(generate_final_prompt("Tell me about the US GDP in the past 3 years."))


# ============================================================================
# 3.6 LLM 调用封装
# ============================================================================
def llm_call(query, top_k=5, use_rag=True, prompt=None):
    """
    调用大语言模型生成回答，可选择是否使用 RAG。

    参数:
        query (str): 交给模型处理的用户查询。
        top_k (int): 检索的相关数据条数。默认为 5。
        use_rag (bool): 是否使用检索增强生成。默认为 True。
        prompt (str): 自定义提示词模板。默认为 None。

    返回:
        str: 模型生成的回答内容。
    """
    
    # 第一步：获取包含查询和相关文档的最终提示词
    final_prompt = generate_final_prompt(query, top_k, use_rag, prompt)

    # 第二步：调用 LLM 获取响应
    generated_response = generate_with_single_input(final_prompt)

    # 第三步：提取响应中的文本内容
    generated_message = generated_response['content']
    
    return generated_message


# ============================================================================
# 测试 LLM 调用效果
# ============================================================================
query = "Tell me about the US GDP in the past 3 years."

# 测试开启 RAG 模式下的回答
print("使用 RAG 的回答：")
print(llm_call(query, use_rag=True))

# 测试关闭 RAG 模式下的回答（模型仅靠内置知识回答）
print("\n不使用 RAG 的回答：")
print(llm_call(query, use_rag=False))


# ============================================================================
# 4 - 实验与自定义测试
# ============================================================================
# 你可以尝试输入不同的问题来观察 RAG 系统的表现！
# 例如：
# * 过去一年最重要的事件有哪些？
# * 2024 年全球变暖的进展如何？
# * 告诉我 AI 领域的最新进展。
#
# 你也可以自定义 prompt 模板，使用 {query} 和 {documents} 标记数据插入的位置。

# 交互式小部件测试（如果在交互式环境运行，可取消下方注释）
# display_widget(llm_call)

print("恭喜！你完成了第一个简单的 RAG 系统！继续加油！")