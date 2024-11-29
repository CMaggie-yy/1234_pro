import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import pandas as pd
import os

# 向量数据库持久化目录
persist_directory = 'data_base/vector_db/chroma/bge_large'

# 读取测试数据的Excel文件
df = pd.read_excel(r"data/QA_test.xlsx")
data = df[["ID", 'Question', 'Title', 'Text']].values.tolist()

# 加载嵌入模型
embeddings = HuggingFaceEmbeddings(model_name="/root/autodl-tmp/BAAI/bge-large-zh-v1___5")
# 初始化向量数据库
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# 获取检索器，设置每次检索返回1个结果
retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# 判断结果文件是否已存在，若存在则读取，不存在则初始化一个空的DataFrame
if os.path.exists('data/top10_chroma_recall_result.xlsx'):
    final_df = pd.read_excel('data/top10_chroma_recall_result.xlsx')
else:
    final_df = pd.DataFrame(columns=["ID","TOP", "question", "recall_title", "recall_context", "title_question_form",
                                      "text_question_form", "mark"])
count = 1
try:
    for ID, query, Title, Text in data:
        print(f"正在处理第{count}个问题...")
        res = retriever.get_relevant_documents(query)
        i = 1
        
        for doc in res:
            mark = ""
            if Title == doc.metadata["title"]:
                mark = "Y"
            else:
                mark = "N"
            # 创建一个临时DataFrame来存储当前查询的结果
            temp_df = pd.DataFrame({
                "ID": [ID],
                "TOP": [i],
                "question": [query],
                "recall_title": [doc.metadata["title"]],
                "recall_context": [doc.page_content],
                "title_question_form": [Title],
                "text_question_form": [Text],
                "mark" : [mark]
            })
            # 将临时DataFrame添加到最终结果DataFrame中
            final_df = pd.concat([final_df, temp_df], ignore_index=True)
            i += 1
        count += 1
except Exception as e:
    print(f"处理过程中出现错误: {e}")
finally:
    if os.path.exists('data/top10_chroma_recall_result.xlsx'):
        # 如果文件已存在，读取已有的文件，删除表头行（因为后续要追加数据，不能重复表头）
        existing_df = pd.read_excel('data/top10_chroma_recall_result.xlsx')
        final_df_without_header = final_df.iloc[1:]
        combined_df = pd.concat([existing_df, final_df_without_header], ignore_index=True)
        combined_df.to_excel('data/top10_chroma_recall_result.xlsx', index=False)
    else:
        # 文件不存在，直接写入并带上表头
        final_df.to_excel('data/top10_chroma_recall_result.xlsx', index=False)
    print("结果已经存入'data/top10_chroma_recall_result.xlsx'中")