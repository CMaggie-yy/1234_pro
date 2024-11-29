import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import pandas as pd
import os
from langchain_milvus import Milvus
import convert_file_to_vector_db
from convert_file_to_vector_db import get_file_name_form_folder, split_text, create_my_db_milvus
import time

# 定义输出文件相关的全局变量
OUTPUT_DIR = 'data/output/embedding_eval/eval_before_split1000'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_FILE_NAME = 'bce_v1_250(1).xlsx'
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)
embedding_model_path = "/root/autodl-tmp/maidalun/bce-embedding-base_v1"
# 向量数据库持久化目录
persist_directory_folder = r'data_base/vector_db/milvus/split1000/bce_v1'
if not os.path.exists(persist_directory_folder):
    os.makedirs(persist_directory_folder)
persist_directory = os.path.join(persist_directory_folder,'bce_v1.db')


# 如果内存中没有对应的向量数据库，则生成
if not os.path.exists(persist_directory):
    json_folder_path = 'data/policy'
    file_path_list_all = get_file_name_form_folder(json_folder_path)
    print(len(file_path_list_all))
    split_texts = split_text(file_path_list_all)
    # print(split_texts[:5])
    
    create_my_db_milvus(split_texts, embedding_model_path, persist_directory) 
    # 等待一段时间，确保文件系统完成写入
    # time.sleep(10)  # 等待5秒，可以根据实际情况调整时间


# 读取测试数据的Excel文件
df = pd.read_excel(r"data/QA/qa_data_1250.xlsx")

# 筛选出dataset列值为eval的行，生成新的DataFrame
eval_df = df[df['dataset'] == 'eval']

# 从筛选后的DataFrame中提取["ID", 'Question', 'Title', 'Text']这几列，并转换为列表形式
data = eval_df[["ID", 'Question', 'Title', 'Text']].values.tolist()

# data = df[["ID", 'Question', 'Title', 'Text']].values.tolist()


# 加载嵌入模型
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
# 初始化向量数据库
vectordb = Milvus(
        connection_args={'uri': persist_directory},
        embedding_function=embeddings,
        search_params = {"HNSW": {"metric_type": "L2", "params": {"ef": 10}}}
    )
# 获取检索器，设置每次检索返回1个结果
retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# 判断结果文件是否已存在，若存在则读取，不存在则初始化一个空的DataFrame
if os.path.exists(OUTPUT_FILE_PATH):
    final_df = pd.read_excel(OUTPUT_FILE_PATH)
else:
    final_df = pd.DataFrame(columns=["ID", "TOP", "question", "recall_title", "recall_context", "title_question_form",
                                      "text_question_form", "mark", "mrr"])
count = 1
try:
    for ID, query, Title, Text in data:
        print(f"正在处理第{count}个问题...")
        first_y_appeared = False  # 用于标记是否已经出现过第一个"Y"
        res = retriever.get_relevant_documents(query)
        i = 1
        for doc in res:
            mark = ""
            if Title == doc.metadata["title"]:
                if not first_y_appeared:
                    mrr_value = 1 / i  # 第一个"Y"出现时，计算MRR值
                    first_y_appeared = True
                else:
                    mrr_value = 0  # 不是第一个"Y"，MRR值为0
                mark = "Y"
            else:
                mrr_value = 0  # 为"N"时，MRR值为0
                mark = "N"
            # 将当前查询的结果添加到final_df中
            temp_df = pd.DataFrame({
                "ID": [ID],
                "TOP": [i],
                "question": [query],
                "recall_title": [doc.metadata["title"]],
                "recall_context": [doc.page_content],
                "title_question_form": [Title],
                "text_question_form": [Text],
                "mark": [mark],
                "mrr": [mrr_value]
            })
            final_df = pd.concat([final_df, temp_df], ignore_index=True)
            i += 1
        count += 1
except Exception as e:
    print(f"处理过程中出现错误: {e}")
    
finally:
    if os.path.exists(OUTPUT_FILE_PATH):
        # 如果文件已存在，读取已有的文件，删除表头行（因为后续要追加数据，不能重复表头）
        existing_df = pd.read_excel(OUTPUT_FILE_PATH)
        final_df_without_header = final_df.iloc[1:]
        combined_df = pd.concat([existing_df, final_df_without_header], ignore_index=True)
        combined_df.to_excel(OUTPUT_FILE_PATH, index=False)
    else:
        # 文件不存在，直接写入并带上表头
        final_df.to_excel(OUTPUT_FILE_PATH, index=False)
    print("结果已经存入'" + OUTPUT_FILE_PATH + "'中")
    