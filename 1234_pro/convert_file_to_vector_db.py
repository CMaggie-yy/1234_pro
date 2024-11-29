import json
import os
import re
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from pymilvus import MilvusClient


# 从文件夹读取文件名称列表，获取所有的文件完整路径
def get_file_name_form_folder(json_folder_path: str) -> List[str]:
    file_path_list_all = []
    for file_name in os.listdir(json_folder_path):
        file_path = os.path.join(json_folder_path, file_name)
        file_path_list_all.append(file_path)
    return file_path_list_all


# 读取和解析 JSON 文件
def parse_file_to_document(file_path_list_all: List[str]) -> List[Document]:
    documents = []
    for file_path in file_path_list_all:
        # 确保路径是文件而不是目录
        if os.path.isfile(file_path):
            document = Document(page_content="", metadata={})
            filename, extension = os.path.splitext(file_path)
            extension = extension.lstrip(".")

            with open(file_path, "r", encoding='utf-8', errors="ignore") as f:
                if extension == "json":
                    data = json.load(f)
                else:
                    data = [json.loads(line) for line in f if line.strip()]

                title = data.get("title", "").strip()
                time = data.get('time', "")
                infosource = data.get('infosource', "")
                metadata = {
                    "title": title,
                    "time": time[0] if time else "",
                    "infosource": infosource
                }

                context = data.get("context", '')
                context_text = "\n".join(context)
                context_text = re.sub(r'\n+', '', context_text)

                document.page_content = context_text
                document.metadata = metadata
                documents.append(document)
        else:
            print(f"Skipping directory: {file_path}")
    return documents


# 将metadata的内容同步放在page_content中
def parse_file_to_document_01(file_path_list_all: List[str]) -> List[Document]:
    documents = []
    for file_path in file_path_list_all:
        # 确保路径是文件而不是目录
        if os.path.isfile(file_path):
            document = Document(page_content="", metadata={})
            filename, extension = os.path.splitext(file_path)
            extension = extension.lstrip(".")

            with open(file_path, "r", encoding='utf-8', errors="ignore") as f:
                if extension == "json":
                    data = json.load(f)
                else:
                    data = [json.loads(line) for line in f if line.strip()]

                title = data.get("title", "").strip()
                time = data.get('time', "")
                infosource = data.get('infosource', "")
                metadata = {
                    "title": title,
                    "time": time[0] if time else "",
                    "infosource": infosource
                }

                context = data.get("context", '')
                context_text = "\n".join(context)
                context_text = re.sub(r'\n+', '', context_text)

                # 将metadata内容拼接成字符串
                metadata_str = ";".join([f"{k}:{v}" for k, v in metadata.items()])
                # 用分号连接文本内容和metadata字符串
                document.page_content = f"{metadata_str};{context_text}"
                document.metadata = metadata
                documents.append(document)
        else:
            print(f"Skipping directory: {file_path}")
    return documents


# 文本分割
def split_text(file_path_list_all):
    docs_list = parse_file_to_document_01(file_path_list_all)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["。"],
        keep_separator="end"
    )
    # chunk_size每多少个文本切分一次；chunk_overlap重叠部分是多少个字符
    splits = text_splitter.split_documents(docs_list)

    return splits
    # 下面将切分结果进行展示：splits 是一个列表，其中每个元素也是一个列表，表示一个文档的分割结果
    # for doc_index, doc_splits in enumerate(splits[0:5]):
    #     print(f"Document {doc_index + 1}:")  # 显示文档编号
    #     for split_index, split_text in enumerate(doc_splits):
    #         print(f"  Split {split_index + 1}: {split_text[:50]}...")  # 打印每个分段的前50个字符
    #     print("\n" + "-" * 60 + "\n")  # 在每个文档之间加入分隔线，增加可读性

def save_splits_to_json(split_docs: List[Document], output_json_path: str):
    """
    将分割后的文档内容保存为JSON文件。
    参数:
    split_docs (List[Document]): 经过文本分割后的文档列表，每个元素为Document类型，包含页面内容和元数据等信息。
    output_json_path (str): 要保存的JSON文件的完整路径。
    """
    if not os.path.exists(output_json_path):
        os.makedirs(output_json_path)
    data_to_save = []
    for doc in split_docs:
        dict_data = {
            'metadata' : doc.metadata,
            'page_content' : doc.page_content
        }
        data_to_save.append(dict_data)
    with open(os.path.join(output_json_path, 'splits_docs_500chunks.json'), 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)


# 数据库创建-CHROMA
def create_my_db(split_docs, embedding_model_path):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
    # 定义持久化路径
    # 加载数据库
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )
    # 将加载的向量数据库持久化到磁盘上
    vectordb.persist()


# 数据库创建-milvus
def create_my_db_milvus(split_docs, embedding_model_path, URI):
    
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
    # 定义持久化路径
    # 加载数据库
    vectordb = Milvus.from_documents(
        documents=split_docs,
        embedding=embeddings,
        connection_args={"uri": URI},
        drop_old=True
    ) # 将加载的向量数据库持久化到磁盘上



def add_new_data_to_db(new_split_docs, persist_directory):
    # 加载已有的数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding=HuggingFaceEmbeddings(model_name=embedding_model_path)
    )
    # 将新的数据添加到数据库中
    vectordb.add_documents(new_split_docs)
    # 将更新后的数据库持久化到磁盘上
    vectordb.persist()


# 测试生成的document是否正确
# def t0():
#     file_path_list_all = get_file_name_form_folder(json_folder_path)
#     documents = parse_file_to_document(file_path_list_all)
#     print(documents)
#     for i, document in enumerate(documents):
#         print(f"{i+1}：document: {document}")

# 运行主函数
# if __name__ == "__main__":
    # json_folder_path = 'data/policy'
    # embedding_model_path = "/root/autodl-tmp/BAAI/bge-large-zh-v1___5"
    # persist_directory = 'data_base/vector_db/milvus/bge_large/bge_large.db'
    # create_my_db_milvus(split_texts, embedding_model_path, persist_directory)  # 创建向量数据库，并传入数据

    # 将分割文本保存成json格式
    
    # json_folder_path = 'data/policy'
    # save_splits_to_json_folder = 'data'  # 修改为表示文件夹的路径
    # file_path_list_all = get_file_name_form_folder(json_folder_path)
    # split_texts = split_text(file_path_list_all)
    # save_splits_to_json(split_texts, save_splits_to_json_folder)



