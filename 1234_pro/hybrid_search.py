import pandas as pd
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)
from langchain_community.embeddings import HuggingFaceEmbeddings

from milvus_model.hybrid import BGEM3EmbeddingFunction
from convert_file_to_vector_db import get_file_name_form_folder, split_text, create_my_db_milvus


json_folder_path = 'data/policy'
file_path_list_all = get_file_name_form_folder(json_folder_path)
split_texts = split_text(file_path_list_all)
docs = split_texts  # 此时是document列表
doc_list = []
for i in range(len(docs)):
    doc = docs[i].page_content
    doc_list.append(doc)
# print(doc_list)
# docs = [
#     "第一章 总则第一条为加强企业增资业务评审专家管理，规范评审工作，提高评审质量，根据上海联合产权交易所（以下简称“联交所”）《企业增资业务规则（试行）》等相关规定，制定本操作流程。",
#     "第二条专家评审工作应按照本操作流程的规定，遵循职责明确、监督制衡、程序清晰的原则，公正、公平地组织评审，并接受联交所管理和相关市场主体的监督。",
#     "第三条本操作流程所称评审专家是指符合本操作流程规定的条件和要求，以独立身份参加企业增资择优选择投资人评审工作的各类专业人员。",
#     "第四条联交所负责建立企业增资评审专家库（以下简称“专家库”）。"
# ]

# 使用BGE-M3模型生成嵌入向量
ef = BGEM3EmbeddingFunction(
    model_name='autodl-tmp/BAAI/bge-m3', # Specify the model name
)
dense_dim = ef.dim["dense"]
# queries = "专家评审工作应该遵循什么原则？"
docs_embeddings = ef.encode_documents(doc_list)
# print(docs_embeddings)

# 连接到Milvus
connections.connect(uri="data_base/vector_db/milvus/split1000/bge_m3/bge_m3.db")
# 定义集合的数据模式
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
]
schema = CollectionSchema(fields)

# 创建集合（如果存在则删除旧的）
col_name = "hybrid_demo"
if utility.has_collection(col_name):
    Collection(col_name).drop()
col = Collection(col_name, schema, consistency_level="Strong")

# 为向量字段创建索引
sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
col.create_index("sparse_vector", sparse_index)
# dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
dense_index = {"index_type": "HNSW", "metric_type": "L2"}
col.create_index("dense_vector", dense_index)
col.load()

# 批量插入文档和嵌入向量到集合中
for i in range(0, len(doc_list), 50):
    batched_entities = [
        doc_list[i : i + 50],
        docs_embeddings["sparse"][i : i + 50],
        docs_embeddings["dense"][i : i + 50],
    ]
    col.insert(batched_entities)

# 定义搜索函数

def dense_search(col, query_dense_embedding, limit=10):
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}  # 修改度量类型为 L2
    res = col.search(
        [query_dense_embedding],
        anns_field="dense_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]

def sparse_search(col, query_sparse_embedding, limit=10):
    search_params = {
        "metric_type": "IP",
        "params": {},
    }
    res = col.search(
        [query_sparse_embedding],
        anns_field="sparse_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]

def hybrid_search(
    col,
    query_dense_embedding,
    query_sparse_embedding,
    sparse_weight=1.0,
    dense_weight=1.0,
    limit=10,
):
    dense_search_params = {"metric_type": "L2", "params": {"nprobe": 10}}  # 修改度量类型为 L2
    dense_req = AnnSearchRequest(
        [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
    )
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text"]
    )[0]
    return [hit.get("text") for hit in res]

# 执行搜索
query = "专家评审工作应该遵循什么原则？"
query_embeddings = ef([query])

dense_results = dense_search(col, query_embeddings["dense"][0])
sparse_results = sparse_search(col, query_embeddings["sparse"]._getrow(0))
hybrid_results = hybrid_search(
    col,
    query_embeddings["dense"][0],
    query_embeddings["sparse"]._getrow(0),
    sparse_weight=0.7,
    dense_weight=1.0,
)

# 打印搜索结果
print("Dense Search Results:", dense_results)
print("*"*100)
print("Sparse Search Results:", sparse_results)
print("*"*100)
print("Hybrid Search Results:", hybrid_results)