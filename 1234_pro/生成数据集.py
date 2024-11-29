# -*- coding: utf-8 -*-
import json
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from convert_file_to_vector_db import get_file_name_form_folder, split_text, create_my_db, json_folder_path
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from langchain.chains import RetrievalQA
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
import os

class Qwen2_LLM(LLM):
    # 基于本地 Qwen2 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, mode_name_or_path: str):
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.bfloat16,
                                                          device_map="auto")
        self.model.generation_config = GenerationConfig.from_pretrained(mode_name_or_path)
        print("完成本地模型的加载")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to('cuda')
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    @property
    def _llm_type(self) -> str:
        return "Qwen2_LLM"


def t1():
    persist_directory = 'data_base/vector_db/chroma/m3e_base'
    embedding_path = r"/root/autodl-tmp/m3e-base"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_path)

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    # 加载模型
    llm = Qwen2_LLM(mode_name_or_path="/root/autodl-tmp/Qwen/Qwen2___5-7B-Instruct")

    # llm.predict("你是谁")
    from langchain.prompts import PromptTemplate

    # 我们所构造的 Prompt 模板
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    有用的回答:"""

    # 调用 LangChain 的方法来实例化一个 Template 对象，该对象包含了 context 和 question 两个变量，在实际调用时，这两个变量会被检索到的文档片段和用户提问填充
    qa_chain_prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    # print(qa_chain_prompt)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_chain_prompt})

    """构建数据集"""
    # 读取CSV文件
    df = pd.read_csv('data/QA/qa_policy_eval.csv')
    
    # 假设我们想要遍历的列名为'ColumnName'
    questions = df['Question'].tolist()
    ground_truths = df['Answer'].tolist()
    ground_truths = [[item] for item in ground_truths]
    # 设置每批处理的问题数量
    batch_size = 50  # 可以根据你的内存大小调整这个值
    
    # 初始化临时文件列表
    temp_files = []
    
    retriever = vectordb.as_retriever()
    
    # 分批处理问题
    for i in tqdm(range(0, len(questions), batch_size), desc="Processing queries"):
        batch_questions = questions[i:i+batch_size]
        batch_answers = []
        batch_contexts = []
        batch_ground_truths = ground_truths[i:i+batch_size]
        batch_references = [" ".join(gt) if isinstance(gt, list) else gt for gt in batch_ground_truths]
        
        # 处理每个问题
        for query in batch_questions:
            ans = qa_chain({"query": query})['result']  # 使用rag链得到answer
            batch_answers.append(ans)
            cont = [docs.page_content for docs in retriever.get_relevant_documents(query)]
            batch_contexts.append(cont)
        
        # 将这批数据保存为一个临时JSON文件
        temp_data = {
            "question": batch_questions,
            "answer": batch_answers,
            "contexts": batch_contexts,
            "ground_truths": batch_ground_truths,
            "reference": batch_references
        }
        with open(f"temp_data_{i}.json", "w", encoding="utf-8") as f:
            json.dump(temp_data, f, ensure_ascii=False, indent=4)
        temp_files.append(f"temp_data_{i}.json")
    
    # 合并所有临时文件到一个最终的JSON文件
    with open("data/dataset_qwen_conan_eval.json", "w", encoding="utf-8") as final_file:
        for temp_file in temp_files:
            with open(temp_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                json.dump(data, final_file, ensure_ascii=False, indent=4)
                final_file.write("\n")  # 在每个数据块之间添加换行符
    

if __name__ == '__main__':
    """运行之前，确保相对应的embedding向量数据库已经生成"""
    t1()

