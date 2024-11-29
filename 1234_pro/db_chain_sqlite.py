# -*- coding: utf-8 -*-

import sqlite3
from operator import itemgetter
from typing import Optional, List

import pymysql
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from zhipuai import ZhipuAI
import os
from langchain.llms.base import LLM
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool


ZHIPUAI_API_KEY = "dd6585399f3103b90ba9e09079a2baeb.kq75iQGYWjkgZCcR"
LANGCHAIN_API_KEY = "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"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"  # 设置langsmith地址
LANGCHAIN_TRACING_V2 = True  # 让Langchain记录追踪信息
LANGCHAIN_PROJECT = "project_name"

os.environ["ZHIPUAI_API_KEY"] = ZHIPUAI_API_KEY


"""将ZhipuAI封装成一个符合langchain框架的LLM接口：
ZhipuAILlm继承自langchain.llms.base的LLM类，定制化一个接口
"""

class ZhipuAILlm(LLM):
    api_key: str
    client: ZhipuAI = None  # 在这里声明client属性

    def __init__(self, api_key):
        super().__init__(api_key=api_key)
        self.client = ZhipuAI(api_key=self.api_key)

    @property
    def _llm_type(self):
        return "zhipuai"

    def _call(self, prompt, stop=None):
        response = self.client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": prompt}]
        )
        text_content = response.choices[0].message.content
        # 移除\nSQLResult:以及后面的内容
        if '\nSQLResult:' in text_content:
            text_content = text_content.split('\nSQLResult:')[0]
        # 去除可能错误添加的反引号
        text_content = text_content.replace("```", "")
        return text_content


"""数据库的连接"""
# 连接到 SQLite 数据库
conn = sqlite3.connect('mydatabase.db')
cursor = conn.cursor()
db_uri = "sqlite:///mydatabase.db"
db = SQLDatabase.from_uri(db_uri)


"""创建模型实例"""
zhipuai_llm_obj = ZhipuAILlm(ZHIPUAI_API_KEY)

"""一个完整链路，对SQL运行结果再进行一个生成"""
write_query_chain = create_sql_query_chain(db=db, llm=zhipuai_llm_obj)
execute_query_chain = QuerySQLDataBaseTool(db=db)
answer_prompt = PromptTemplate.from_template(
    """根据以下用户提供的问题、对应的SQL查询以及查询结果，以清晰、专业的方式回答用户的问题，没有查到就说没有查到，不要杜撰答案。

用户问题：{question}
SQL查询：{query}
SQL结果：{result}

以下是一些在查询类似需求时正确的SQLite数据库查询示例，供你参考：
- 查询某个特定值的记录：SELECT "采购人名称" FROM call_bid_info WHERE "包名称" = 'CT机';
- 查询满足条件的前几条记录：SELECT "采购人名称" FROM call_bid_info WHERE "包名称" = 'CT机' LIMIT 5;

请注意，在生成SQL查询时，需要严格按照SQLite数据库的语法规范，避免使用反引号等可能导致语法错误的字符，应使用单引号来表示字符串值。

回答："""
)

answer = answer_prompt | zhipuai_llm_obj | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query_chain).assign(
        result=itemgetter("""query""") | execute_query_chain
    )
    | answer
)

res = chain.invoke({"question": "请帮我查询上海市老年医学中心的所有的包名称"})
print(res)
