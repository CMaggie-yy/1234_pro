import getpass

import pymysql
from zhipuai import ZhipuAI
import os
from langchain.llms.base import LLM
from langchain.chains import create_sql_query_chain

ZHIPUAI_API_KEY = "dd6585399f3103b90ba9e09079a2baeb.kq75iQGYWjkgZCcR"
LANGCHAIN_API_KEY = "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"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"  # 设置langsmith地址
LANGCHAIN_TRACING_V2 = True  # 让Langchain记录追踪信息
LANGCHAIN_PROJECT = "project_name"

# 数据库连接信息：
host = '192.168.10.138',
user = 'iflytek',
password = '.19900504tT',
database = 'xunfei666',
port = '3306'

os.environ["ZHIPUAI_API_KEY"] = ZHIPUAI_API_KEY

"""数据库的连接"""

# 连接数据库测试
# def test_db():
#     sql_file_path = "xunfei666.call_bid_info"
#     try:
#         # 判断sql_file_path是否是文件路径，如果是则读取文件内容作为查询语句
#         if os.path.isfile(sql_file_path):
#             with open(sql_file_path, 'r') as f:
#                 sql_query = f.read()
#         else:
#             # 如果不是文件路径，就假设它是表名并构建查询语句
#             sql_query = f"SELECT 项目名称 FROM {sql_file_path} LIMIT 10"
#
#         # 执行查询
#         cursor.execute(sql_query)
#
#         # 获取查询结果
#         results = cursor.fetchall()
#
#         # 遍历结果并打印（你可以根据实际需求进行更复杂的处理）
#         for row in results:
#             print(row)
#
#     except pymysql.Error as e:
#         print(f"查询过程中出现错误: {e}")
#
#     finally:
#         # 关闭游标和连接
#         cursor.close()
#         conn.close()

# test_db()
conn = pymysql.connect(host='192.168.10.138', user='iflytek', password='.19900504tT', database='xunfei666')
# 创建游标
cursor = conn.cursor()
from langchain_community.utilities import SQLDatabase

db_uri = "mysql+pymysql://iflytek:.19900504tT@192.168.10.138/xunfei666"
db = SQLDatabase.from_uri(db_uri)
res = db.run("SELECT 项目名称 FROM call_bid_info LIMIT 10")
print(res)

"""创建模型实例"""


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
            model="glm-4",  # 假设你一直使用这个型号，实际可根据需求调整
            messages=[{"role": "user", "content": prompt}]
        )
        text_content = response.choices[0].message.content

        return text_content


zhipuai_llm_obj = ZhipuAILlm(ZHIPUAI_API_KEY)

"""生成SQL查询链路"""
# chain = create_sql_query_chain(db=db, llm=zhipuai_llm_obj)
# response = chain.invoke({"question": "请帮我查询表格前10条项目名称"})
# print(response)
# sql_query = response.split(';')[0] + ';'
# print(sql_query)
# db.run(sql_query)
#
# """执行SQL命令"""
# cursor.close()
# conn.close()

"""生成并查询执行"""
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
write_query = create_sql_query_chain(db=db, llm=zhipuai_llm_obj)
execute_query = QuerySQLDataBaseTool(db=db)

chain = write_query | execute_query
res = chain.invoke({"question": "请帮我查询表格前10条项目名称"})
print(res)


