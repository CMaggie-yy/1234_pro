from typing import List
import pandas as pd
import os
import logging

import convert_file_to_vector_db
from convert_file_to_vector_db import get_file_name_form_folder, split_text

# 配置logging模块基本设置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('recall.log'), logging.StreamHandler()])


def recall_topn(eval_file, topn_list: list[int]):
    df = pd.read_excel(eval_file)
    length = len(df.groupby('ID'))  # 集合去重,可计算一共多少个问题
    logging.info(f"{eval_file}文件一共{length}个问题。")
    num_questions = length
    result_log = ""
    for topn in topn_list:
        true_positives = 0
        for id, group in df.groupby('ID'):  # 根据ID分组，遍历每个组
            group_truths = set(group['title_question_form'])  # 集合去重,在我的表格里，同一个问题出现了10次
            recalls = group['recall_title'].astype(str).head(topn)  # 取前topn个
            hits = recalls.isin(group_truths)  # 创建一个布尔变量，代表当前召回的title是否是真的
            hits[hits.duplicated()] = False  # 去掉重复值，因为一个title不止召回一次，我们只要在topn中召回了就行，不计算次数
            true_positives += hits.sum()  # 有召回就+1，没有召回就+0，累计计算若干个问题召回成功了几个
        recall = true_positives / length
        result_log += f"top{topn}时召回了{true_positives}个；recall_{topn} = {recall}\n"
    logging.info(f"{result_log}")
    return result_log, num_questions


def mrr_topn(eval_file, topn_list):
    df = pd.read_excel(eval_file)
    length = len(df.groupby('ID'))
    logging.info(f"{eval_file}文件一共{length}个问题。")
    result_log = ""
    for topn in topn_list:
        mrr_sum = 0  # 重置mrr_sum
        recall_sum = 0
        for _, group in df.groupby('ID'):
            group_truths = set(group['title_question_form'])
            recalls = group['recall_title'].astype(str).head(topn)

            for index, recall in enumerate(recalls, start=1):
                if recall in group_truths:  # 第一次出现，就停止本次循环
                    mrr_sum += 1 / index
                    recall_sum += 1
                    break

        mrr_avg = mrr_sum / length
        result_log += f"top_k取{topn}时，一共召回了{recall_sum}个；mrr_{topn} = {mrr_avg}\n"
    logging.info(f"\n{result_log}")
    return result_log, length


def calculate_and_save(eval_file_path, topn_list: list[int], mrr=True, recall=True):
    result_log = ""
    num_questions = 0

    if mrr:
        mrr_result_log, mrr_num_questions = mrr_topn(eval_file_path, topn_list)
        result_log += mrr_result_log
        num_questions = mrr_num_questions
        base_name = 'mrr.txt'
        base_name01 = os.path.splitext(base_name)[0]
        file_name = os.path.splitext(os.path.basename(eval_file_path))[0]
        save_file_path = os.path.join(output, f"{base_name01}_{file_name}.txt")
        index = 0
        if os.path.exists(save_file_path):
            index += 1
            new_path = os.path.join(output, f"{base_name01}_{file_name}({index}).txt")
            save_file_path = new_path

        with open(save_file_path, 'w', encoding='utf-8') as f:
            f.write(str(f"{eval_file_path}文件一共{mrr_num_questions}个问题。\n" + mrr_result_log))
        logging.info(f"结果已经成功保存至{save_file_path}")

    if recall:
        recall_result_log, recall_num_questions = recall_topn(eval_file_path, topn_list)
        result_log += recall_result_log
        num_questions = recall_num_questions
        base_name = 'recall.txt'
        base_name01 = os.path.splitext(base_name)[0]
        file_name = os.path.splitext(os.path.basename(eval_file_path))[0]
        save_file_path = os.path.join(output, f"{base_name01}_{file_name}.txt")
        index = 0
        if os.path.exists(save_file_path):
            index += 1
            new_path = os.path.join(output, f"{base_name01}_{file_name}({index}).txt")
            save_file_path = new_path

        with open(save_file_path, 'w', encoding='utf-8') as f:
            f.write(str(f"{eval_file_path}文件一共{recall_num_questions}个问题。\n" + recall_result_log))
        logging.info(f"结果已经成功保存至{save_file_path}")

    return result_log, num_questions


if __name__ == '__main__':
    output = r'../data/output'
    os.makedirs(output, exist_ok=True)

    # file_path = '../data/embed_eval/milvus/bge_base.xlsx'
    # file_path = '../data/embed_eval/milvus/bge_large.xlsx'
    # file_path = '../data/embed_eval/milvus/m3e_base.xlsx'
    file_path = '../data/embed_eval/milvus/conan.xlsx'

    calculate_and_save(file_path, topn_list=[1, 5, 10])

    persist_directory = 'data_base/vector_db/milvus/bge_large/bge_large.db'
    if not os.path.exists(persist_directory):
        convert_file_to_vector_db.main()

