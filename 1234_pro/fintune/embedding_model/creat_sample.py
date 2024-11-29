import os

import pandas as pd
from tqdm.auto import tqdm
import math
import json


def build_qa_samples(df, neg_batch_size=-1, n_neg_batch=5):
    """
    构建qa样本
    :param df: 包含qa的DataFrame，共两列，question和answer
    :param neg_batch_size: 每个批次中的负样本数量，为-1时表示将所有负样本和单个正样本配对，否则会将负样本拆开，结果中的query可能会重复
    :param n_neg_batch:表示需要多少个批次来处理所有的负样本；考虑了所有的负样本，确保他们能够被均匀分配到不同的批次中；
    如果neg_batch_size是-1,n_neg_batch将不被使用，表示所有的负样本将与单个正样本配对，不涉及分批
    :return:
    """
    data = []
    for inx, row in tqdm(df.iterrows(), total=len(df)):
        question = row['Question']
        answer = row['Answer']
        """
        NOTO:可以筛选同类型的问题，增加难度；
        我们可以将同一类别下的不同问题作为负样本（即不是正确答案的样本），这样模型就需要学会在相似的问题中找到正确的答案，从而提高模型的鲁棒性和准确性。
        做法：按照title排序，一般选择负例会选下一个问题，由于数据同一主题的问题是放在一起的，下一个问题一般和当前问题属于同一主题问题
        """
        # 创建负样本:选择与当前行的question不同的其他的question下的答案,作为负样本--->在df中选择一条与当前问题不同的问题的答案
        neg_samples = df[df['Question'] != question]['Answer'].values.tolist()
        neg_batch_count = math.ceil((len(df) - 1) / neg_batch_size)  # (len(df) - 1)保证当前问题不被视为负样本，ceil向上取整
        neg_batch_count = min(neg_batch_count, n_neg_batch)  # 保证负样本批次数量不超过给定的n_neg_batch,保证问题可以被均匀分配，

        prompt = "Given a question, retrieve from context that answer the question.",
        type =  "str"

        neg_sample = []
        for neg_batch_idx in range(neg_batch_count):
            # 从负样本列表中提取当前批次的负样本
            neg_batch_samples = neg_samples[neg_batch_idx * neg_batch_size : (neg_batch_idx + 1) * neg_batch_size]
            neg_batch_samples = [item for item in neg_batch_samples if item != answer]
            neg_sample = neg_batch_samples
        data.append(
            {
                'query': question,
                'pos': [answer],
                'neg': neg_sample,
                'prompt': prompt,
                'type': type
            }
                )
    return data

def save_samples_to_jsonl(datas, save_path):
    with open(os.path.join(save_path), 'w', encoding='utf-8') as f:
        for sample in datas:
            json.dump(sample, f, ensure_ascii=False)  # 直接使用json.dump，不需要手动写换行符了，因为每个dump操作后会自动换行
            f.write('\n')
    print(f"数据已保存至{save_path}")


if __name__ == '__main__':
    output_path = '../../data/output/embed_samples'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    qa_df = pd.read_excel('../../data/QA/qa_data_1250.xlsx')

    # 训练数据的创建与保存
    train_data = qa_df[qa_df['dataset'] == 'train']
    save_path_train_data = os.path.join(output_path, 'qa_pairs_train.jsonl')
    samples_train = build_qa_samples(train_data,neg_batch_size=7,n_neg_batch=143)
    save_samples_to_jsonl(samples_train, save_path_train_data)




