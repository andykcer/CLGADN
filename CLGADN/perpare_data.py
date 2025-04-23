import os

from keras_preprocessing.sequence import pad_sequences

from Parse import parse_mooc_args
import pandas as pd
import numpy as np
from tqdm import tqdm


def trn_val_split(args):
    all_click_df = pd.read_csv(args.data_path + 'all_ratings.csv')
    all_click = all_click_df
    num_student = all_click.student_id.nunique()
    all_user_ids = all_click.student_id.unique()
    all_item_ids = all_click.courses.unique()
    # replace=True表示可以重复抽样，反之不可以 20%用户当作验证集
    sample_user_ids = np.random.choice(all_user_ids, size=int(num_student * 0.2), replace=False)

    test_pos = all_click_df[all_click_df['student_id'].isin(sample_user_ids)]
    train_pos = all_click_df[~all_click_df['student_id'].isin(sample_user_ids)]

    # 将测试集中的最后一次点击给抽取出来作为答案
    test_pos = test_pos.sort_values(['student_id', 'date'])
    test_ans = test_pos.groupby('student_id').tail(1)
    test_pos = test_pos.groupby('student_id').apply(lambda x: x[:-1]).reset_index(drop=True)
    train_data = pd.concat([train_pos, test_pos])  # 除了测试集用户的最后一跳都在里面。
    train_data = train_data.sort_values(['student_id', 'date'])

    train_data.to_csv(args.data_path + 'train_data.csv', index=False, sep=",")
    test_pos.to_csv(args.data_path + 'test_pos.csv', index=False, sep=",")
    test_ans.to_csv(args.data_path + 'test_ans.csv', index=False, sep=",")
    print("The dataset has been split")
    return all_item_ids


# 将输入的数据进行padding，使得序列特征的长度都一致
def gen_model_input(train_set, seq_max_len=20):
    student_id = np.array([line[0] for line in train_set])
    courses = [line[1] for line in train_set]
    category_id = np.array([line[2] for line in train_set])
    candidate = np.array([line[3] for line in train_set])
    candidate_cate = np.array([line[4] for line in train_set])
    label = np.array([line[5] for line in train_set])
    train_hist_len = np.array([line[6] for line in train_set])

    courses_seq_pad = pad_sequences(courses, maxlen=seq_max_len, padding='post', truncating='pre', value=0)
    courses_str = [",".join([str(v) for v in line]) for line in courses_seq_pad]
    category_id_seq_pad = pad_sequences(category_id, maxlen=seq_max_len, padding='post', truncating='pre', value=0)
    category_id_str = [",".join([str(v) for v in line]) for line in category_id_seq_pad]

    train_model_input = {"student_id": student_id, "courses": courses_str, "category_id": category_id_str
        , "candidate": candidate, "candidate_cate": candidate_cate, "label": label, "hist_len": train_hist_len}
    train_data_transformed = pd.DataFrame(train_model_input)
    # 每个用户只保留最近10次正例
    train_data_transformed = train_data_transformed.groupby(['student_id']).tail(100)
    train_data_transformed = train_data_transformed.reset_index(drop=True)
    return train_data_transformed

def gen_train_data_set(data, all_item_ids, cid2cateid_dict, negsample=0):
    data.sort_values("date", inplace=True)
    item_ids = all_item_ids
    train_set = []
    for reviewerID, hist in tqdm(data.groupby('student_id')):
        pos_course_list = hist['courses'].tolist()
        pos_cate_list = hist['category_id'].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_course_list))  # 用户没看过的文章里面选择负样本
            neg_list = np.random.choice(candidate_set, size=len(pos_course_list) * negsample, replace=True)
            # 滑窗构造正负样本
            for i in range(0, len(pos_course_list)):
                course_hist = pos_course_list[:i]
                cate_list = pos_cate_list[:i]
                train_set.append((reviewerID, course_hist[:], cate_list[:], pos_course_list[i], pos_cate_list[i], 1,
                                  len(course_hist[:])))  # 正样本 [user_id, his_item, pos_item, label, len(his_item)]
                for negi in range(negsample):
                    neg_course = neg_list[i * negsample + negi]
                    neg_cate = cid2cateid_dict[neg_course]
                    train_set.append((reviewerID, course_hist[:], cate_list[:], neg_course, neg_cate, 0,
                                      len(course_hist[:])))
    return train_set

def gen_test_data_set(data, all_item_ids, cid2cateid_dict, negsample=0):
    data.sort_values("date", inplace=True)
    item_ids = all_item_ids
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('student_id')):
        pos_course_list = hist['courses'].tolist()
        pos_cate_list = hist['category_id'].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_course_list))  # 用户没看过的文章里面选择负样本
            neg_list = np.random.choice(candidate_set, size=len(pos_course_list) * negsample, replace=True)

            course_hist = pos_course_list[:-1]
            cate_list = pos_cate_list[:-1]
            test_set.append((reviewerID, course_hist[:], cate_list[:], pos_course_list[-1], pos_cate_list[-1], 1,
                              len(course_hist[:])))  # 正样本 [user_id, his_item, pos_item, label, len(his_item)]
            for negi in range(negsample):
                neg_course = neg_list[negi]
                neg_cate = cid2cateid_dict[neg_course]
                test_set.append((reviewerID, course_hist[:], cate_list[:], neg_course, neg_cate, 0,
                                  len(course_hist[:])))
    return test_set


def get_cid2cateid_dict(courses_info):
    cid2cateid_dict = {}
    for _, value in courses_info.iterrows():
        cid2cateid_dict[value['id']] = int(value['category_id'])
    return cid2cateid_dict


if __name__ == '__main__':
    args = parse_mooc_args()
    courses_info = pd.read_csv(args.data_path + 'courses_info_with_pre.csv')
    cid2cateid_dict = get_cid2cateid_dict(courses_info)

    all_item_ids = trn_val_split(args)
    train_data = pd.read_csv(args.data_path + 'train_data.csv');
    train_set = gen_train_data_set(train_data, all_item_ids, cid2cateid_dict, negsample=4)
    train_data_transformed = gen_model_input(train_set, seq_max_len=20)
    # 保存训练集
    train_data_transformed.to_csv(args.data_path + 'train_data_transformed_01.csv', index=False, sep=",")

    test_pos = pd.read_csv(args.data_path + 'test_pos.csv');
    test_ans = pd.read_csv(args.data_path + 'test_ans.csv');
    test_data = pd.concat([test_pos, test_ans])
    test_set = gen_test_data_set(test_data, all_item_ids, cid2cateid_dict, negsample=99)
    test_data_transformed = gen_model_input(test_set, seq_max_len=20)
    # 保存训练集
    test_data_transformed.to_csv(args.data_path + 'test_data_transformed.csv', index=False, sep=",")
