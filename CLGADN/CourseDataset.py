import torch.utils.data as data
import pandas as pd
import os
import torch
import numpy as np

# DATA_PATH = "../data/courses/"
from Parse import parse_mooc_args


class CourseDataset(data.Dataset):
    """Course dataset."""

    def __init__(self, args, is_training=True):
        self.args = args
        self.num_course = args.num_course  # 课程数量,不包括填充的0 [1,2583]
        self.num_student = args.num_student  # 学生数量 [0,58906]
        self.num_category = args.num_category  # 课程类别数量，不包括填充的0 [1,23]
        self.quick_test = self.args.quick_test
        self.is_training = is_training
        self.instances_file = self.args.data_path + ('train_data_transformed_01.csv'
                                                     if self.is_training else "test_data_transformed.csv")
        self.course_file = self.args.data_path + 'courses_info_with_pre.csv'

        # 负样本 如果是训练集则是4 否则为99
        self.num_ng = 4 if is_training else 99

        if self.quick_test:
            self.instances = pd.read_csv(self.instances_file, delimiter=",", nrows=100)
        else:
            self.instances = pd.read_csv(self.instances_file, delimiter=",")
        self.instances = self.instances[self.instances['hist_len'] != 0].reset_index(drop=True)
        self.cast_instances()


    def cast_instances(self):
        def process_hist(x):
            x = np.array([int(i) for i in x.split(',')])
            x = np.concatenate((x[x == 0], x[x != 0]))
            return x
        self.instances['courses'] = self.instances['courses'].apply(process_hist)
        self.instances['category_id'] = self.instances['category_id'].apply(process_hist)

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, idx):
        data = self.instances.iloc[idx]
        student = data.student_id
        # 注册课程信息 第一个是目标课程
        courses = data.courses
        category = data.category_id

        # # 候选课程信息
        candidate = data.candidate
        candidate_cate = data.candidate_cate
        # label
        label = data.label

        courses = torch.LongTensor(courses)
        category = torch.LongTensor(category)

        return student, courses, category, candidate, candidate_cate, label


if __name__ == '__main__':
    args = parse_mooc_args()
    print("Loading datasets")

    train_dataset = CourseDataset(args=args, is_training=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=os.cpu_count(),
    )

    # test_dataset = CourseDataset(args=args, is_training=False)
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=args.test_batch_size,
    #     shuffle=False,
    #     num_workers=os.cpu_count(),
    # )
    for index, inputs in enumerate(train_loader):
        (student, courses, category, candidate, candidate_cate, label) = inputs
    print("Done")
