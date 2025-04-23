import os
import pickle
import random
import numpy as np
import torch
import scipy.sparse as sp
from time import time
from scipy.sparse import csr_matrix
import pandas as pd
from tqdm import tqdm


def seed_everthing(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    np.random.seed(seed)
    random.seed(seed)


class myGraph():
    def __init__(self, args):
        super().__init__()
        super(myGraph, self).__init__()
        self.args = args
        self.num_student = args.num_student
        self.num_category = args.num_category
        self.num_course = args.num_course
        self.Graph = None
        self.getUserItemNet()
        self.getCourseOrderNet()

    def getCourseOrderNet(self):
        course_info = pd.read_csv(self.args.data_path + 'courses_info_with_pre.csv')
        self.predCourse = []
        self.SuccCourse = []
        for index, row in course_info.iterrows():
            if row['required_course1'] != 0:
                self.predCourse.append(row['required_course1'])
                self.SuccCourse.append(row['id'])
                self.predCourse.append(row['id'])
                self.SuccCourse.append(row['required_course1'])
            if row['required_course2'] != 0:
                self.predCourse.append(row['required_course2'])
                self.SuccCourse.append(row['id'])
                self.predCourse.append(row['id'])
                self.SuccCourse.append(row['required_course2'])
            if row['required_course3'] != 0:
                self.predCourse.append(row['required_course3'])
                self.SuccCourse.append(row['id'])
                self.predCourse.append(row['id'])
                self.SuccCourse.append(row['required_course3'])

        self.CourseOrderNet = csr_matrix((np.ones(2 * len(self.predCourse)), (
        np.append(self.predCourse, self.SuccCourse), np.append(self.SuccCourse, self.predCourse))),
                                         shape=(self.num_course + 1, self.num_course + 1))
        print("KG is ready to go")

    def getUserItemNet(self):
        train_data = pd.read_csv(self.args.data_path + 'train_data.csv')
        self.trainUser = train_data['student_id'].values
        self.trainItem = train_data['courses'].values

        # (users,items), bipartite graph shape (58907, 2584)
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.num_student, self.num_course + 1))
        print("train dataset is ready to go")

    def getSparseGraph(self):
        print("loading KG and adjacency matrix....")
        if self.Graph is None:
            try:
                kg_mat = sp.load_npz(self.args.data_path + 'kg_mat.npz')
                adj_mat = sp.load_npz(self.args.data_path + 'adj_mat.npz')
                print("successfully loaded KG and adjacency matrix")
                norm_adj = adj_mat
                norm_kg = kg_mat
            except:
                print("generating adjacency matrix")
                s = time()

                ## 第一个图
                adj_mat = sp.dok_matrix(
                    (self.num_student + self.num_course + 1, self.num_student + self.num_course + 1), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()  # (58907, 2584)

                adj_mat[:self.num_student, self.num_student:] = R
                adj_mat[self.num_student:, :self.num_student] = R.T

                adj_mat = adj_mat.todok()
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_adj_mat...")
                sp.save_npz(self.args.data_path + 'adj_mat.npz', norm_adj)

                ## 第二个图
                s = time()
                kg_mat = sp.dok_matrix(
                    (self.num_student + self.num_course + 1, self.num_student + self.num_course + 1), dtype=np.float32)
                KG = self.CourseOrderNet.tolil()  # (2584, 2584)
                kg_mat[self.num_student:, self.num_student:] = KG

                kg_mat = kg_mat.todok()
                rowsum = np.array(kg_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                norm_kg = d_mat.dot(kg_mat)
                norm_kg = norm_kg.dot(d_mat)
                norm_kg = norm_kg.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_kg_mat...")
                sp.save_npz(self.args.data_path + 'kg_mat.npz', norm_kg)

            self.Graph1 = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph1 = self.Graph1.coalesce()

            self.Graph2 = self._convert_sp_mat_to_sp_tensor(norm_kg)
            self.Graph2 = self.Graph2.coalesce()
        return self.Graph1, self.Graph2

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


if __name__ == '__main__':
    x = [1, 2, 3, 4]
