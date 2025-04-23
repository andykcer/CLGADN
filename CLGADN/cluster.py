import pandas as pd
import numpy as np
import matplotlib
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
import torch
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder


# 定义T-sne函数
def tsne(data, n):
    # transfer2 = PCA(n_components=2)
    tsne = TSNE(n_components=2, perplexity=250, n_iter=1000, init='random')  ###使用fit_transform而不是fit,因为TSNE没有transform方法
    digits_tsne = tsne.fit_transform(data)  ###运行时间较久
    return digits_tsne


plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置为宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

datapath = '../group_recall/'
all_ratings = pd.read_csv(datapath + "all_ratings.csv")
course_data = pd.read_excel(datapath + 'course_data.xlsx')
course_data = course_data[course_data['id'].isin(all_ratings.courses.unique())]
# Group the user-item interactions by item_id and sum the clicks
item_clicks = all_ratings.groupby('courses')['student_id'].nunique()

# Sort the items by click counts in descending order
item_clicks_sorted = item_clicks.sort_values(ascending=False)
# Create a new DataFrame with the sorted item clicks
sorted_items = pd.DataFrame({'courses': item_clicks_sorted.index, 'cnt': item_clicks_sorted.values})
courses_cnt = sorted_items.set_index("courses").to_dict()["cnt"]
course_data = course_data[['id', 'merged_title', 'category_id', 'category']]
course_data['cnt'] = course_data['id'].map(courses_cnt)

# 连续型特征等间隔分箱
data_cnt = course_data['cnt'].values.reshape(-1, 1)
KBins = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile").fit_transform(data_cnt)
print(KBins)

cate = ['社科·法律', '文学', '化学', '艺术·设计', '环境·地球', '生命科学', '电子', '计算机', '外语',
        '物理', '教育', '经管·会计', '其他', '历史', '工程', '哲学', '创业', '医学·健康', '数学',
        '建筑', '大学先修课']
# # 1 文科 2 理科 3工科 4 法学 5 艺术 6 商科 7医学 8其他
# cate_idx = [4, 1, 2, 5, 3, 3, 3, 3, 1, 2, 1, 6, 8, 1, 3, 1, 6, 7, 2, 3, 8]

# 1文科 2理科 3工科 4其他
cate_idx = [1, 1, 2, 4, 4, 4, 3, 3, 1, 2, 1, 1, 4, 1, 3, 1, 4, 4, 2, 3, 4]
cate_idx_map = dict(zip(cate, cate_idx))
course_data['category'] = course_data['category'].map(cate_idx_map)
n = course_data.category_id.nunique()  # 21
clmodel = torch.load('clgadn.pkl')

idx = course_data.id.values

all_users1, all_items1 = clmodel.computer(clmodel.Graph1)  # 得到传播后的embedding
all_users2, all_items2 = clmodel.computer(clmodel.Graph2)
all_items = 0.5 * all_items1 + 0.5 * all_items2
course_embedding = all_items.cpu().detach().numpy()
course_embedding = course_embedding[idx, :]
print(course_embedding)

cate_embedding = clmodel.category_embedding.weight.cpu().detach().numpy()
cate_idx = course_data.category_id.values
cate_embedding = cate_embedding[cate_idx, :]

final_course_embedding = (course_embedding + cate_embedding)/2

# 实现T-sne降维与数据可视化
tsne_data = tsne(final_course_embedding, 20)
# c=KBins.flatten()
plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=course_data.category.values)
plt.show()

# fig = plt.figure()
# from mpl_toolkits.mplot3d import Axes3D
# #
# # ax = Axes3D(fig
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(tsne_data[:, 0], tsne_data[:, 1], tsne_data[:, 2], c=course_data.category_id.values)
# plt.show()


