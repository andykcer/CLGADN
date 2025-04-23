import math
import numpy as np
import heapq
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score


def eva_rating(candidate_list, predict_list, target_list):
    data_size = len(candidate_list)
    map_item_predict = {candidate_list[i]: predict_list[i] for i in range(data_size)}

    rank_list_1 = heapq.nlargest(1, map_item_predict, key=map_item_predict.get)
    rank_list_5 = heapq.nlargest(5, map_item_predict, key=map_item_predict.get)
    rank_list_10 = heapq.nlargest(10, map_item_predict, key=map_item_predict.get)
    rank_list_all = heapq.nlargest(data_size, map_item_predict, key=map_item_predict.get)

    hr1 = HitRatio(rank_list_1, target_list)
    hr5 = HitRatio(rank_list_5, target_list)
    ndcg5 = NDCG(rank_list_5, target_list)
    hr10 = HitRatio(rank_list_10, target_list)
    ndcg10 = NDCG(rank_list_10, target_list)
    mrr = MRR(rank_list_all, target_list)

    return hr1, hr5, ndcg5, hr10, ndcg10, mrr

# calculate metrics
def HitRatio(rank_list, target):
    for item in rank_list:
        if item in target:
            return 1
    return 0


def MRR(rank_list, target):
    for index, item in enumerate(rank_list):
        if item in target:
            return 1.0 / (index + 1.0)
    return 0


def NDCG(rank_list, target):
    for i in range(len(rank_list)):
        if rank_list[i] in target:
            return math.log(2) / math.log(i + 2)
    return 0


# plot loss, auc and metrics
def plot(title, xticks, values, labels):
    plt.figure(num=1, figsize=(8, 6))
    for i, value in enumerate(values):
        plt.plot(xticks, value, label=labels[i])
    plt.legend()
    plt.grid(axis='both')
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title(title)
    plt.show()

def evaluation(loader, model, device):
    model.eval()
    predict_array = np.array([])
    candidate_array = np.array([], dtype=int)
    label_array = np.array([], dtype=int)
    print('start testing...')
    with torch.no_grad():
        for inputs in tqdm(loader, desc='testing'):
            (student, courses, category, candidate, candidate_cate, label) = inputs

            label = label.float()
            # print('data to', device, 'done!')
            if device == 'cuda:0':
                student = student.cuda()

                courses = courses.cuda()
                category = category.cuda()

                candidate = candidate.cuda()
                candidate_cate = candidate_cate.cuda()

                label = label.cuda()
            # print("==================data location:", student.device)
            student, candidate, y_hat, _ ,_= model(student, courses, category, candidate, candidate_cate)

            predict_array = np.concatenate((predict_array, y_hat.cpu().flatten().numpy()))
            candidate_array = np.concatenate((candidate_array, candidate.cpu().numpy()))
            label_array = np.concatenate((label_array, label.cpu().numpy()))
            # taDBST_2layer_5headrget_list = candidate.cpu().numpy().tolist()[:1]
            # assert len(target_list) == 1
            # hr1, hr5, ndcg5, hr10, ndcg10, mrr = eva_rating(list(candidate.cpu().numpy()), y_hat_list, target_list)

    candidate_list = candidate_array.reshape(-1, 100)
    predict_list = predict_array.reshape(-1, 100)
    target_list = candidate_list[:, :1]

    hits_1, hits_5, ndcgs_5, hits_10, ndcgs_10, mrrs = [], [], [], [], [], []

    assert candidate_list.shape == predict_list.shape
    for i in range(len(candidate_list)):
        hr1, hr5, ndcg5, hr10, ndcg10, mrr = eva_rating(candidate_list[i], predict_list[i], target_list[i])
        hits_1.append(hr1)
        hits_5.append(hr5)
        ndcgs_5.append(ndcg5)
        hits_10.append(hr10)
        ndcgs_10.append(ndcg10)
        mrrs.append(mrr)

    auc = roc_auc_score(label_array, predict_array)
    hr1_result = np.array(hits_1).mean()
    hr5_result = np.array(hits_5).mean()
    ndcg5_result = np.array(ndcgs_5).mean()
    hr10_result = np.array(hits_10).mean()
    ndcg10_result = np.array(ndcgs_10).mean()
    mrr_result = np.array(mrrs).mean()
    return auc, hr1_result, hr5_result, ndcg5_result, hr10_result, ndcg10_result, mrr_result
