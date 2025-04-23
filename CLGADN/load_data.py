import torch
from CourseDataset import CourseDataset
from Parse import parse_mooc_args
from Utils import myGraph
import os

def load_data(args):
    train_dataset = CourseDataset(args, is_training=True)
    test_dataset = CourseDataset(args, is_training=False)
    Graph1, Graph2 = myGraph(args).getSparseGraph()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=4
    )

    print("already load train data and test data...")
    print('train data size:', len(train_dataset))
    print('test data size:', len(test_dataset))
    print('num_student:', train_loader.dataset.num_student)
    print('num_course:', train_loader.dataset.num_course)
    print('num_category:', train_loader.dataset.num_category)
    return train_loader, test_loader, Graph1, Graph2


if __name__ == '__main__':
    args = parse_mooc_args()
    args.quick_test = False
    train_loader, test_loader, Graph1, Graph2 = load_data(args)
    for inputs in test_loader:
        (student, courses, category, candidate, candidate_cate, label) = inputs