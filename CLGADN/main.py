from load_data import load_data
from CLGADN import CLGADN
import torch.nn as nn
from Evaluation import *
from Parse import parse_mooc_args
from Utils import seed_everthing
import os
from log_helper import *
from helper import *
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.multiprocessing.set_sharing_strategy('file_system')
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':

    args = parse_mooc_args()

    args.quick_test = False

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    # logging.info(args)

    seed_everthing(args.seed)

    logging.info('quick_test:{} and transformer:{}'.format(args.quick_test, args.transformer))

    train_loader, test_loader, Graph1, Graph2 = load_data(args)

    args.device = 'cpu'
    if args.use_cuda and torch.cuda.is_available():
        logging.info('cuda ready...')
        args.device = 'cuda:0'
    else:
        logging.info('cpu ready...')
    logging.info(args)

    model = CLGADN(args, Graph1, Graph2)
    # logging.info(model)
    loss_func = nn.BCELoss(reduction='mean').to(args.device)
    # print("===================== model location:", next(model.parameters()).device)

    # print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        total_loss_epoch = 0.0
        total_tmp = 0
        model.train()
        logging.info('***************************************')
        logging.info('epoch:{}_start training...'.format(epoch))
        for inputs in tqdm(train_loader, desc='training'):
            # print(x.shape,x.dtype)
            (student, courses, category, candidate, candidate_cate, label) = inputs

            label = label.float()
            # print('data to', device, 'done!')
            if args.device == 'cuda:0':
                student = student.cuda()

                courses = courses.cuda()
                category = category.cuda()

                candidate = candidate.cuda()
                candidate_cate = candidate_cate.cuda()

                label = label.cuda()
            # print("==================data location:", student.device)
            student, candidate, y_hat, _, cl_loss = model(student, courses, category, candidate, candidate_cate)

            optimizer.zero_grad()
            loss = loss_func(y_hat.flatten(), label)
            loss += 100*cl_loss  ## 100 超参数 可调整

            loss.backward()
            optimizer.step()
            total_loss_epoch += loss.item()
            total_tmp += 1
        auc, hits_1, hits_5, ndcgs_5, hits_10, ndcgs_10, mrrs = evaluation(test_loader, model, args.device)

        test_msg = 'epoch: %d     test auc: %.4f     test hits@1: %.4f     test hits@5: %.4f     test ndcg@5: %.4f     test hits@10: %.4f     test ndcg@10: %.4f     test MRR: %.4f' % (
            epoch, auc, hits_1, hits_5, ndcgs_5, hits_10, ndcgs_10, mrrs)
        logging.info(test_msg)

        # 效果最好，直接保存然后break了
        # if epoch == 4:
        #     torch.save(model, 'clgadn.pkl')
        #     break
    # os.system("shutdown")
