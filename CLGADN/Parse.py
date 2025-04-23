import argparse


def parse_mooc_args():
    parser = argparse.ArgumentParser(description="Go BST")
    parser.add_argument('--data_path', type=str, default="../data/courses/",
                        help="path to load dataset")

    parser.add_argument('--quick_test', type=bool, default=True,
                        help="quick test for debugging")
    parser.add_argument('--transformer', type=bool, default=True,
                        help="using transformer or not")
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help="using gpu or not")
    parser.add_argument('--decive', type=str, default="cpu",
                        help="default decive")

    parser.add_argument('--num_student', type=int, default=58907,
                        help="the num of student")
    parser.add_argument('--num_course', type=int, default=2583,
                        help="the num of course")
    parser.add_argument('--num_category', type=int, default=23,
                        help="the num of course category")

    parser.add_argument('--sequence_length', type=int, default=20,
                        help="the length of user history")

    parser.add_argument('--train_batch_size', type=int, default=1024,
                        help="the batch size for training procedure")
    parser.add_argument('--test_batch_size', type=int, default=2048,
                        help="the batch size for testing procedure")

    parser.add_argument('--recdim', type=int, default=100,
                        help="the embedding size of user and item")

    parser.add_argument('--layer', type=int, default=1,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int, default=0.2,
                        help="using the dropout or not")
    parser.add_argument('--keep_prob', type=float, default=0.6,
                        help="the batch size for training procedure")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--tensorboard', type=int, default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str, default="lgn")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--nhead', type=int, default=1, help='number of head')
    args = parser.parse_args()

    save_dir = 'log/recdim{}_LGCNLayer{}_droprate{}_lr{}_weightdecay{}_nhead{}/'.format(
        args.recdim, args.layer, args.dropout, args.lr, args.weight_decay, args.nhead)
    args.save_dir = save_dir

    return args


if __name__ == '__main__':
    args = parse_mooc_args()
    print(args)
    print(args.save_dir)
