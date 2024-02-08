import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="SACL")

    # ===== log ===== #
    parser.add_argument('--desc', type=str, default="", help='EXP description')
    parser.add_argument('--log', action='store_true', default=False, help='log in file or not')
    parser.add_argument('--log_fn', type=str, default=None, help='log file_name')
    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="music", help="Choose a dataset:[music, movie, book]")
    parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')

    parser.add_argument('--l2', type=float, default=1e-4, help='l2')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")

    parser.add_argument("--node_dropout", type=int, default=1, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=int, default=1, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")

    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--cuda", type=int, default=1, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--cross_cl_reg', type=float, default=0.01, help='contrastive learning loss weights')
    parser.add_argument('--cross_cl_tau', type=float, default=0.1, help='temperature hyperparameters for contrastive learning')

    parser.add_argument('--Ks', nargs='?', default='[20]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--mae_rate', type=float, default=0.05, help='mask size for MAE')
    parser.add_argument('--kg_dropout', type=bool, default=True, help='kg dropout')
    parser.add_argument('--ui_dropout', type=bool, default=True, help='ui dropout')

    parser.add_argument('--aug_kg_rate', type=float, default=0.2, help='long-tail augmentation keeping rate for knowledge graph')
    parser.add_argument('--aug_ui_rate', type=float, default=0.8, help='long-tail augmentation keeping rate for user-item graph')
    parser.add_argument('--kl_eps', type=float, default=0.1, help='KL weight')
    parser.add_argument('--mi_eps', type=float, default=0.1, help='Missing weight')

    parser.add_argument("--annealing_type", type=int, default=0, help="annealing_type：0、1、2...")

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
    
    # ===== save model ===== #
    parser.add_argument("--save", action='store_true', default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")

    return parser.parse_args()
