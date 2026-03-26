import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--g', dest='GPU', type=int, default=10,
                        help='GPU')
    parser.add_argument('-p', '--pp', dest='num_workers', type=int, default=2,
                        help='num_workers')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=250,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-mi', '--max_iter', dest='max_iter', type=int, default=20000,
                        help='max iteration number')
    parser.add_argument('-fs', '--fs', dest='num_frames_per_sample', type=int, default=300,
                        help='num_frames_per_sample')
    parser.add_argument('-stride', '--stride', dest='stride', type=int, default=30,
                        help='stride')
    parser.add_argument('-K', '--K', dest='num_expert', type=int, default=4,
                        help='num_block')
    parser.add_argument('-L', '--L', dest='num_block', type=int, default=1,
                        help='num_block')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=1024,
                        help='seed')

    return parser.parse_args()
