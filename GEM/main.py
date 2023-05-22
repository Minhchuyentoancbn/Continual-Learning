import argparse
import sys

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--finetune', type=bool, default=True)
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--memory_strength', type=float, default=0.3)
    parser.add_argument('--n_memories', type=int, default=256)
    parser.add_argument('--memory_strength', type=float, default=0.5)
    parser.add_argument('--n_memories', type=int, default=256)
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])