import argparse
import random
import subprocess

params = {
    'batch_size': [32, 64, 128, 256],
    'num_layers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'hidden_size': [32, 64, 128, 256, 512, 1024],
}

parser = argparse.ArgumentParser()
parser.add_argument("--num_exps", type=int, default=2)
args = parser.parse_args()

for i in range(args.num_exps):
    sel_params = {k: random.choice(v) for k, v in params.items()}
    func = ['python3', 'mlp_mem_alloc.py',
            '--batch_size', str(sel_params['batch_size']),
            '--num_layers', str(sel_params['num_layers']),
            '--hidden_size', str(sel_params['hidden_size'])]
    print(' '.join(func))
    subprocess.call(func)
