import os
import subprocess
FNULL = open(os.devnull, 'w')

args_names = [
    'device-number',
    'batch-size',
    'n-layers',
    'model-type',
    'autocast',
]

def run_script(args):

    main_command = [
        'python3',
        'train_model.py',
    ]

    options = {}
    options.update(dict(zip(args_names, args)))
    

    whole_command = main_command
    for k, v in options.items():
        whole_command.append('--' + k)
        whole_command.append(v)

    print(whole_command)
    process = subprocess.Popen(args=whole_command, stdout=FNULL)
    return process

import numpy as np
model_type = 'resnet'
autocast = 'False'
configs = {
    (4, 224, 34, model_type, autocast),
    (5, 88, 50, model_type, autocast),
    (6, 56, 101, model_type, autocast),
    (7, 40, 152, model_type, autocast)
}

for config in configs:
    process = run_script([str(c) for c in config])