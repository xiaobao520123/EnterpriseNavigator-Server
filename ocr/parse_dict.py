import os
import re
import numpy as np

def get_dict(path, add_space=False, add_eos=False):
    label_dict = dict()
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            m = re.match(r'(\d+) (.*)', line)
            idx, label = int(m.group(1)), m.group(2)
            label_dict[idx] = label 
        if add_space:
            idx = idx + 1
            label_dict[idx] = ' ' 
        if add_eos:
            idx = idx + 1
            label_dict[idx] = 'EOS' 
    return label_dict