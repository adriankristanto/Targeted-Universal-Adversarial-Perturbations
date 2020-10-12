# reference: https://github.com/NetoPedro/Universal-Adversarial-Perturbations-Pytorch/blob/master/adversarial_perturbation.py
# reference: https://github.com/LTS4/universal
from targeted_deepfool import targeted_deepfool
import torch
import numpy as np 

def proj_lp(v, xi=10, p=np.inf):
    """
        v: resulting perturbation
        xi: controls the l_p magnitude of the perturbation (default = 10)
        p: norm to be used (default = np.inf)
    """
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    return v
