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

def targeted_universal_adversarial_perturbations(images, net, target_class, max_iter, xi=10, p=np.inf, overshoot=0.02, max_iter_deepfool=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    universal_v = torch.zeros(images[0][None, :, :, :].shape)

    num_images = len(images)
    image_indices = np.arange(num_images)

    i = 0

    while i < max_iter: 
        print(f"Iteration {i+1}/{max_iter}\n")
        i += 1

        # shuffle the dataset
        np.random.shuffle(image_indices)

        # for each image in the dataset
        for index in image_indices:
            image = images[index]
            # forward propagation
            outputs = net(image[None, :, :, :] + universal_v)
            # print(torch.argmax(outputs))
            # if the prediction doesn't equal to the target class
            if torch.argmax(outputs) != target_class:
                # 1. compute the minimal perturbation
                v_total, i_deepfool, perturbed_image = targeted_deepfool(image + universal_v[0].detach(), net, target_class, overshoot, max_iter_deepfool)
                # 2. update the perturbation using the projection function
                # make sure it converges
                if i_deepfool < max_iter_deepfool-1:
                    universal_v = universal_v + torch.from_numpy(v_total).to(device)
                    universal_v = proj_lp(universal_v, xi, p)
    
    return universal_v