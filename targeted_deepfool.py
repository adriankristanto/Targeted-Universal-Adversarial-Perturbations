# reference: https://github.com/LTS4/DeepFool
import torch
import copy
from torch.autograd.gradcheck import zero_gradients
import numpy as np
# import torchvision
from tqdm import tqdm

def targeted_deepfool(image, net, target_class, overshoot=0.02, max_iter=50):
    """
        image: a single input image of shape H * W * C
        net: classifier without the final softmax activation
        target_class: the target label that we want the network to misclassify the image into
        overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02)
        max_iter: maximum number of iterations for deepfool (default = 50)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get the clean prediction of the classifier
    # make sure the shape of the image contains the batch size dimension
    outputs = net(image[None, :, :, :].requires_grad_(True)).data.cpu().numpy().flatten()

    # get the clean label
    clean_label = np.argmax(outputs)

    # initialise the perturbed image with the original image
    perturbed_image = copy.deepcopy(image)

    # initialise the total perturbation
    v_total = np.zeros(image.shape)

    # obtain the prediction on the clean image and the target
    x = perturbed_image[None, :, :, :].requires_grad_(True)
    outputs = net(x)
    current_prediction = clean_label

    wrapped = tqdm(total=max_iter)

    i = 0

    while current_prediction != target_class and i < max_iter:

        # gradient computation on clean prediction
        # set retain_graph=True to make sure that the graph doesn't get cleaned 
        # as we need it to compute the gradient for the target prediction
        outputs[0, clean_label].backward(retain_graph=True)
        clean_gradient = x.grad.data.cpu().numpy().copy()
        # don't forget to zero the gradient
        zero_gradients(x)
        # compute target gradient
        outputs[0, target_class].backward(retain_graph=True)
        target_gradient = x.grad.data.cpu().numpy().copy()
        
        w = target_gradient - clean_gradient
        f = (outputs[0, target_class] - outputs[0, clean_label]).data.cpu().numpy()

        perturbation = abs(f) / np.linalg.norm(w.flatten())

        wrapped.set_description(f"perturbation: {perturbation:.5f}")
        wrapped.update(1)

        # compute vi
        vi = (perturbation+1e-4) * w / np.linalg.norm(w)
        
        v_total = np.float32(v_total + vi)

        perturbed_image = image + (1+overshoot) * torch.from_numpy(v_total).to(device)

        x = perturbed_image.requires_grad_(True)
        outputs = net(x)
        current_prediction = np.argmax(outputs.data.cpu().numpy().flatten())

        i += 1

    # torchvision.utils.save_image(perturbed_image, 'backdoored.png')
    return perturbed_image