import torch
import torchvision
import torchvision.transforms as transforms
import os
import random

def get_mnist_set(train_or_test, vtype="bernoulli", path=""):

    tx = [transforms.ToTensor(),
                 ]
    
    if vtype == "bernoulli":
        tx.append(transforms.Lambda(lambda x: 2*torch.bernoulli(x) -1))
    elif vtype == "categorical":
        # tx.append(transforms.Lambda(lambda x: torch.stack([x,x,x])))
        # tx.append(transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02))
        # tx.append(transforms.Lambda(lambda x: torch.clamp(x, 0, 1)))
        tx.append(transforms.Lambda(lambda x: x * 255))
        tx.append(transforms.Lambda(lambda x: torch.round(x)))

    transform = transforms.Compose(tx)

    root_dir = os.path.join(path, "data")
    # Create multiple train_set versions for data augmentation and cat them
    # for i in range(3):
    if train_or_test == "train":
        mnist_set = torchvision.datasets.MNIST(root=root_dir, train=True,
                                               download=True, transform=transform)
    elif train_or_test == "test":
        mnist_set = torchvision.datasets.MNIST(root=root_dir, train=False,
                                               download=True, transform=transform)
    return mnist_set

def get_sum_combos():
    sum_combos = []
    for i in range(10):
        for j in range(0, 10-i):
            sum_combos.append((i,j,i+j))
    return sum_combos
def create_sum_mnist(mnist_set, num_samples=100000):
    class_dict = {
        0:[],
        1:[],
        2:[],
        3:[],
        4:[],
        5:[],
        6:[],
        7:[],
        8:[],
        9:[]
    }
    for image, label in mnist_set:
        class_dict[label].append(image)

    sum_combos = get_sum_combos()
    print(len(sum_combos))
    sum_images = []
    for i in range(num_samples//len(sum_combos)):
        for sum_combo in sum_combos:
            first_digit = random.sample(class_dict[sum_combo[0]], 1)
            second_digit = random.sample(class_dict[sum_combo[1]], 1)
            third_digit = random.sample(class_dict[sum_combo[2]], 1)
            sum_image = merge_digits([first_digit[0], second_digit[0], third_digit[0]])
            sum_images.append(sum_image)
    
    breakpoint()

def merge_digits(digits):
    return torch.cat(digits, dim=2)

mnist_set = get_mnist_set("train")
sum_mnist_set = create_sum_mnist(mnist_set)