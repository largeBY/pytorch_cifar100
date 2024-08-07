import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np


# 计算 VOCSegmentation 数据集的均值和标准差
def calculate_mean_std_voc(root='./data', image_set='train'):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.VOCDetection(root=root, year="2012", image_set=image_set, download=True, transform=transform)

    # 计算均值和标准差
    mean = np.zeros(3)
    std = np.zeros(3)
    total_pixels = 0

    for img, _ in dataset:
        mean += np.mean(img.numpy(), axis=(1, 2))
        std += np.std(img.numpy(), axis=(1, 2))
        total_pixels += img.size(1) * img.size(2)

    mean /= len(dataset)
    std /= len(dataset)

    return mean, std

# 获取 VOCSegmentation 数据集的均值和标准差
mean_voc, std_voc = calculate_mean_std_voc()

print(f"VOCDetection 数据集的均值：{mean_voc}")
print(f"VOCDetection 数据集的标准差：{std_voc}")
