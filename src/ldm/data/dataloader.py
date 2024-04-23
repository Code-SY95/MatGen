import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

root_path = '/home/sogang/mnt/db_2/pbr_data/Croped/train'
# data structure : train > category > material name > rotation_angle_crop_num > PBR maps.png
category_list = ['Blends', 'Ceramic', 'Concrete', 'Fabric','Ground',
                 'Marble', 'Metal', 'Misc', 'Plaster', 'Plastic',
                 'Stone', 'Terracotta', 'Wood']
name = 
data_dir = 

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# load data
dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)

# DataLoader를 생성합니다.
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 데이터를 사용하는 예시
for images, labels in dataloader:
    # 여기에 모델 학습 등을 수행하는 코드를 작성합니다.
    pass
