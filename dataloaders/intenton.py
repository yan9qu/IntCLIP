import xml.dom.minidom
from randaugment import RandAugment
from dataloaders.helper import CutoutPIL
import torchvision.transforms as transforms
import os
import torch
from PIL import Image
import torch.utils.data as data
import numpy as np
import sys
import json
sys.path.insert(0, './')
sys.path.insert(0, '../')


class intentonomy(data.Dataset):
    def __init__(self, root, data_split, img_size=224, p=1, annFile="", label_mask=None, partial=1+1e-6):
        self.classnames = ['Attractive, fastidious',
                           'Beat other, compete',
                           'Communicate, interpersonally effective',
                           'Creative, unique',
                           'Curious, adventurous, exciting life, exploration',
                           'Easy life, financial freedom',
                           'Enjoy life',
                           'Fine design learn arch',
                           'Fine design learn art',
                           'Fine design learn culture',
                           'Good parent, emotionally close to child',
                           'Happy, happiness',
                           'Hard working, mastery and perseverance',
                           'Harmony',
                           'Health',
                           'In love, emotional intimacy',
                           'In love with animals',
                           'Inspire others',
                           'Manageable, make plans, organize',
                           'Natural beauty',
                           'Passionate about something, persue ideals and passions',
                           'Playful, enjoy life',
                           'Share feelings with others, interpersonally effective',
                           'Social life, friendship',
                           'Success in occupation, having a good job',
                           'Teach others',
                           'Things in order, neat, tidy',
                           'Work I enjoy']
        self.classnames = [
'Being attractive and fastidious in appearance and actions',
'Beating others and competing successfully',
'Communicating effectively and being interpersonally skilled',
'Being creative and unique in endeavors and self-expression',
'Displaying curiosity and adventurousness, seeking an exciting life of exploration',
'Enjoying an easy life with financial freedom',
'Truly enjoying life to its fullest',
'Learning about and appreciating fine architectural design',
'Learning about and appreciating fine art in various forms',
'Learning about and appreciating different cultures',
'Being a good parent with an emotionally close relationship to children',
'Seeking and experiencing happiness in daily life',
'Demonstrating hard work, mastery, and perseverance',
'Creating and maintaining harmony in life and surroundings',
'Prioritizing and maintaining good health in all aspects',
'Experiencing deep love and emotional intimacy',
'Showing strong love and connection with animals',
'Inspiring others through actions and words',
'Making life manageable through planning and organization',
'Appreciating and seeking out natural beauty',
'Pursuing ideals and passions with enthusiasm',
'Being playful and enjoying moments',
'Sharing feelings and being interpersonally effective',
'Valuing social life and nurturing meaningful friendships',
'Achieving success in occupation and having a fulfilling job',
'Teaching others and sharing knowledge effectively',
'Keeping things in order and maintaining a neat environment',
'Engaging in enjoyable and fulfilling work'
]
        root = '/data0/home/yangqu/.allfiles/IntCLIP/dataset/Intentonomy'
        train_path = root+'/annotations/train_label_vectors_intentonomy2020.npy'
        val_path = root+'/annotations/val_label_vectors_intentonomy2020.npy'
        test_path = root+'/annotations/test_label_vectors_intentonomy2020.npy'
        inte_train_anno_path = root+'/annotations/intentonomy_train2020.json'
        inte_val_anno_path = root+'/annotations/intentonomy_val2020.json'
        inte_test_anno_path = root+'/annotations/intentonomy_test2020.json'
        self.image_dir = root
        if data_split == 'trainval':
            self.labels_path = train_path
            self.anno_path = inte_train_anno_path
        elif data_split == 'val':
            self.labels_path = val_path
            self.anno_path = inte_val_anno_path
        elif data_split == 'test':
            self.labels_path = test_path
            self.anno_path = inte_test_anno_path

        self.labels = []
        if os.path.exists(self.labels_path):
            self.labels = np.load(self.labels_path).astype(np.float64)
            self.labels = (self.labels > 0).astype(np.float64)

        train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(img_size)
            transforms.Resize((img_size, img_size)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            # transforms.CenterCrop(img_size),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        if data_split == 'train':
            self.input_transform = train_transform
        else:
            self.input_transform = test_transform

    def _load_image(self, index):
        image_path = self._get_image_path(index)

        return Image.open(image_path).convert("RGB")

    def _get_image_path(self, index):
        with open(self.anno_path, 'r') as f:
            annos_dict = json.load(f)
            annos_i = annos_dict['annotations'][index]
            id = annos_i['id']
            if id != index:
                raise ValueError('id not equal to index')
            img_id_i = annos_i['image_id']

            imgs = annos_dict['images']

            for img in imgs:
                if img['id'] == img_id_i:
                    image_file_name = img['filename']
                    image_file_path = os.path.join(
                        self.image_dir, image_file_name)
                    break

        return image_file_path

    def __getitem__(self, index):
        input = self._load_image(index)
        if self.input_transform:
            input = self.input_transform(input)
        label = self.labels[index]
        # label to one-hot
        target = torch.zeros(1, 28)
        for i in range(28):
            if label[i] == 1:
                target[0][i] = 1
        return input, target

    def __len__(self):
        return self.labels.shape[0]

    def name(self):
        return 'Intentonomy'
