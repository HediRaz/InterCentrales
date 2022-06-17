"""Script to encode all CelebA dataset with the e4e encoder.

All encodings are saved in ./latent_celeba
"""

import os
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from encoder4editing.models.psp import pSp_encoder

if __name__ == '__main__':
    MODEL_PATH = "encoder4editing/pretrained_models/e4e_ffhq_encode.pt"

    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    opts = ckpt['opts']
    for u in list(ckpt.keys()):
        if "opts" not in u:
            ckpt.pop(u)
    opts['checkpoint_path'] = MODEL_PATH
    opts = Namespace(**opts)

    net = pSp_encoder(opts)
    net.eval()
    net.cuda()

    img_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    df = pd.read_csv("list_attr_celeba.csv", header=0)
    df["idx"] = pd.to_numeric(df["idx"])
    KEY = "idx"
    df.set_index(KEY, inplace=True)
    ATTRIBUTES = df.columns[1:]
    print(ATTRIBUTES)
    for att in ATTRIBUTES:
        df[att] = df[att] == 1

    for i in tqdm(range(len(os.listdir("latents_celeba")),
                        len(os.listdir("img_align_celeba")))):
        img_filename = df.loc[i]["filename"]
        img_path = os.path.join("img_align_celeba", img_filename)
        with Image.open(img_path) as img:
            img = img_transforms(img)
            with torch.no_grad():
                latents = net(img.unsqueeze(0).to("cuda"))[0].cpu().numpy()
            np.save(os.path.join("latents_celeba", img_filename[:-3]+"npy"),
                    latents)
