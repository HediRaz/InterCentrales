import os
import sys
sys.path.append(os.path.join(os.path.realpath(os.curdir), "encoder4editing"))
import argparse
from argparse import Namespace
import torch
import torchvision.transforms as transforms

from encoder4editing.models.psp import pSp

from pca_ui import EditWindow
from celeba_ui import CelebaEditWindow


MODEL_PATH = "encoder4editing/pretrained_models/e4e_ffhq_encode.pt"
resize_dims = (256, 256)
# Setup required image transformations
img_transforms = transforms.Compose([
    transforms.Resize(resize_dims),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


ckpt = torch.load(MODEL_PATH, map_location='cpu')
for u in list(ckpt.keys()):
    if "opts" not in u:
        ckpt.pop(u)
opts = ckpt['opts']
# pprint.pprint(opts)  # Display full options used
# update the training options
opts['checkpoint_path'] = MODEL_PATH
opts= Namespace(**opts)
net = pSp(opts)
# net.eval()
# net.cuda()
print('Model successfully loaded!')


resize_dims = (256, 256)
img_transforms = transforms.Compose([
    transforms.Resize(resize_dims),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
encoder = net.encoder
# encoder = lambda img: torch.rand((1, 16, 1, 512), device="cuda")
# def generator(latents, randomize_noise, input_is_latent):
#     return torch.rand((1, 1, 3, 256, 256), device="cuda")
generator = net.decoder


parser = argparse.ArgumentParser()
parser.add_argument("--pca", action="store_true", help="Run UI that uses PCA vectors")
parser.add_argument("--custom", action="store_true", help="Run UI that uses custom vectors")
args = parser.parse_args()

if args.pca:
    ganspace_pca = torch.load('encoder4editing/editings/ganspace_pca/ffhq_pca.pt')
    latents_avg = net.latent_avg


    app = EditWindow(img_transforms, encoder, generator, ganspace_pca, latents_avg)

elif args.custom:
    app = CelebaEditWindow(img_transforms, encoder, generator, net.latent_avg)

else:
    print("Please specify which UI to run")
    sys.exit(1)

app.mainloop()
