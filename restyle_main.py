import os
import sys

sys.path.append(os.path.join(os.path.realpath(os.curdir), "restyle_encoder"))
from argparse import Namespace

import torch
import torchvision.transforms as transforms

from celeba_ui import CelebaEditWindow
from pca_ui import EditWindow
from restyle_encoder.models.psp import pSp

MODEL_PATH = "restyle_encoder/pretrained_models/restyle_psp_ffhq_encode.pt"
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
avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]

avg_image = avg_image.to('cuda').float().detach()

def encoder(img):
    return net.encoder(torch.cat([img, avg_image], dim=1))
# encoder = lambda img: torch.rand((1, 16, 1, 512), device="cuda")
# def generator(latents, randomize_noise, input_is_latent):
#     return torch.rand((1, 1, 3, 256, 256), device="cuda")
generator = net.decoder

ganspace_pca = torch.load('encoder4editing/editings/ganspace_pca/ffhq_pca.pt')
latents_avg = net.latent_avg


app = EditWindow(img_transforms, encoder, generator, ganspace_pca, latents_avg)
app.mainloop()

# app = CelebaEditWindow(img_transforms, encoder, generator, net.latent_avg)
# app.mainloop()

