import os
import tkinter as tk

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageTk

df = pd.read_csv("list_attr_celeba.csv", header=0)
df["idx"] = pd.to_numeric(df["idx"])
KEY = "idx"
df.set_index(KEY, inplace=True)
ATTRIBUTES = df.columns[1:]
print(ATTRIBUTES)
for att in ATTRIBUTES:
    df[att] = df[att] == 1


def get_att_df(df, att_name, yes=True):
    if yes:
        res = df[df[att_name]]
    else:
        res = df[df[att_name] == False]
    return res


def load_latents(filename, latents_folder="latents_celeba"):
    latents_path = os.path.join(latents_folder, filename[:-3]+"npy")
    return np.load(latents_path)


def get_latents_mean(df, latents_folder="latents_celeba", **kwargs):
    for att in ATTRIBUTES:
        if att in kwargs:
            df = get_att_df(df, att, kwargs[att])

    for i, f in enumerate(df["filename"]):
        if i == 0:
            m = load_latents(f, latents_folder=latents_folder) / len(df["filename"])
        else:
            m += load_latents(f, latents_folder=latents_folder) / len(df["filename"])
    return m


class ImgCanvas(tk.Canvas):
    def __init__(self, root, width=256, height=512):
        super().__init__(root, width=width, height=height)
        self.width = width
        self.height = height
        self.create_line(2, 0, 2, height)

    def draw_image(self, img, original=False):
        img = ImageTk.PhotoImage(img.resize((self.width, self.height//2)))
        if original:
            y = self.height // 2
            self.original_img = img
            self.create_image(0, y, anchor="nw", image=self.original_img)
        else:
            y = 0
            self.edit_img = img
            self.create_image(0, y, anchor="nw", image=self.edit_img)


class PathFrame(tk.Frame):
    def __init__(self, root, entry_callback, width=700, height=50):
        super().__init__(root, width=width, height=height)
        self.width = width
        self.height = height
        
        self.entry_callback = entry_callback

        self.path = tk.StringVar(value="Enter image path")
        self.entry = tk.Entry(self, textvariable=self.path, width=50)
        self.entry.bind("<Return>", lambda e: self.entry_callback())
        self.entry.place(x=5, y=5)

        self.button = tk.Button(self, text="Load", command=self.entry_callback)
        self.button.place(x=320, y=1)

        self.result = tk.Label(self)
        self.result.place(x=5, y=25)


class DeltaScaleFrame(tk.Frame):
    def __init__(self, root, edit_command):
        super().__init__(root)
        self.edit_command = edit_command
        self.scale_bar = tk.Scale(self, orient='horizontal', from_=-2, to=2, command=self.edit_command,
                                  resolution=0.02, tickinterval=10, width=10, length=580)
        self.scale_bar.pack(side=tk.TOP)


class RadioButtonFrame(tk.Frame):
    def __init__(self, root, width=600, height=840):
        super().__init__(root, width=width, height=height)
        self.att_frames = []
        for att in ATTRIBUTES:
            self.att_frames.append({"frame": tk.Frame(self, width=600, height=20)})
            self.att_frames[-1]["label"] = tk.Label(self.att_frames[-1]["frame"], text=att, height=1)
            self.att_frames[-1]["var1"] = tk.IntVar(value=-1)
            self.att_frames[-1]["bt11"] = tk.Radiobutton(self.att_frames[-1]["frame"], value=-1, variable=self.att_frames[-1]["var1"], width=1)
            self.att_frames[-1]["bt12"] = tk.Radiobutton(self.att_frames[-1]["frame"], value=1, variable=self.att_frames[-1]["var1"], width=1)
            self.att_frames[-1]["bt13"] = tk.Radiobutton(self.att_frames[-1]["frame"], value=0, variable=self.att_frames[-1]["var1"], width=1)
            self.att_frames[-1]["var2"] = tk.IntVar(value=-1)
            self.att_frames[-1]["bt21"] = tk.Radiobutton(self.att_frames[-1]["frame"], value=-1, variable=self.att_frames[-1]["var2"], width=1)
            self.att_frames[-1]["bt22"] = tk.Radiobutton(self.att_frames[-1]["frame"], value=1, variable=self.att_frames[-1]["var2"], width=1)
            self.att_frames[-1]["bt23"] = tk.Radiobutton(self.att_frames[-1]["frame"], value=0, variable=self.att_frames[-1]["var2"], width=1)
            self.att_frames[-1]["label"].place(x=0, y=0)
            self.att_frames[-1]["bt11"].place(x=150, y=0)
            self.att_frames[-1]["bt12"].place(x=180, y=0)
            self.att_frames[-1]["bt13"].place(x=210, y=0)
            self.att_frames[-1]["bt21"].place(x=280, y=0)
            self.att_frames[-1]["bt22"].place(x=310, y=0)
            self.att_frames[-1]["bt23"].place(x=340, y=0)
            self.att_frames[-1]["frame"].pack(side=tk.TOP)
    
    def get_dict(self):
        self.args1 = dict()
        self.args2 = dict()
        for i, att in enumerate(ATTRIBUTES):
            if self.att_frames[i]["var1"] == 1:
                self.args1[att] = True
            elif self.att_frames[i]["var1"] == 0:
                self.args1[att] = False
            if self.att_frames[i]["var2"] == 1:
                self.args2[att] = True
            elif self.att_frames[i]["var2"] == 0:
                self.args2[att] = False
        return self.args1, self.args2


class CelebaConfigFrame(tk.Frame):
    def __init__(self, root, width=600, height=870):
        super().__init__(root, width=width, height=height)
        self.delta_scale = DeltaScaleFrame(self, lambda x: 0)
        self.radio_buttons = RadioButtonFrame(self)

        # Place
        self.delta_scale.place(x=0, y=0)
        self.radio_buttons.place(x=0, y=50)


class CelebaEditWindow(tk.Tk):
    def __init__(self, df, img_transforms, encoder, generator, ganspace_pca, latents_avg, latents_folder="celeba_latents"):
        super().__init__()
        self.df = df
        self.latents_folder = latents_folder
        self.img_transforms = img_transforms
        self.encoder = encoder
        self.generator = generator
        self.ganspace_pca = ganspace_pca
        self.latents_avg = latents_avg

        self.title("Edit window")
        self.geometry("856x900")

        self.img_canvas = ImgCanvas(self, width=256, height=512)
        self.path_frame = PathFrame(self, lambda: self.img_path_callback())
        self.config_frame = CelebaConfigFrame(self)

        self.img_canvas.place(x=600, y=0)
        self.path_frame.place(x=0, y=0)
        self.config_frame.place(x=0, y=30)
    
    def img_path_callback(self):
        img_path = self.path_frame.path.get()
        img_path = os.path.normpath(img_path)
        if not os.path.exists(img_path):
            self.path_frame.result.config(text="Path does not exist")
            return
        else:
            img_pil = Image.open(img_path)
            self.img_canvas.draw_image(img_pil, original=True)
            # run_alignment(img_path)
            img = self.img_transforms(img_pil)
            with torch.no_grad():
                self.latents = self.encoder(img.unsqueeze(0).to("cuda"))[0]
                if self.latents.ndim == 2:
                    self.latents = self.latents + self.latents_avg.repeat(self.latents.shape[0], 1, 1)[:, 0, :]
                else:
                    self.latents = self.latents + self.latents_avg.repeat(self.latents.shape[0], 1, 1)

        self.infer_image()

    def infer_image(self, *args, **kwargs):
        edit_args1, edit_args2 = self.config_frame.radio_buttons.get_dict()
        delta_latents = get_latents_mean(self.df, **edit_args1) + get_latents_mean(self.df, **edit_args2)
        delta_latents = torch.tensor(delta_latents, device="cuda")
        coef = self.config_frame.delta_scale.get()
        self.edit_latents = self.latents + coef * delta_latents
        with torch.no_grad():
            edit_image = self.generator([self.edit_latents.unsqueeze(0)], randomize_noise=False, input_is_latent=True)[0][0]
        edit_image = edit_image.detach().cpu().transpose(0, 1).transpose(1, 2).numpy()
        edit_image = ((edit_image + 1) / 2)
        edit_image[edit_image < 0] = 0
        edit_image[edit_image > 1] = 1
        edit_image = edit_image * 255
        edit_image = edit_image.astype("uint8")
        edit_image = Image.fromarray(edit_image)

        self.img_canvas.draw_image(edit_image, original=False)
        

if __name__ == "__main__":
    from torchvision import transforms
    resize_dims = (256, 256)
    img_transforms = transforms.Compose([
        transforms.Resize(resize_dims),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # encoder = net.encoder

    def encoder(img):
        return torch.rand((1, 16, 1, 512), device="cuda")

    def generator(latents, randomize_noise, input_is_latent):
        return torch.rand((1, 1, 3, 256, 256), device="cuda")
    # generator = net.decoder
    ganspace_pca = torch.load('encoder4editing/editings/ganspace_pca/ffhq_pca.pt')

    app = CelebaEditWindow(df, img_transforms, encoder, generator, ganspace_pca, torch.rand((512,), device="cuda"))
    app.mainloop()
