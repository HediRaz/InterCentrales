import os
import pandas as pd
import numpy as np
import tkinter as tk
import torch

from PIL import Image, ImageTk



def load_latents(filename):
    return np.load(filename)


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
    def __init__(self, root, entry_callback, text="Enter image path", width=512, height=80):
        super().__init__(root, width=width, height=height)
        self.width = width
        self.height = height
        
        self.entry_callback = entry_callback

        self.path = tk.StringVar(value=text)
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
                                  resolution=0.02, tickinterval=10, width=10, length=500)
        self.scale_bar.pack(side=tk.TOP)


class CelebaEditWindow(tk.Tk):
    def __init__(self, img_transforms, encoder, generator, latents_avg):
        super().__init__()
        self.img_transforms = img_transforms
        self.encoder = encoder
        self.generator = generator
        self.latents_avg = latents_avg

        self.title("Edit window")
        self.geometry("768x512")

        self.img_canvas = ImgCanvas(self, width=256, height=512)
        self.img_path_frame = PathFrame(self, lambda: self.img_path_callback(), width=512)
        self.latents_path_frame = PathFrame(self, lambda: self.latents_path_callback(), text="Enter latents path", width=512)
        self.delta_scale = DeltaScaleFrame(self, lambda e: self.infer_image())

        self.img_canvas.place(x=512, y=0)
        self.img_path_frame.place(x=0, y=0)
        self.latents_path_frame.place(x=0, y=50)
        self.delta_scale.place(x=0, y=200)
    
    def img_path_callback(self):
        img_path = self.img_path_frame.path.get()
        img_path = os.path.normpath(img_path)
        if not os.path.exists(img_path):
            self.img_path_frame.result.config(text="Path does not exist")
            return
        else:
            img_pil = Image.open(img_path)
            self.img_canvas.draw_image(img_pil, original=True)
            # run_alignment(img_path)
            img = self.img_transforms(img_pil)
            with torch.no_grad():
                self.latents = self.encoder(img.unsqueeze(0).to("cuda").float())[0]
                if self.latents.ndim == 2:
                    self.latents = self.latents + self.latents_avg.repeat(self.latents.shape[0], 1, 1)[:, 0, :]
                else:
                    self.latents = self.latents + self.latents_avg.repeat(self.latents.shape[0], 1, 1)
            self.img_path_frame.result.config(text="Image loaded !")

    def latents_path_callback(self):
        latents_path = self.latents_path_frame.path.get()
        latents_path = os.path.normpath(latents_path)
        if not os.path.exists(latents_path):
            self.latents_path_frame.result.config(text="Path does not exist")
            return
        else:
            direction = load_latents(latents_path)
            self.direction = torch.tensor(direction, device="cuda").float()
            self.latents_path_frame.result.config(text="Latents loaded !")

    def infer_image(self, *args, **kwargs):
        coef = self.delta_scale.scale_bar.get()
        self.edit_latents = self.latents + coef * self.direction
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

    def encoder(img):
        return torch.rand((1, 16, 1, 512), device="cuda")

    def generator(latents, randomize_noise, input_is_latent):
        return torch.rand((1, 1, 3, 256, 256), device="cuda")

    app = CelebaEditWindow(img_transforms, encoder, generator, torch.rand((512,), device="cuda"))
    app.mainloop()
