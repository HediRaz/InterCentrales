"""Celeba UI."""

import os
import tkinter as tk
from functools import partial

import numpy as np
import torch
from PIL import Image, ImageTk


def load_latents(filename):
    """Load latents from file."""
    return np.load(filename)


class ImgCanvas(tk.Canvas):
    """Image canvas."""

    def __init__(self, root, width=256, height=512):
        super().__init__(root, width=width, height=height)
        self.width = width
        self.height = height
        self.create_line(2, 0, 2, height)
        self.original_img = None
        self.edit_img = None

    def draw_image(self, img, original=False):
        """Draw image."""
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
    """Path frame."""

    def __init__(self, root, entry_callback, text="Enter image path",
                 width=512, height=80):
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
        self.result.place(x=380, y=5)


class DeltaScaleFrame(tk.Frame):
    """Delta scale frame."""

    def __init__(self, root, edit_command):
        super().__init__(root)
        self.edit_command = edit_command
        self.scale_bar = tk.Scale(self, orient='horizontal', from_=-2, to=2,
                                  command=self.edit_command, resolution=0.02,
                                  tickinterval=1, width=10, length=500)
        self.scale_bar.pack(side=tk.TOP)


class CelebaEditWindow(tk.Tk):
    """Celeba edit window."""

    def __init__(self, img_transforms, encoder, generator, latents_avg):
        super().__init__()
        self.img_transforms = img_transforms
        self.encoder = encoder
        self.generator = generator
        self.latents_avg = latents_avg
        self.direction = {}

        self.title("Edit window")
        self.geometry("1536x600")

        self.img_canvas = []
        self.img_path_frame = []
        self.img_proj_label = []
        for i in range(4):
            self.img_canvas.append(ImgCanvas(self, width=256, height=512))
            self.img_path_frame.append(PathFrame(self, partial(
                self.img_path_callback, i=i
                ), width=512))
            self.img_proj_label.append(tk.Label(self, text="0"))

        self.latents_path_frame = []
        self.delta_scale = []
        for i in range(5):
            self.latents_path_frame.append(
                PathFrame(self, partial(self.latents_path_callback, i=i),
                          text="Enter latents path", width=512)
                )
            self.delta_scale.append(DeltaScaleFrame(self, self.infer_image))

        for i in range(4):
            self.img_canvas[i].place(x=512 + 256*i, y=0)
            self.img_path_frame[i].place(x=0, y=30*i)
            self.img_proj_label[i].place(x=600 + 256*i, y=550)
        for i in range(5):
            self.latents_path_frame[i].place(x=0, y=140 + 30*i)
            self.delta_scale[i].place(x=0, y=340 + 45*i)

        self.latents = [None for _ in range(4)]
        self.edit_latents = [None for _ in range(4)]
        self.direction = [torch.zeros((18, 512), dtype=torch.float32,
                                      device="cuda") for _ in range(5)]

    def img_path_callback(self, i):
        """Image path callback."""
        img_path = self.img_path_frame[i].path.get()
        img_path = os.path.normpath(img_path)
        if not os.path.exists(img_path):
            self.img_path_frame[i].result.config(text="Path does not exist")
            return

        img_pil = Image.open(img_path)
        self.img_canvas[i].draw_image(img_pil, original=True)
        # run_alignment(img_path)
        img = self.img_transforms(img_pil)
        with torch.no_grad():
            self.latents[i] = self.encoder(
                img.unsqueeze(0).to("cuda").float()
                )[0]
            self.latents[i] = self.latents[i] + self.latents_avg.repeat(
                self.latents[i].shape[0], 1, 1
                )[:, 0, :]
        self.img_path_frame[i].result.config(text="Image loaded !")

    def latents_path_callback(self, i):
        """Latents path callback."""
        latents_path = self.latents_path_frame[i].path.get()
        latents_path = os.path.normpath(latents_path)
        if not os.path.exists(latents_path):
            self.latents_path_frame[i].result.config(
                text="Path does not exist"
                )
            return
        direction = load_latents(latents_path)
        self.direction[i] = torch.tensor(direction, device="cuda").float()
        self.latents_path_frame[i].result.config(text="Latents loaded !")

    def get_proj(self, j):
        """Get projection."""
        with torch.no_grad():
            proj = torch.mean(torch.square(self.direction[0]
                                           - self.edit_latents[j]))
            proj = proj.cpu().item()
        self.img_proj_label[j].config(text=str(proj))

    def infer_image(self, *args, **kwargs):
        """Infer image."""
        coef = self.delta_scale[0].scale_bar.get()
        for j in range(4):
            if self.latents[j] is not None:
                self.edit_latents[j] = (self.latents[j]
                                        + coef*self.direction[0])
                for i in range(1, 5):
                    if (len(self.latents_path_frame[i].path.get()) > 0
                        and self.latents_path_frame[i].path.get()
                            != "Enter latents path"):
                        self.edit_latents[j] += (
                            self.delta_scale[i].scale_bar.get()
                            * coef
                            * self.direction[i]
                            )
                self.get_proj(j)
                with torch.no_grad():
                    edit_image = self.generator(
                        [self.edit_latents[j].unsqueeze(0)],
                        randomize_noise=False,
                        input_is_latent=True
                        )[0][0]
                edit_image = (edit_image.detach().cpu().transpose(0, 1)
                              .transpose(1, 2).numpy())
                edit_image = ((edit_image + 1) / 2)
                edit_image[edit_image < 0] = 0
                edit_image[edit_image > 1] = 1
                edit_image = edit_image * 255
                edit_image = edit_image.astype("uint8")
                edit_image = Image.fromarray(edit_image)

                self.img_canvas[j].draw_image(edit_image, original=False)


if __name__ == "__main__":
    from torchvision import transforms
    resize_dims = (256, 256)
    img_transforms = transforms.Compose([
        transforms.Resize(resize_dims),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def encoder(img):
        """Wrap function for encoder."""
        return torch.rand((1, 18, 1, 512), device="cuda")

    def generator(latents, randomize_noise, input_is_latent):
        """Wrap function for generator."""
        return torch.rand((1, 1, 3, 256, 256), device="cuda")

    app = CelebaEditWindow(img_transforms, encoder, generator,
                           torch.rand((512,), device="cuda"))
    app.mainloop()
