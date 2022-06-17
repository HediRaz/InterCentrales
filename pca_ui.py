"""PCA UI."""

import os
import tkinter as tk

import torch
from PIL import Image, ImageTk
from torchvision import transforms


class ImgCanvas(tk.Canvas):
    """Canvas for images."""

    def __init__(self, root, width=256, height=512):
        super().__init__(root, width=width, height=height)
        self.width = width
        self.height = height
        self.create_line(2, 0, 2, height)

        self.pack(side=tk.RIGHT, fill=tk.Y)
        self.original_img = None
        self.edit_img = None

    def draw_image(self, img, original=False):
        """Draw image on canvas."""
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
    """Frame for path input."""

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

        self.pack(side=tk.TOP)


class PCAEditFrame(tk.Frame):
    """Frame for PCA editing."""

    def __init__(self, root, edit_command, nb_layer=16, nb_pca=80,
                 width=700, height=512):
        super().__init__(root, width=width, height=height)
        self.width = width
        self.height = height
        self.nb_layer = nb_layer
        self.nb_pca = nb_pca

        self.edit_command = edit_command

        self.pca = []

        for i in range(self.nb_pca):
            # List of dictionary
            self.pca.append({"canvas": tk.Canvas(self, width=self.width,
                                                 height=150)})
            # define instances
            self.pca[i]["label"] = tk.Label(self.pca[i]["canvas"],
                                            text=f"PCA {i}")
            self.pca[i]["entry notes"] = tk.Entry(self.pca[i]["canvas"],
                                                  width=30)
            self.pca[i]["min var"] = tk.StringVar(self.pca[i]["canvas"],
                                                  value="-2")
            self.pca[i]["max var"] = tk.StringVar(self.pca[i]["canvas"],
                                                  value="2")
            self.pca[i]["min entry"] = tk.Entry(
                self.pca[i]["canvas"], textvariable=self.pca[i]["min var"],
                width=5
                )
            self.pca[i]["max entry"] = tk.Entry(
                self.pca[i]["canvas"], textvariable=self.pca[i]["max var"],
                width=5
                )
            self.pca[i]["scale"] = tk.Scale(
                self.pca[i]["canvas"], orient='horizontal', from_=-2, to=2,
                command=self.edit_command, resolution=0.02, tickinterval=10,
                width=10, length=580
                )
            self.pca[i]["checkboxes name"] = [
                tk.Label(self.pca[i]["canvas"], text=str(j))
                for j in range(self.nb_layer)
                ]
            self.pca[i]["checkboxes var"] = [
                tk.IntVar() for _ in range(self.nb_layer)
                ]
            self.pca[i]["checkboxes"] = [
                tk.Checkbutton(
                    self.pca[i]["canvas"],
                    variable=self.pca[i]["checkboxes var"][j],
                    onvalue=1, offvalue=0, command=self.edit_command,
                    width=1, height=1) for j in range(self.nb_layer)
                ]
            self.pca[i]["proj value"] = [tk.StringVar()
                                         for _ in range(self.nb_layer)]
            self.pca[i]["proj label"] = [
                tk.Entry(self.pca[i]["canvas"],
                         textvariable=self.pca[i]["proj value"][j],
                         width=6)
                for j in range(self.nb_layer)
                ]
            # define comportment
            self.pca[i]["min entry"].bind("<Return>", self.minmax_callback)
            self.pca[i]["max entry"].bind("<Return>", self.minmax_callback)
            # Pack
            self.pca[i]["label"].place(x=5, y=5)
            self.pca[i]["entry notes"].place(x=50, y=5)
            self.pca[i]["min entry"].place(x=5, y=40)
            self.pca[i]["max entry"].place(x=650, y=40)
            self.pca[i]["scale"].place(x=50, y=23)
            for j in range(self.nb_layer):
                self.pca[i]["checkboxes name"][j].place(x=j*43 + 5, y=78)
                self.pca[i]["checkboxes"][j].place(x=j*43, y=93)
                self.pca[i]["proj label"][j].place(x=j*43 + 5, y=120)

            self.pca[i]["canvas"].pack(side=tk.TOP)

    def minmax_callback(self, *arg, **kwargs):
        """Apply the callback for min and max variance."""
        for i in range(self.nb_pca):
            mini = int(self.pca[i]["min var"].get())
            maxi = int(self.pca[i]["max var"].get())
            self.pca[i]["scale"].config(from_=mini, to=maxi,
                                        resolution=(maxi - mini) / 100)

    def get_pca_directions(self):
        """Get PCA directions."""
        strengths = []
        all_indexes = []
        for i in range(self.nb_pca):
            strength = self.pca[i]["scale"].get()
            idx = [self.pca[i]["checkboxes var"][j].get()
                   for j in range(self.nb_layer)]
            strengths.append(strength)
            all_indexes.append(idx)
        return strengths, all_indexes


class PCAEditCanvas(tk.Canvas):
    """Canvas for PCA editing."""

    def __init__(self, root, edit_command, width=700, height=512):
        super().__init__(root, width=width, height=height)
        self.pca_frame = PCAEditFrame(self, edit_command)
        self.scrollbar = tk.Scrollbar(root, orient="vertical",
                                      command=self.yview)
        self.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.pca_frame.bind(
            "<Configure>",
            lambda e: self.configure(
                scrollregion=self.bbox("all")
                )
            )
        self.create_window((0, 0), window=self.pca_frame, anchor="nw")
        self.pack(side=tk.TOP)


class ConfigFrame(tk.Frame):
    """Frame for configuring the PCA."""

    def __init__(self, root, path_callback, edit_command,
                 width=700, height=512):
        super().__init__(root)
        self.width = width
        self.height = height

        self.path_frame = PathFrame(self, path_callback, width=width)
        self.pca_canvas = PCAEditCanvas(self, edit_command)

        self.pack(anchor="nw")


class EditWindow(tk.Tk):
    """Window for edition."""

    def __init__(self, img_transforms, encoder, generator, ganspace_pca,
                 latents_avg):
        super().__init__()
        self.img_transforms = img_transforms
        self.encoder = encoder
        self.generator = generator
        self.nb_pca = 80
        self.nb_layer = 16
        self.ganspace_pca_mean = ganspace_pca["mean"].to("cuda")
        self.ganspace_pca_std = ganspace_pca["std"].to("cuda")
        self.ganspace_pca_comp = ganspace_pca["comp"].to("cuda")
        self.latents_avg = latents_avg.to("cuda")

        # Main window
        self.title("Edit image")
        # Frames
        self.img_canvas = ImgCanvas(self)
        self.config_frame = ConfigFrame(self, self.img_path_callback,
                                        self.infer_image)

        self.edit_latents = None
        self.latents = None

    def img_path_callback(self):
        """Apply the callback for image path."""
        img_path = self.config_frame.path_frame.path.get()
        img_path = os.path.normpath(img_path)
        if not os.path.exists(img_path):
            self.config_frame.path_frame.result.config(
                text="Path does not exist"
                )
            return

        img_pil = Image.open(img_path)
        self.img_canvas.draw_image(img_pil, original=True)
        # run_alignment(img_path)
        img = self.img_transforms(img_pil)
        with torch.no_grad():
            self.latents = self.encoder(img.unsqueeze(0).to("cuda"))[0]
            if self.latents.ndim == 2:
                self.latents = self.latents + self.latents_avg.repeat(
                    self.latents.shape[0], 1, 1
                    )[:, 0, :]
            else:
                self.latents = self.latents + self.latents_avg.repeat(
                    self.latents.shape[0], 1, 1
                    )

        self.infer_image()

    def get_delta(self, pca_idx, strength):
        """Get delta for a given PCA and strength."""
        # pca: ganspace checkpoint. latents: (16, 512) w+
        w_centered = self.latents - self.ganspace_pca_mean
        lat_comp = self.ganspace_pca_comp
        lat_std = self.ganspace_pca_std
        w_coord = torch.sum(
            w_centered[0].reshape(-1)*lat_comp[pca_idx].reshape(-1)
            ) / lat_std[pca_idx]
        delta = (strength - w_coord)*lat_comp[pca_idx]*lat_std[pca_idx]
        return delta

    def get_proj(self):
        """Get projection."""
        for i in range(self.nb_pca):
            for j in range(self.nb_layer):
                proj = torch.dot(self.edit_latents[j].reshape(-1),
                                 self.ganspace_pca_comp[i].reshape(-1))
                self.config_frame.pca_canvas\
                    .pca_frame.pca[i]['proj value'][j].set(
                        f'{proj.item():.0e}'
                        )

    def ganspace_edit(self):
        """Edit latents with GAN-space."""
        edit_latents = torch.clone(self.latents)
        strengths, all_indexes = (self.config_frame.pca_canvas
                                  .pca_frame.get_pca_directions())
        for pca_idx, strength, indexes in zip(range(len(strengths)),
                                              strengths, all_indexes):
            delta = self.get_delta(pca_idx, strength)
            delta_padded = torch.zeros(self.latents.shape).to('cuda')
            for i, ind in enumerate(indexes):
                delta_padded[i] += delta.reshape(-1) * ind
                edit_latents += delta_padded
        self.edit_latents = edit_latents

    def infer_image(self, *args, **kwargs):
        """Infer image."""
        self.ganspace_edit()
        self.get_proj()
        with torch.no_grad():
            edit_image = self.generator([self.edit_latents.unsqueeze(0)],
                                        randomize_noise=False,
                                        input_is_latent=True)[0][0]
        edit_image = (edit_image.detach().cpu().transpose(0, 1)
                      .transpose(1, 2).numpy())
        edit_image = ((edit_image + 1) / 2)
        edit_image[edit_image < 0] = 0
        edit_image[edit_image > 1] = 1
        edit_image = edit_image * 255
        edit_image = edit_image.astype("uint8")
        edit_image = Image.fromarray(edit_image)

        self.img_canvas.draw_image(edit_image, original=False)


if __name__ == "__main__":
    GANSPACE_PCA_PATH = 'encoder4editing/editings/ganspace_pca/ffhq_pca.pt'

    resize_dims = (256, 256)
    img_transforms = transforms.Compose([
        transforms.Resize(resize_dims),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def encoder(img):
        """Return random encodage."""
        return torch.rand((1, 16, 1, 512), device="cuda")

    def generator(latents, randomize_noise, input_is_latent):
        """Return random image."""
        return torch.rand((1, 1, 3, 256, 256), device="cuda")

    ganspace_pca = torch.load(GANSPACE_PCA_PATH)

    app = EditWindow(img_transforms, encoder, generator, ganspace_pca,
                     torch.rand((512,)))
    app.mainloop()
