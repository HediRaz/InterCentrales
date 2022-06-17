"""Class for latent editor."""

import torch

from encoder4editing.editings import ganspace, sefa
from encoder4editing.utils.common import tensor2im


class LatentEditor():
    """Latent editor."""

    def __init__(self, stylegan_generator, is_cars=False):
        self.generator = stylegan_generator
        # Since the cars StyleGAN output is 384x512,
        # there is a need to crop the 512x512 output.
        self.is_cars = is_cars

    def apply_ganspace(self, latent, ganspace_pca, edit_directions):
        """Apply gan space manipulation."""
        edit_latents = ganspace.edit(latent, ganspace_pca, edit_directions)
        return self._latents_to_image(edit_latents)

    def apply_interfacegan(self, latent, direction, factor=1,
                           factor_range=None):
        """Apply interface GAN."""
        edit_latents = []
        # Apply a range of editing factors. for example, (-5, 5)
        if factor_range is not None:
            for fac in range(*factor_range):
                edit_latent = latent + fac * direction
                edit_latents.append(edit_latent)
            edit_latents = torch.cat(edit_latents)
        else:
            edit_latents = latent + factor * direction
        return self._latents_to_image(edit_latents)

    def apply_sefa(self, latent, indices=(2, 3, 4, 5), **kwargs):
        """Apply SEFA."""
        edit_latents = sefa.edit(self.generator, latent, indices, **kwargs)
        return self._latents_to_image(edit_latents)

    # Currently, in order to apply StyleFlow editings, one should
    # run inference, save the latent codes and load them form the
    # official StyleFlow repository.

    def _latents_to_image(self, latents):
        with torch.no_grad():
            images, _ = self.generator([latents], randomize_noise=False,
                                       input_is_latent=True)
            if self.is_cars:
                images = images[:, :, 64:448, :]  # 512x512 -> 384x512
        horizontal_concat_image = torch.cat(list(images), 2)
        final_image = tensor2im(horizontal_concat_image)
        return final_image
