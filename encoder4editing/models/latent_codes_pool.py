"""Latent codes pooling."""

import random

import torch


class LatentCodesPool:
    """Latent codes pooling.

    This class implements latent codes buffer that stores previously
    generated w latent codes. This buffer enables us to update
    discriminators using a history of generated w's rather than
    the ones produced by the latest encoder.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class.

        Parameters
        ----------
        pool_size : int
            The size of image buffer, if pool_size=0,
            no buffer will be created.
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            # Create an empty pool of latents
            self.num_latents = 0
            self.latents = []

    def query(self, latents):
        """Return w's from the pool.

        Parameters
        ----------
        latents : torch.tensor
            The latest generated w's from the generator.

        Returns
        -------
        return_latents : torch.tensor
            Latents from the buffer.
            By 50%, the buffer will return input latents.
            By 50%, the buffer will return latents previously stored
            in the buffer, and insert the current latents to the buffer.
        """
        if self.pool_size == 0:
            # If the buffer size is 0, do nothing
            return latents

        return_latents = []
        # latents.shape: (batch, 512) or (batch, n_latent, 512)
        for latent in latents:
            if latent.ndim == 2:
                # Apply a random latent index as a candidate
                i = random.randint(0, len(latent) - 1)
                latent = latent[i]
            self.handle_latent(latent, return_latents)
        # Collect all the images and return
        return_latents = torch.stack(return_latents, 0)
        return return_latents

    def handle_latent(self, latent, return_latents):
        """Handle a latent."""
        # If the buffer is not full; keep inserting current
        # codes to the buffer
        if self.num_latents < self.pool_size:
            self.num_latents = self.num_latents + 1
            self.latents.append(latent)
            return_latents.append(latent)
        else:
            proba = random.uniform(0, 1)
            # By 50% chance, the buffer will return a previously stored
            # latent code, and insert the current code into the buffer
            if proba > 0.5:
                # (NOTE: randint is inclusive)
                random_id = random.randint(0, self.pool_size - 1)
                tmp = self.ws[random_id].clone()
                self.ws[random_id] = latent
                return_latents.append(tmp)
            else:
                # By another 50% chance, the buffer will return the
                # current image
                return_latents.append(latent)
