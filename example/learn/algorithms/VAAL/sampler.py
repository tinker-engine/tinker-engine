import torch
import ubelt as ub
import numpy as np


class AdversarySampler:
    def __init__(self, budget):
        self.budget = budget

    def sample(self, vae, discriminator, data, cuda):
        all_preds = []
        all_indices = []

        for images, _, indices in ub.ProgIter(data, desc="VAE sampling"):
            if cuda:
                images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk
        all_preds *= -1

        # Select the points which the discriminator things are the most
        # likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices
