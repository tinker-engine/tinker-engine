import torch
import pytest

from VAAL.sampler import AdversarySampler
from VAAL.model import VAE
from VAAL.model import Discriminator
from ..dataset import JPLDataset
from .test_problem import get_problem


@pytest.mark.parametrize("latent_dim,budget",
                         [(32, 10),
                          (64, 5)]
)
def test_adversary_sampler_sample_method(requests_mock, latent_dim, budget):
    problem = get_problem(requests_mock)
    dataset = JPLDataset(problem)
    unlabeled_sampler = torch.utils.data.sampler.SubsetRandomSampler(
        dataset.get_unlabeled_indices()
    )
    unlabeled_dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=unlabeled_sampler,
        batch_size=min(len(dataset.get_unlabeled_indices()),
                       128),
        num_workers=8,
        drop_last=False,
    )
    adv_sampler = AdversarySampler(budget)
    adv_sampler.sample(VAE(latent_dim), Discriminator(latent_dim), unlabeled_dataloader, False)
