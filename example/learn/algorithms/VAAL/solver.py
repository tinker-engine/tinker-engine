import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.metrics import accuracy_score

import ubelt as ub


class Solver:
    def __init__(self, args, test_dataloader):
        self.args = args
        self.test_dataloader = test_dataloader

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.sampler = None

    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img

    def train(
        self, querry_dataloader, task_model, vae, discriminator, unlabeled_dataloader,
    ):
        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)

        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)

        if task_model:
            optim_task_model = optim.Adam(task_model.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

        vae.train()
        discriminator.train()
        if task_model:
            task_model.train()

        if self.args["cuda"]:
            vae = vae.cuda()
            discriminator = discriminator.cuda()
            if task_model:
                task_model = task_model.cuda()

        change_lr_iter = int(self.args["train_iterations"]) // 25

        for iter_count in ub.ProgIter(
            range(int(self.args["train_iterations"])), desc="Training on Data"
        ):
            if iter_count is not 0 and iter_count % change_lr_iter == 0:
                for param in optim_vae.param_groups:
                    param["lr"] = param["lr"] * 0.9
                if task_model:
                    for param in optim_task_model.param_groups:
                        param["lr"] = param["lr"] * 0.9

                for param in optim_discriminator.param_groups:
                    param["lr"] = param["lr"] * 0.9
            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)

            if self.args["cuda"]:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

            if task_model:
                # task_model step
                preds = np.squeeze(task_model(labeled_imgs))
                task_loss = self.ce_loss(preds, labels)
                optim_task_model.zero_grad()
                task_loss.backward()
                optim_task_model.step()

            # VAE step
            for count in range(int(self.args["num_vae_steps"])):
                recon, z, mu, logvar = vae(labeled_imgs)
                unsup_loss = self.vae_loss(
                    labeled_imgs, recon, mu, logvar, self.args["beta"]
                )
                unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
                transductive_loss = self.vae_loss(
                    unlabeled_imgs,
                    unlab_recon,
                    unlab_mu,
                    unlab_logvar,
                    self.args["beta"],
                )

                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)

                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_real_preds = torch.ones(unlabeled_imgs.size(0))

                if self.args["cuda"]:
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_real_preds = unlab_real_preds.cuda()

                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + self.bce_loss(
                    unlabeled_preds, unlab_real_preds
                )
                total_vae_loss = (
                    unsup_loss
                    + transductive_loss
                    + self.args["adversary_param"] * dsc_loss
                )
                optim_vae.zero_grad()
                total_vae_loss.backward()
                optim_vae.step()

                # sample new batch if needed to train the adversarial network
                if count < int(self.args["num_vae_steps"] - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    if self.args["cuda"]:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()

            # Discriminator step
            for count in range(int(self.args["num_adv_steps"])):
                with torch.no_grad():
                    _, _, mu, _ = vae(labeled_imgs)
                    _, _, unlab_mu, _ = vae(unlabeled_imgs)

                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)

                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

                if self.args["cuda"]:
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_fake_preds = unlab_fake_preds.cuda()

                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + self.bce_loss(
                    unlabeled_preds, unlab_fake_preds
                )

                optim_discriminator.zero_grad()
                dsc_loss.backward()
                optim_discriminator.step()

                # sample new batch if needed to train the adversarial network
                if count < int(self.args["num_adv_steps"] - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    if self.args["cuda"]:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()

            if iter_count % 100 == 0:
                print("\nCurrent training iteration: {}".format(iter_count))

                if task_model:
                    print("\tCurrent task model loss: {:.4f}".format(task_loss.item()))
                print("\tCurrent vae model loss: {:.4f}".format(total_vae_loss.item()))
                print(
                    "\tCurrent discriminator model loss: {:.4f}\n".format(
                        dsc_loss.item()
                    )
                )

        if task_model:
            final_accuracy = self.test(task_model, querry_dataloader)
        else:
            final_accuracy = 0

        return final_accuracy, task_model, vae, discriminator

    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader):
        querry_indices = self.sampler.sample(
            vae, discriminator, unlabeled_dataloader, self.args["cuda"]
        )

        return querry_indices

    def test(self, task_model, test_dataloader):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels, _ in test_dataloader:
            if self.args["cuda"]:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100

    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD
