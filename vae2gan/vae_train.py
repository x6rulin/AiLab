import sys
import os
import torch
import torchvision

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

from dnnlib.miscs import ArgParse, Trainer
from dnnlib.util import Logger
from vae2gan.core.loss import G_loss, D_loss


class Args(ArgParse):

    def __init__(self):
        super(Args, self).__init__(description="Generative Adversarial Networks.")

        self.parser.add_argument("--betas", nargs=2, type=float, default=(0.9, 0.999), help="betas for optimizer Adam")
        self.parser.add_argument("--alpha", type=float, default=0.5, help="weight of the KLDiv loss for encoder")
        self.parser.add_argument("--img-dir", type=str, default='images', help="directory saving images generated")
        self.parser.add_argument("--log", type=str, default=None, help="file to save the training process log")


class VaeTrain(Trainer):

    def __init__(self, latent_size, enet, dnet, train_dataset, args=Args()):
        self.latent_size = latent_size

        super(VaeTrain, self).__init__(train_dataset, args=args)

        self.net = {'enet': enet.to(self.device), 'dnet': dnet.to(self.device)}
        self.optimizer = {'enet': torch.optim.Adam(self.net['enet'].parameters(), lr=self.args.lr, betas=self.args.betas),
                          'dnet': torch.optim.Adam(self.net['dnet'].parameters(), lr=self.args.lr, betas=self.args.betas)}
        self.criterion = {'enet': lambda mu, sigma2: torch.mean((-torch.log(sigma2) + mu ** 2 + sigma2 - 1) / 2),
                          'dnet': torch.nn.MSELoss(reduction='mean')}

        self.epoch = 0
        self.value = 0.

        if not os.path.exists(self.args.img_dir):
            os.makedirs(self.args.img_dir, 0o775)

    def train(self):
        with Logger(self.args.log, file_mode='a') as log:
            log.write(f"epochs: {self.epoch}\n")
            _i = 5
            for i, reals in enumerate(self.train_loader, 1):
                real_images = reals.to(self.device)
                normal = self.net['enet'](real_images)
                mu, sigma = normal[:, [0]], torch.exp(normal[:, [1]] / 2)
                encoder_loss = self.criterion['enet'](mu, sigma ** 2) * self.args.alpha

                latents = torch.randn(reals.size(0), self.latent_size, device=self.device) * sigma + mu
                fake_images = self.net['dnet'](latents)
                decoder_loss = self.criterion['dnet'](fake_images, real_images)

                loss = encoder_loss + decoder_loss
                self.optimizer['enet'].zero_grad()
                self.optimizer['dnet'].zero_grad()
                loss.backward()
                self.optimizer['enet'].step()
                self.optimizer['dnet'].step()

                if i % self.args.print_freq == 0:
                    log.write(f"[epoch: {self.epoch} - {i}/{len(self.train_loader)}]"
                              f"Loss: {loss:.6f} - En_loss: {encoder_loss:.6f} - De_loss: {decoder_loss:.6f}\n")

                    torchvision.utils.save_image(real_images, os.path.join(self.args.img_dir, f"real_sample_{i}.png"),
                                                 nrow=round(pow(self.args.batch_size, 0.5)), normalize=True, scale_each=True)
                    torchvision.utils.save_image(fake_images, os.path.join(self.args.img_dir, f"fake_sample_{i}.png"),
                                                 nrow=round(pow(self.args.batch_size, 0.5)), normalize=True, scale_each=True)

    def validate(self):
        return self.value


if __name__ == "__main__":
    from vae2gan.core.network import Lat2Img, Img2Dis
    from vae2gan.core.dataset import FFHQDataset

    img_root = r"/home/data/ffhq_thumbnails128x128"
    resolution, num_channels, _latent_size, num_dis = 128, 3, 128, 2

    _train_dataset = FFHQDataset(img_root, num_channels, resolution)
    _enet = Img2Dis(num_dis, num_channels, resolution=resolution, nonlinearity='prelu', normalization='LN')
    _dnet = Lat2Img(_latent_size, num_channels, resolution=resolution, nonlinearity='prelu', normalization='LN')

    _train = VaeTrain(_latent_size, _enet, _dnet, _train_dataset)
    _train()
