import sys
import os
import torch
from torchvision.utils import save_image

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

from dnnlib.miscs import ArgParse, Trainer
from dnnlib.util import Logger, EMA
from vae2gan.core.loss import G_loss, D_loss


class Args(ArgParse):

    def __init__(self):
        super(Args, self).__init__(description="Generative Adversarial Networks.")

        self.parser.add_argument("--betas", nargs=2, type=float, default=(0., 0.99), help="betas for optimizer Adam")
        self.parser.add_argument("--loss", nargs=2, type=str, default=('nonsaturating', 'logistic_simplegp'),
                                 help="wgan | saturating | nonsaturating and wgan | wgan_gp | hinge | hinge_gp | logistic | logistic_simplegp")
        self.parser.add_argument("--ema", type=float, default=0.999, help="decay ratio for Generator's weighted EMA")
        self.parser.add_argument("--nc", type=int, default=1, help="iterations of critic training each turn")
        self.parser.add_argument("--ng", type=int, default=1, help="iterations of generator training each turn")
        self.parser.add_argument("--img-dir", type=str, default='images', help="directory saving images generated")
        self.parser.add_argument("--log", type=str, default=None, help="file to save the training process log")


class GanTrain(Trainer):

    def __init__(self, latent_size, gnet, dnet, train_dataset, args=Args()):
        self.latent_size = latent_size

        super(GanTrain, self).__init__(train_dataset, args=args)

        self.net = {'gnet': gnet.to(self.device), 'dnet': dnet.to(self.device)}
        self.optimizer = {'gnet': torch.optim.Adam(self.net['gnet'].parameters(), lr=self.args.lr, betas=self.args.betas),
                          'dnet': torch.optim.Adam(self.net['dnet'].parameters(), lr=self.args.lr, betas=self.args.betas)}
        self.criterion = {'gnet': lambda D, fakes: G_loss(D, fakes, self.args.loss[0]),
                          'dnet': lambda D, fakes, reals: D_loss(D, fakes, reals, self.args.loss[1])}

        self.epoch = 0
        self.value = 0.

        self.ema = EMA(self.net['gnet'], self.args.ema)
        self._appendcell(['ema'])

        if not os.path.exists(self.args.img_dir):
            os.makedirs(self.args.img_dir, 0o775)

    def train(self):
        with Logger(self.args.log, file_mode='a') as log:
            log.write(f"epochs: {self.epoch}\n")
            _ic, _pf = 1, 1
            for i, reals in enumerate(self.train_loader, 1):
                self.net['dnet'].requires_grad_(True)

                with torch.no_grad():
                    latents = torch.randn(reals.size(0), self.latent_size, device=self.device)
                    fake_images = self.net['gnet'](latents)
                real_images = reals.to(self.device)
                d_loss = self.criterion['dnet'](self.net['dnet'], fake_images, real_images).mean()

                self.optimizer['dnet'].zero_grad()
                d_loss.backward()
                self.optimizer['dnet'].step()

                if self.args.loss[1] == 'wgan':
                    for param in self.net['dnet'].parameters():
                        param.data.clamp_(-0.01, 0.01)

                if _ic == self.args.nc:
                    _ic = 1
                    for _ in range(self.args.ng):
                        self.net['dnet'].requires_grad_(False)

                        latents = torch.randn(reals.size(0), self.latent_size, device=self.device)
                        fake_images = self.net['gnet'](latents)
                        g_loss = self.criterion['gnet'](self.net['dnet'], fake_images).mean()

                        self.optimizer['gnet'].zero_grad()
                        g_loss.backward()
                        self.optimizer['gnet'].step()

                        self.ema.update()

                    if _pf == self.args.print_freq:
                        _pf = 1
                        log.write(f"[epoch: {self.epoch} - {i}/{len(self.train_loader)}]"
                                  f"Loss_dnet: {d_loss:.6f} - Loss_gnet: {g_loss:.6f}\n")

                        with torch.no_grad():
                            self.ema.apply_shadow()
                            latents = torch.randn(self.args.batch_size, self.latent_size, device=self.device)
                            fake_images = self.net['gnet'](latents)
                            self.ema.restore()

                        save_image(real_images, os.path.join(self.args.img_dir, f"real_sample_{i}.png"),
                                   nrow=round(pow(self.args.batch_size, 0.5)), normalize=True, scale_each=True)
                        save_image(fake_images, os.path.join(self.args.img_dir, f"fake_sample_{i}.png"),
                                   nrow=round(pow(self.args.batch_size, 0.5)), normalize=True, scale_each=True)
                    else: _pf += 1
                else: _ic += 1

    def validate(self):
        self.ema.apply_shadow()
        torch.save(self.ema.model.state_dict(), os.path.join(self.args.chkpt_dir, "generator.pth"))
        self.ema.restore()

        return self.value


if __name__ == "__main__":
    from vae2gan.core.network import Lat2Img, Img2Dis
    from vae2gan.core.dataset import FFHQDataset

    img_root = r"/home/data/ffhq_thumbnails128x128"
    resolution, num_channels, _latent_size, num_dis = 128, 3, 128, 1

    _train_dataset = FFHQDataset(img_root, num_channels, resolution)
    _gnet = Lat2Img(_latent_size, num_channels, resolution=resolution, nonlinearity='prelu', normalization='LN')
    _dnet = Img2Dis(num_dis, num_channels, resolution=resolution, nonlinearity='prelu', normalization='LN')

    _train = GanTrain(_latent_size, _gnet, _dnet, _train_dataset)
    _train()
