"""Loss functions. """
import torch


#----------------------------------------------------------------------------
# WGAN & WGAN-GP loss functions.

def G_wgan(D, fakes):
    fake_scores_out = D(fakes)
    loss = -fake_scores_out

    return loss


def D_wgan(D, fakes, reals, eps=1e-3):
    real_scores_out = D(reals)
    fake_scores_out = D(fakes)

    loss = fake_scores_out - real_scores_out
    loss += (real_scores_out ** 2) * eps

    return loss


def D_wgan_gp(D, fakes, reals, gp_lambda=10.0, gp_target=1.0, eps=1e-3):
    real_scores_out = D(reals)
    fake_scores_out = D(fakes)
    loss = fake_scores_out - real_scores_out

    batch_size = reals.size(0)
    mixing_factor = torch.rand(batch_size, 1, 1, 1, device=reals.device)

    interpolates = torch.lerp(fakes, reals, mixing_factor)
    interpolates = torch.nn.Parameter(interpolates)
    disc_interpolates = D(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(disc_interpolates),
                                    retain_graph=True, create_graph=True, only_inputs=True)[0]
    gradients = gradients.reshape(batch_size, -1)
    gradient_penalty = (gradients.norm(2, dim=1, keepdim=True) - gp_target) ** 2

    loss += gp_lambda * gradient_penalty.reshape_as(loss)
    loss += (real_scores_out ** 2) * eps

    return loss


#----------------------------------------------------------------------------
# Hinge loss functions. (Use G_wgan with these)

def D_hinge(D, fakes, reals):
    real_scores_out = D(reals)
    fake_scores_out = D(fakes)
    loss = torch.relu(1. + fake_scores_out) + torch.relu(1. - real_scores_out)

    return loss


def D_hinge_gp(D, fakes, reals, gp_lambda=10.0, gp_target=1.0):
    real_scores_out = D(reals)
    fake_scores_out = D(fakes)
    loss = torch.relu(1. + fake_scores_out) + torch.relu(1. - real_scores_out)

    batch_size = reals.size(0)
    mixing_factor = torch.rand(batch_size, 1, 1, 1, device=reals.device)

    interpolates = torch.lerp(fakes, reals, mixing_factor)
    interpolates = torch.nn.Parameter(interpolates)
    disc_interpolates = D(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(disc_interpolates),
                                    retain_graph=True, create_graph=True, only_inputs=True)[0]
    gradients = gradients.reshape(batch_size, -1)
    gradient_penalty = (gradients.norm(2, dim=1, keepdim=True) - gp_target) ** 2

    loss += gp_lambda * gradient_penalty.reshape_as(loss)

    return loss


#----------------------------------------------------------------------------
# Loss functions advocated by the paper
# "Which Training Methods for GANs do actually Converge?"

def G_logistic_saturating(D, fakes):
    fake_scores_out = D(fakes)
    loss = torch.log(1 - torch.sigmoid(fake_scores_out)) # -softplus(fake_scores_out)

    return loss


def G_logistic_nonsaturating(D, fakes):
    fake_scores_out = D(fakes)
    loss = -torch.log(torch.sigmoid(fake_scores_out)) # softplus(-fake_scores_out)

    return loss


def D_logistic(D, fakes, reals):
    real_scores_out = D(reals)
    fake_scores_out = D(fakes)
    loss = -torch.log(1 - torch.sigmoid(fake_scores_out)) # softplus(fake_scores_out)
    loss += -torch.log(torch.sigmoid(real_scores_out))    # softplus(-real_scores_out)

    return loss


def D_logistic_simplegp(D, fakes, reals, r1_gamma=10.0, r2_gamma=0.0):
    reals = torch.nn.Parameter(reals)
    fakes = torch.nn.Parameter(fakes)

    real_scores_out = D(reals)
    fake_scores_out = D(fakes)
    loss = -torch.log(1 - torch.sigmoid(fake_scores_out)) # softplus(fake_scores_out)
    loss += -torch.log(torch.sigmoid(real_scores_out))    # softplus(-real_scores_out)

    if r1_gamma != 0.0:
        real_grads = torch.autograd.grad(outputs=real_scores_out, inputs=reals,
                                         grad_outputs=torch.ones_like(real_scores_out),
                                         retain_graph=True, create_graph=True, only_inputs=True)[0]
        r1_penalty = torch.sum(real_grads ** 2, dim=(1, 2, 3)).reshape_as(loss)
        loss += r1_penalty * (r1_gamma * 0.5)

    if r2_gamma != 0.0:
        fake_grads = torch.autograd.grad(outputs=fake_scores_out, inputs=fakes,
                                         grad_outputs=torch.ones_like(fake_scores_out),
                                         retain_graph=True, create_graph=True, only_inputs=True)[0]
        r2_penalty = torch.sum(fake_grads ** 2, dim=(1, 2, 3)).reshape_as(loss)
        loss += r2_penalty * (r2_gamma * 0.5)

    return loss

#----------------------------------------------------------------------------

def G_loss(D, fakes, key: str = 'wgan'):
    if key == 'wgan':
        return G_wgan(D, fakes)
    if key == 'saturating':
        return G_logistic_saturating(D, fakes)
    if key == 'nonsaturating':
        return G_logistic_nonsaturating(D, fakes)

    raise RuntimeError(f"Not surpports loss function: '{key}'")


def D_loss(D, fakes, reals, key: str = 'wgan', eps: float = 1e-3,
           gp_lambda: float = 10.0, gp_target: float = 1.0,
           r1_gamma: float = 10.0, r2_gamma: float = 0.0):
    if key == 'wgan':
        return D_wgan(D, fakes, reals, eps)
    if key == 'wgan_gp':
        return D_wgan_gp(D, fakes, reals, gp_lambda, gp_target)
    if key == 'hinge':
        return D_hinge(D, fakes, reals)
    if key == 'hinge_gp':
        return D_hinge_gp(D, fakes, reals, gp_lambda, gp_target)
    if key == 'logistic':
        return D_logistic(D, fakes, reals)
    if key == 'logistic_simplegp':
        return D_logistic_simplegp(D, fakes, reals, r1_gamma, r2_gamma)

    raise RuntimeError(f"Not surpports loss function: '{key}'")
