import random

import numpy as np
from torch import autograd

from networks.blocks import *
from networks.loss import *


class F2DGAN(nn.Module):
    def __init__(self, config):
        super(F2DGAN, self).__init__()

        self.gen = Generator(config['gen'])
        self.dis = Discriminator(config['dis'])

        self.w_adv_g = config['w_adv_g']
        self.w_adv_d = config['w_adv_d']
        self.w_recon = config['w_recon']
        self.w_feat = config['w_feat']
        self.w_ortho = config['w_ortho']
        self.w_cls = config['w_cls']
        self.w_gp = config['w_gp']
        self.n_sample = config['n_sample_train']

    def forward(self, xs, y, mode):
        if mode == 'gen_update':
            fake_x, loss_vae, qe_hat, proto_feat_rec = self.gen(xs)

            loss_recon = efdm_loss(fake_x, xs)
            loss_reg = ortho_loss(qe_hat, proto_feat_rec, y)
            logit_adv_fake, logit_c_fake = self.dis(fake_x)

            loss_adv_gen = torch.mean(-logit_adv_fake)
            loss_cls_gen = F.cross_entropy(logit_c_fake, y.squeeze())

            loss_recon = loss_recon * self.w_recon
            loss_feat = loss_vae * self.w_feat
            loss_reg = loss_reg * self.w_ortho
            loss_adv_gen = loss_adv_gen * self.w_adv_g
            loss_cls_gen = loss_cls_gen * self.w_cls

            loss_total = loss_recon + loss_adv_gen + loss_cls_gen + loss_feat + loss_reg
            loss_total.backward()

            return {'loss_total': loss_total,
                    'loss_recon': loss_recon,
                    'loss_feat': loss_feat,
                    'loss_reg': loss_reg,
                    'loss_adv_gen': loss_adv_gen,
                    'loss_cls_gen': loss_cls_gen}

        elif mode == 'dis_update':
            xs.requires_grad_()

            logit_adv_real, logit_c_real = self.dis(xs)
            loss_adv_dis_real = torch.nn.ReLU()(1.0 - logit_adv_real).mean()
            loss_adv_dis_real = loss_adv_dis_real * self.w_adv_d
            loss_adv_dis_real.backward(retain_graph=True)

            y_extend = y.repeat(1, self.n_sample).view(-1)
            index = torch.LongTensor(range(y_extend.size(0))).cuda()
            logit_c_real_forgp = logit_c_real[index, y_extend].unsqueeze(1)
            loss_reg_dis = self.calc_grad2(logit_c_real_forgp, xs)

            loss_reg_dis = loss_reg_dis * self.w_gp
            loss_reg_dis.backward(retain_graph=True)

            loss_cls_dis = F.cross_entropy(logit_c_real, y_extend)
            loss_cls_dis = loss_cls_dis * self.w_cls
            loss_cls_dis.backward()

            with torch.no_grad():
                fake_x = self.gen(xs)[0]

            logit_adv_fake, _ = self.dis(fake_x.detach())
            loss_adv_dis_fake = torch.nn.ReLU()(1.0 + logit_adv_fake).mean()
            loss_adv_dis_fake = loss_adv_dis_fake * self.w_adv_d
            loss_adv_dis_fake.backward()

            loss_total = loss_adv_dis_real + loss_adv_dis_fake + loss_cls_dis
            return {'loss_total': loss_total,
                    'loss_adv_dis': loss_adv_dis_fake + loss_adv_dis_real,
                    'loss_adv_dis_real': loss_adv_dis_real,
                    'loss_adv_dis_fake': loss_adv_dis_fake,
                    'loss_cls_dis': loss_cls_dis,
                    'loss_reg': loss_reg_dis}

        else:
            assert 0, 'Not support operation'

    def generate(self, xs):
        fake_x = self.gen(xs)[0]
        return fake_x

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()
        reg /= batch_size
        return reg


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.soft_label = False
        nf = config['nf']
        n_class = config['num_classes']
        n_res_blks = config['n_res_blks']

        cnn_f = [Conv2dBlock(3, nf, 5, 1, 2,
                             pad_type='reflect',
                             norm='sn',
                             activation='none')]
        for i in range(n_res_blks):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])

        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
        cnn_adv = [nn.AdaptiveAvgPool2d(1),
                   Conv2dBlock(nf_out, 1, 1, 1,
                               norm='none',
                               activation='none',
                               activation_first=False)]
        cnn_c = [nn.AdaptiveAvgPool2d(1),
                 Conv2dBlock(nf_out, n_class, 1, 1,
                             norm='none',
                             activation='none',
                             activation_first=False)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_adv = nn.Sequential(*cnn_adv)
        self.cnn_c = nn.Sequential(*cnn_c)

    def forward(self, x):
        if len(x.size()) == 5:
            B, K, C, H, W = x.size()
            x = x.view(B * K, C, H, W)
        else:
            B, C, H, W = x.size()
            K = 1
        feat = self.cnn_f(x)
        logit_adv = self.cnn_adv(feat).view(B * K, -1)
        logit_c = self.cnn_c(feat).view(B * K, -1)
        return logit_adv, logit_c


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        nf = config['nf']
        self.alpha = config['alpha']
        self.encoder = Encoder(
            3,
            nf,
            'bn',
            activ='relu',
            pad_type='reflect')

        self.decoder = Decoder(
            128,
            3,
            norm='bn',
            activ='relu',
            pad_type='reflect')
        self.vae = VAE(8192, 8192, 8192)
        # self.vae = VAE(2048, 2048, 2048)

    def forward(self, xs):
        b, k, C, H, W = xs.size()
        xs = xs.view(-1, C, H, W)
        querys, s_feats = self.encoder(xs)
        c, h, w = querys.size()[-3:]
        querys = querys.view(b, k, c, h, w)
        base_index = random.choice(range(k))
        cont_feat = querys[:, base_index, :, :, :]  # (b, 512, 8, 8)
        skips = []
        for i in range(len(s_feats)):
            q, u, v = s_feats[i].size()[-3:]
            skip = s_feats[i].view(b, k, q, u, v)
            b_skip = skip[:, base_index, :, :, :]
            r_skip = torch.mean(skip, dim=1)
            value_cont, index_cont = torch.sort(b_skip.view(b, q, -1))
            value_style, _ = torch.sort(r_skip.view(b, q, -1))
            inverse_index = index_cont.argsort(-1)
            skip = b_skip.view(b, q, -1) + (value_style.gather(-1, inverse_index) - b_skip.view(b, q, -1).detach())
            skip = skip.view(b, q, u, v)
            skips.append(skip)
        qe_input = cont_feat.view(b, -1)
        en_input = torch.mean(querys.view(b, k, -1), dim=1)
        proto_feat_rec, mu, log_var = self.vae(en_input)

        loss_vae = self.vae.loss_function(qe_input, proto_feat_rec, mu, log_var)

        proto_feat = proto_feat_rec.view(b, c, h, w)
        qe_hat_feat = qe_input.view(b, c, h, w)

        value_cont, index_cont = torch.sort(qe_hat_feat.view(b, c, -1))
        value_style, _ = torch.sort(proto_feat.view(b, c, -1))
        inverse_index = index_cont.argsort(-1)
        feat = qe_hat_feat.view(b, c, -1) + (
                    value_style.gather(-1, inverse_index) - qe_hat_feat.view(b, c, -1).detach())
        feat = feat.view(b, c, h, w)
        final_feat = self.alpha * qe_hat_feat + (1 - self.alpha) * feat
        fake_x = self.decoder(final_feat, skips)

        return fake_x, loss_vae, qe_input, proto_feat_rec


class Encoder(nn.Module):
    def __init__(self, input_dim, dim, norm, activ, pad_type):
        super(Encoder, self).__init__()

        self.conv1 = Conv2dBlock(input_dim, dim, 7, 1, 3,
                                 norm=norm,
                                 activation=activ,
                                 pad_type=pad_type)
        self.conv2 = Conv2dBlock(dim, 2 * dim, 4, 2, 1,
                                 norm=norm,
                                 activation=activ,
                                 pad_type=pad_type)
        self.conv3 = Conv2dBlock(2 * dim, 4 * dim, 4, 2, 1,
                                 norm=norm,
                                 activation=activ,
                                 pad_type=pad_type)
        self.conv4 = Conv2dBlock(4 * dim, 4 * dim, 4, 2, 1,
                                 norm=norm,
                                 activation=activ,
                                 pad_type=pad_type)

        self.conv5 = Conv2dBlock(4 * dim, 4 * dim, 4, 2, 1,
                                 norm=norm,
                                 activation=activ,
                                 pad_type=pad_type)
        self.output_dim = dim * 8

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        return out5, [out4, out3, out2, out1]


class Decoder(nn.Module):
    def __init__(self, dim, out_dim, norm, activ, pad_type):
        super(Decoder, self).__init__()

        self.conv5 = nn.Sequential(nn.Upsample(scale_factor=2),
                                   Conv2dBlock(dim, dim, 5, 1, 2,
                                               norm=norm,
                                               activation=activ,
                                               pad_type=pad_type))

        self.conv4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                   Conv2dBlock(2 * dim, dim, 5, 1, 2,
                                               norm=norm,
                                               activation=activ,
                                               pad_type=pad_type)
                                   )
        self.conv3 = nn.Sequential(nn.Upsample(scale_factor=2),
                                   Conv2dBlock(2 * dim, dim // 2, 5, 1, 2,
                                               norm=norm,
                                               activation=activ,
                                               pad_type=pad_type)
                                   )
        self.conv2 = nn.Sequential(nn.Upsample(scale_factor=2),
                                   Conv2dBlock(dim, dim // 4, 5, 1, 2,
                                               norm=norm,
                                               activation=activ,
                                               pad_type=pad_type)
                                   )
        self.conv1 = nn.Sequential(Conv2dBlock(dim // 2, out_dim, 7, 1, 3,
                                               norm='none',
                                               activation='tanh',
                                               pad_type=pad_type)
                                   )

    def forward(self, x, skips):
        out1 = self.conv5(x)
        out2 = self.conv4(torch.cat([skips[0], out1], dim=1))
        out3 = self.conv3(torch.cat([skips[1], out2], dim=1))
        out4 = self.conv2(torch.cat([skips[2], out3], dim=1))
        out5 = self.conv1(torch.cat([skips[3], out4], dim=1))
        return out5


class VAE(nn.Module):
    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, True)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, in_channels),
            nn.Sigmoid()
        )

    def encode(self, input):
        result = self.encoder(input)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        z = self.decoder_input(z)
        z_out = self.decoder(z)
        return z_out

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z= self.reparameterize(mu, log_var)
        z_out = self.decode(z)

        return z_out, mu, log_var

    def loss_function(self, input, rec, mu, log_var, kld_weight=0.0025):
        recon_loss = F.mse_loss(rec, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recon_loss + kld_weight * kld_loss

        return loss


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dim, n_blk, norm, activ):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(in_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, out_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))