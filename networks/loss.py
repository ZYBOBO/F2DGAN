import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instant_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def efdm_loss(fake, xs):
    b, c, h, w = fake.size()
    xs = xs.view(b, -1, c, h, w)
    real = torch.mean(xs, dim=1)
    value_content, index_content = torch.sort(fake.view(b, c, -1))
    value_style, index_style = torch.sort(real.view(b, c, -1))
    inverse_index = index_content.argsort(-1)
    return nn.MSELoss()(fake.view(b, c, -1), value_style.gather(-1, inverse_index))


def ortho_loss(x, y, c):
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    mask = torch.eq(c, c.t()).bool().to(x.device)
    eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(x.device)
    mask_pos = mask.masked_fill(eye, 0).float()
    mask_neg = (~mask).float()

    dot_prod = torch.matmul(x, y.t())
    pos_total = (mask_pos * dot_prod).sum()
    neg_total = torch.abs(mask_neg * dot_prod).sum()
    pos_mean = pos_total / (mask_pos.sum() + 1e-6)
    neg_mean = neg_total / (mask_neg.sum() + 1e-6)

    loss = (1.0 - pos_mean) + (2.0 * neg_mean)
    return loss


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def calc_content_loss(input, target, norm=False):
    mse_loss = nn.MSELoss()
    if (norm == False):
        return mse_loss(input, target)
    else:
        return mse_loss(mean_variance_norm(input), mean_variance_norm(target))


def calc_style_loss(input, target):
    mse_loss = nn.MSELoss()
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return mse_loss(input_mean, target_mean) + \
           mse_loss(input_std, target_std)


def adv_recon_criterion(real, fake, base_index):
    b, c, h, w = fake.size()
    real = real.view(b, -1, c, h, w)
    ref = torch.cat([real[:, :base_index, :, :, :], real[:, (base_index + 1):, :, :, :]], dim=1)
    losses = [torch.mean(torch.abs(ref[:, i, :, :, :] - fake)) for i in range(ref.size(1))]
    losses = min(losses)

    return losses


def feat_recon_criterion(real_feats, fake_feat):
    b, c, h, w = fake_feat.size()
    real_feats = real_feats.view(b, -1, c, h, w)
    chan_feat = torch.mean(real_feats, dim=1, keepdim=False)
    cont_loss = calc_content_loss(fake_feat, chan_feat)
    styl_loss = calc_style_loss(fake_feat, chan_feat)

    return cont_loss + styl_loss


def recon_criterion(target, output):
    loss = torch.mean(torch.abs(output - target))
    return loss


def feat_matching_loss(fake_feat, real_feat):
    b, c, h, w = fake_feat.size()
    value_cont, index_cont = torch.sort(fake_feat.view(b, c, -1))
    real_feat = real_feat.view(b, -1, c, h, w)
    real_feat = torch.mean(real_feat, dim=1)
    value_style, index_style = torch.sort(real_feat.view(b, c, -1))
    inverse_index = index_cont.argsort(-1)
    mse_loss = nn.MSELoss()

    return mse_loss(fake_feat.view(b, c, -1), value_style.gather(-1, inverse_index))


