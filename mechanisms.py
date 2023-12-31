import math
import torch
import torch.nn.functional as F
from scipy.special import erf


class Mechanism:
    def __init__(self, eps, input_range, **kwargs):
        self.eps = eps
        self.alpha, self.beta = input_range

    def __call__(self, x):
        raise NotImplementedError


class Laplace(Mechanism):
    def __call__(self, x):
        d = x.size(1)
        sensitivity = (self.beta - self.alpha) * d
        scale = torch.ones_like(x) * (sensitivity / self.eps)
        out = torch.distributions.Laplace(x, scale).sample()
        # out = torch.clip(out, min=self.alpha, max=self.beta)
        return out


class MultiBit(Mechanism):
    def __init__(self, *args, m='best', **kwargs):
        super().__init__(*args, **kwargs)
        self.m = m

    def __call__(self, x):
        n, d = x.size()
        if self.m == 'best':
            m = int(max(1, min(d, math.floor(self.eps / 2.18))))
        elif self.m == 'max':
            m = d
        else:
            m = self.m

        # sample features for perturbation
        BigS = torch.rand_like(x).topk(m, dim=1).indices
        s = torch.zeros_like(x, dtype=torch.bool).scatter(1, BigS, True)
        del BigS

        # perturb sampled features
        em = math.exp(self.eps / m)
        p = (x - self.alpha) / (self.beta - self.alpha)
        p = (p * (em - 1) + 1) / (em + 1)
        t = torch.bernoulli(p)
        x_star = s * (2 * t - 1)
        del p, t, s

        # unbiase the result
        x_prime = d * (self.beta - self.alpha) / (2 * m)
        x_prime = x_prime * (em + 1) * x_star / (em - 1)
        x_prime = x_prime + (self.alpha + self.beta) / 2

        return x_prime


class OneBit(MultiBit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, m='max', **kwargs)


class Gaussian(Mechanism):
    def __init__(self, *args, delta=1e-10, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.sigma = None
        self.sensitivity = None

    def __call__(self, x):
        len_interval = self.beta - self.alpha
        if torch.is_tensor(len_interval) and len(len_interval) > 1:
            self.sensitivity = torch.norm(len_interval, p=2)
        else:
            d = x.size(1)
            self.sensitivity = len_interval * math.sqrt(d)

        self.sigma = self.calibrate_gaussian_mechanism()
        out = torch.normal(mean=x, std=self.sigma)
        # out = torch.clip(out, min=self.alpha, max=self.beta)
        return out

    def calibrate_gaussian_mechanism(self):
        return self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.eps



class Piecewise(Mechanism):
    def __call__(self, x):
        # normalize x between -1,1
        t = (x - self.alpha) / (self.beta - self.alpha)
        t = 2 * t - 1

        # piecewise mechanism's variables
        P = (math.exp(self.eps) - math.exp(self.eps / 2)) / (2 * math.exp(self.eps / 2) + 2)
        C = (math.exp(self.eps / 2) + 1) / (math.exp(self.eps / 2) - 1)
        L = t * (C + 1) / 2 - (C - 1) / 2
        R = L + C - 1

        # thresholds for random sampling
        threshold_left = P * (L + C) / math.exp(self.eps)
        threshold_right = threshold_left + P * (R - L)

        # masks for piecewise random sampling
        x = torch.rand_like(t)
        mask_left = x < threshold_left
        mask_middle = (threshold_left < x) & (x < threshold_right)
        mask_right = threshold_right < x

        # random sampling
        t = mask_left * (torch.rand_like(t) * (L + C) - C)
        t += mask_middle * (torch.rand_like(t) * (R - L) + L)
        t += mask_right * (torch.rand_like(t) * (C - R) + R)

        # unbias data
        x_prime = (self.beta - self.alpha) * (t + 1) / 2 + self.alpha
        return x_prime


class MultiDimPiecewise(Piecewise):
    def __call__(self, x):
        n, d = x.size()
        k = int(max(1, min(d, math.floor(self.eps / 2.5))))
        sample = torch.rand_like(x).topk(k, dim=1).indices
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(1, sample, True)
        self.eps /= k
        y = super().__call__(x)
        z = mask * y * d / k
        return z


class RandomizedResopnse:
    def __init__(self, eps, d):
        self.d = d
        self.q = 1.0 / (math.exp(eps) + self.d - 1)
        self.p = self.q * math.exp(eps)

    def __call__(self, y):
        pr = y * self.p + (1 - y) * self.q
        out = torch.multinomial(pr, num_samples=1)
        return F.one_hot(out.squeeze(), num_classes=self.d)


supported_feature_mechanisms = {
    'mbm': MultiBit,
    'mpm': MultiDimPiecewise
}
