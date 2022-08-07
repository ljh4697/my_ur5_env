import pymc as mc
import numpy as np
import theano as th
import theano.tensor as tt
import theano.tensor.nnet as tn
from theano.ifelse import ifelse
from scipy.stats import gaussian_kde
from utils import matrix
from pymc.Matplot import plot



def sample(self, N, T=50, burn=1000):
    x = mc.Uniform('x', -np.ones(self.D), np.ones(self.D), value=np.zeros(self.D))
    def sphere(x):
        if (x**2).sum()>=1.:
            return -np.inf
        else:
            return self.f(x)
    p1 = mc.Potential(
        logp = sphere,
        name = 'sphere',
        parents = {'x': x},
        doc = 'Sphere potential',
        verbose = 0)
    chain = mc.MCMC([x])
    chain.use_step_method(mc.AdaptiveMetropolis, x, delay=burn, cov=np.eye(self.D)/10000)
    chain.sample(N*T+burn, thin=T, burn=burn, verbose=-1)
    samples = x.trace()
    samples = np.array([x/np.linalg.norm(x) for x in samples])
    return samples

def main():
    task = 'Tosser'
    print(task)
        




if __name__ == "__main__":
    main()