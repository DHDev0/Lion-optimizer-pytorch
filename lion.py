import torch
from torch.optim.optimizer import Optimizer

#codebase on https://github.com/lucidrains/lion-pytorch

class Lion(Optimizer):
    def __init__( self, params, lr: float = 1e-4, betas: tuple= (0.9, 0.99), weight_decay: float = 0.0):
        
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        defaults = dict(lr = lr, betas = betas, weight_decay = weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step( self, closure= None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None, group['params']):
                #parameter
                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]
                if len(state) == 0: state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                
                 # Lion optimizer
                p.data.mul_(1 - lr * wd)
                update = exp_avg.clone().lerp_(grad, 1 - beta1)
                p.add_(torch.sign(update), alpha = -lr)
                exp_avg.lerp_(grad, 1 - beta2)
        return loss
    
    
# def train(weight, gradient, momentum, lr):
#     update = interp(gradient,momentum, Beta1)
#     update = sign(update)
#     momentum = interp( gradient, momentum, Beta2)
#     weight_decay = weight * Sigma
#     update = update + weight_decay
#     update = update * lr
#     return update, momentum
