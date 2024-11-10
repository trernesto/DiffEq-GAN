from data.exp_decay import ExpDecay
from data.coupled_oscillators import CoupOsc
from models.pinn import PINN

import torch
import torch.nn as nn
from torch.autograd import Variable

# Loss Section
# variables - tensor of all variables needed to differentate
def loss_phys(network: nn.Module, variables: torch.tensor):
    u = network(variables)
    pdes = torch.tensor(())
    for i in range(len(variables)):
        du_dvar = torch.autograd.grad(u.sum(), variables[i], create_graph=True)[0]
        pdes = torch.concat(pdes, du_dvar)
    return pdes

def loss_full(network: nn.Module, variables: torch.tensor, true_u: torch.tensor):
    mse_loss = nn.MSELoss()
    u = network(variables)
    mse = mse_loss(true_u, u)
    phys = loss_phys(network=network, variables=variables)
    return mse + phys

# Main section
if __name__ == '__main__':
    
    exp_decay = ExpDecay(C = 1)
    #exp_decay.plot()

    coup_osc = CoupOsc(C = 1)
    coup_osc.plot()

    model = ...
    opt = nn.optim.Adam(model.parameters())
    scheduler = ...
    

    epochs = 100
    losses_train = []
    losses_val = []    
    for epoch in range(epochs):
        mean_loss_batch = 0
        for batch_t, batch_x in data_train:
            opt.zero_grad()
            model_out = model(batch_t)
            loss = criterion(model_out, batch_x)
            mean_loss_batch += loss/len(data_train)
            loss.backwards()
            opt.step()
        losses_train.append(mean_loss_batch)

        with torch.no_grad():
            mean_loss_batch = 0
            for batch_t, batch_x in data_val:
                model_out = model(batch_t)
                loss = criterion(model_out, batch_x)
                mean_loss_batch += loss / len(data_val)
            losses_val.append(mean_loss_batch)

