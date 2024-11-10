from data.exp_decay import ExpDecay
from data.coupled_oscillators import CoupOsc
from models.pinn import PINN

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

# Loss Section
# variables - tensor of all variables needed to differentate
def loss_phys(network: nn.Module, variables: torch.tensor):
    u = network(variables)
    pdes = torch.tensor(())
    for i in range(len(variables)):
        du_dvar = torch.autograd.grad(u.sum(), variables[i], create_graph=True)[0]
        pdes = torch.concat(pdes, du_dvar)
    return pdes

# same written in train, no need to use
def loss_full(network: nn.Module, variables: torch.tensor, true_u: torch.tensor):
    mse_loss = nn.MSELoss()
    u = network(variables)
    mse = mse_loss(true_u, u)
    pdes = loss_phys(network=network, variables=variables)

    #our equation with respect of var[0] = t for ExpDec
    phys = pdes[0] + u

    return mse + phys

# Main section
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # raw data
    exp_decay = ExpDecay(C = 1)
    u = exp_decay.x
    t = exp_decay.t
    exp_decay_const = exp_decay.C
    dataset = TensorDataset(u, t)

    
    g_cpu = torch.Generator()
    g_cpu = g_cpu.manual_seed(1231292572129)
    
    train_set, val_set, test_set = torch.utils.data.random_split(dataset=dataset, lengths=[0.8, 0.1, 0.1], generator = g_cpu)
    
    batch_size = 16
    data_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    data_val = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    data_test = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    #exp_decay.plot()

    #coup_osc = CoupOsc(C = 1)
    #coup_osc.plot()

    # data to train val test section

    model = PINN()
    model.to(device)
    opt = nn.optim.Adam(model.parameters())

    criterion_mse = nn.MSELoss()
    criterion_phys = loss_phys

    #scheduler = ...
    

    epochs = 100
    losses_train = []
    losses_val = []    
    for epoch in range(epochs):
        mean_loss_batch = 0
        for batch_u, batch_t in data_train:
            opt.zero_grad()
            model_out = model(batch_t)

            #mse loss section
            loss_mse = criterion_mse(model_out, batch_u)

            #pde loss section
            pdes = loss_phys(model, batch_t)
            dudt = pdes[0]

            eq = dudt + model_out
            zeros = torch.zeros_like(eq)
            loss_ph = criterion_mse(eq, zeros)


            loss = loss_mse + loss_ph
            mean_loss_batch += loss/len(data_train)
            loss.backwards()
            opt.step()
        print('train loss:', mean_loss_batch)
        losses_train.append(mean_loss_batch)

        with torch.no_grad():
            mean_loss_batch = 0
            for batch_t, batch_x in data_val:
                model_out = model(batch_t)

                #mse loss section
                loss_mse = criterion_mse(model_out, batch_u)

                #pde loss section
                pdes = loss_phys(model, batch_t)
                dudt = pdes[0]

                eq = dudt + model_out
                zeros = torch.zeros_like(eq)
                loss_ph = criterion_mse(eq, zeros)


                loss = loss_mse + loss_ph
                mean_loss_batch += loss/len(data_val)
            print('val loss:', mean_loss_batch)
            losses_val.append(mean_loss_batch)

