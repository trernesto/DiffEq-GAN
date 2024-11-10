from data.exp_decay import ExpDecay
from data.coupled_oscillators import CoupOsc
from models.pinn import PINN

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

# Loss Section
def loss_phys(network: nn.Module, t: torch.tensor):
    u = network(t)
    du_dt = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    return du_dt

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
    u = torch.from_numpy(exp_decay.x).float()
    t = torch.from_numpy(exp_decay.t).float()
    u = u.view(-1, 1)
    t = u.view(-1, 1)
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
    opt = torch.optim.Adam(model.parameters())

    criterion_mse = nn.MSELoss()
    criterion_phys = loss_phys

    #scheduler = ...
    

    epochs = 100
    print_epoch = 50
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
            bt = Variable(batch_t, requires_grad=True)
            pdes = loss_phys(model, bt)
            dudt = pdes[0]

            eq = dudt + model_out
            zeros = torch.zeros_like(eq)
            loss_ph = criterion_mse(eq, zeros)


            loss = loss_mse + loss_ph
            mean_loss_batch += loss/len(data_train)
            loss.backward()
            opt.step()
        
        if (epoch + 1) % print_epoch == 0:
            print(f'epoch: {epoch + 1}')
            print('train loss:', mean_loss_batch.item())
        losses_train.append(mean_loss_batch.item())

        mean_loss_batch = 0
        for batch_u, batch_t in data_val:
            model_out = model(batch_t)

            #mse loss section
            loss_mse = criterion_mse(model_out, batch_u)

            #pde loss section
            bt = Variable(batch_t, requires_grad=True)
            pdes = loss_phys(model, bt)
            dudt = pdes[0]

            eq = dudt + model_out
            zeros = torch.zeros_like(eq)
            loss_ph = criterion_mse(eq, zeros)


            loss = loss_mse + loss_ph
            mean_loss_batch += loss/len(data_val)
        if (epoch + 1) % print_epoch == 0:    
            print('val loss:', mean_loss_batch.item())
        losses_val.append(mean_loss_batch.item())

