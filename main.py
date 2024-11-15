from data.exp_decay import ExpDecay
from data.coupled_oscillators import CoupOsc
from models.pinn import PINN

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt

# Loss Section
def loss_phys(network: nn.Module, t: torch.tensor):
    u = network(t)
    du_dt = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    #du_dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    return du_dt

# Main section
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # raw data
    exp_decay = ExpDecay(C = 1)
    u = torch.tensor(exp_decay.x, dtype = torch.float).view(-1, 1)
    t = torch.tensor(exp_decay.t, dtype = torch.float).view(-1, 1)
    exp_decay_const = exp_decay.C
    dataset = TensorDataset(u, t)

    
    g_cpu = torch.Generator()
    g_cpu = g_cpu.manual_seed(42)
    
    train_set, val_set, test_set = torch.utils.data.random_split(dataset=dataset, lengths=[0.3, 0.35, 0.35], generator = g_cpu)
    
    batch_size = 64
    data_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    data_val = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    data_test = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    #exp_decay.plot()

    #coup_osc = CoupOsc(C = 1)
    #coup_osc.plot()

    # data to train val test section

    model = PINN(number_of_hidden_layers=2, hidden_layer_size=8)
    model.to(device)
    opt = torch.optim.Adam(model.parameters())

    criterion_mse = nn.MSELoss()
    criterion_phys = loss_phys

    #scheduler = ...
    

    epochs = 500
    print_epoch = 100
    
    losses_train = []
    losses_val = []
    losses_mse_train = []
    losses_mse_val = []
    losses_phys_train = []
    losses_phys_val = []    
    
    for epoch in range(epochs):
        mean_loss_batch = 0
        mean_loss_mse = 0
        mean_loss_phys = 0
        for batch_u, batch_t in data_train:
            opt.zero_grad()
            model_out = model(batch_t)

            #mse loss section
            loss_mse = criterion_mse(model_out, batch_u)

            #pde loss section
            bt = Variable(batch_t, requires_grad=True)
            dudt = loss_phys(model, bt)

            eq = dudt + model_out
            zeros = torch.zeros_like(eq)
            loss_ph = criterion_mse(eq, zeros)


            loss = loss_mse + loss_ph
            mean_loss_batch += loss/len(data_train)
            mean_loss_mse += loss_mse/len(data_train)
            mean_loss_phys += loss_ph/len(data_train)
            loss.backward()
            opt.step()
        
        if (epoch + 1) % print_epoch == 0:
            print(f'epoch: {epoch + 1}')
            print('train mse:', mean_loss_mse.item())
            print('train phys:', mean_loss_phys.item())
        losses_train.append(mean_loss_batch.item())
        losses_mse_train.append(mean_loss_mse.item())
        losses_phys_train.append(mean_loss_phys.item())

        mean_loss_batch = 0
        mean_loss_mse = 0
        mean_loss_phys = 0
        for batch_u, batch_t in data_val:
            model_out = model(batch_t)

            #mse loss section
            loss_mse = criterion_mse(model_out, batch_u)

            #pde loss section
            bt = Variable(batch_t, requires_grad=True)
            dudt = loss_phys(model, bt)

            eq = dudt + model_out
            zeros = torch.zeros_like(eq)
            loss_ph = criterion_mse(eq, zeros)
            #loss_ph = torch.tensor((0), dtype = torch.float,  requires_grad =  True)


            loss = loss_mse + loss_ph
            mean_loss_batch += loss/len(data_val)
            mean_loss_mse += loss_mse/len(data_val)
            mean_loss_phys += loss_ph/len(data_val)
        if (epoch + 1) % print_epoch == 0:    
            print('val mse:', mean_loss_mse.item())
            print('val phys:', mean_loss_phys.item())
        losses_val.append(mean_loss_batch.item())
        losses_mse_val.append(mean_loss_mse.item())
        losses_phys_val.append(mean_loss_phys.item())

    # Test secti
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle('Loss function')

    x = np.arange(epochs)

    axs[0].set_title('mse + phys loss')
    axs[0].plot(x, losses_train, label = 'train')
    axs[0].plot(x, losses_val, label = 'val')
    axs[0].legend()

    
    axs[1].set_title('mse only')
    axs[1].plot(x, losses_mse_train, label = 'train')
    axs[1].plot(x, losses_mse_val, label = 'val')
    axs[1].legend()

    
    axs[2].set_title('phys only')
    axs[2].plot(x, losses_phys_train, label = 'train')
    axs[2].plot(x, losses_phys_val, label = 'val')
    axs[2].legend()

    #plt.show()
    plt.close()

    plt.clf()
    plt.cla()
    
    plot_true_u = np.array([])
    plot_u = np.array([])
    plot_t = np.array([])

    for true_u, t in data_test:
        u = model(t)
        plot_true_u = np.concatenate((plot_true_u, true_u.detach().numpy().flatten()), axis = None)
        plot_u = np.concatenate((plot_u, u.detach().numpy().flatten()), axis = None)
        plot_t = np.concatenate((plot_t, t.detach().numpy().flatten()), axis = None)
    
    plt.title('True u and predicted u')
    _, plot_true_u = zip(*sorted(zip(plot_t, plot_true_u)))
    plot_t, plot_u = zip(*sorted(zip(plot_t, plot_u)))
    plt.plot(plot_t, plot_true_u, label = 'true u')
    plt.plot(plot_t, plot_u, label = 'Network prediction')
    plt.legend()
    plt.show()
