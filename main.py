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

    train_exp_decay = ExpDecay(C = 1, N = 301, a = 0, b = 3)
    train_exp_decay.set_points_to_random(C_start=0, C_finish=10)
    train_u = torch.tensor(train_exp_decay.x, dtype = torch.float).view(-1, 1)
    train_t = torch.tensor(train_exp_decay.t, dtype = torch.float).view(-1, 1)
    exp_decay_const = train_exp_decay.C
    train_dataset = TensorDataset(train_u, train_t)

    # raw data
    exp_decay = ExpDecay(C = 3, N = 101)
    test_u = torch.tensor(exp_decay.x, dtype = torch.float).view(-1, 1)
    test_t = torch.tensor(exp_decay.t, dtype = torch.float).view(-1, 1)
    exp_decay_const = exp_decay.C
    test_dataset = TensorDataset(test_u, test_t)

    
    g_cpu = torch.Generator()
    g_cpu = g_cpu.manual_seed(42)
    
    train_set, val_set = torch.utils.data.random_split(dataset=train_dataset, lengths=[0.5, 0.5], generator = g_cpu)
    
    test_set = test_dataset
    
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
    

    epochs = 20_000
    print_epoch = 1_000
    
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

            #Boundary conditions
            x_right_boundary = torch.tensor([20], dtype = torch.float, requires_grad =  True)
            
            u_predicted_boundary = model(x_right_boundary)
            #You can use this, but lim b->inf -> u_true -> 0
            #u_true = train_exp_decay.equation(x_right_boundary)
            u_true = torch.zeros_like(u_predicted_boundary)
            
            loss_boundary = criterion_mse(u_predicted_boundary, u_true)

            loss = loss_mse + loss_ph + loss_boundary
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


    plt.savefig('./losses.png')
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
        
    plot_train_u = np.array([])
    plot_train_t = np.array([])
        
    for train_u, train_t in data_train:
        plot_train_u = np.concatenate((plot_train_u, train_u.detach().numpy().flatten()), axis = None)
        plot_train_t = np.concatenate((plot_train_t, train_t.detach().numpy().flatten()), axis = None)
        
    
    plt.title('True u and predicted u')
    _, plot_true_u = zip(*sorted(zip(plot_t, plot_true_u)))
    plot_t, plot_u = zip(*sorted(zip(plot_t, plot_u)))
    #Plot true and model predicted values
    plt.plot(plot_t, plot_true_u, label = 'true u')
    plt.plot(plot_t, plot_u, label = 'Network prediction')
    #plot trained points
    plt.scatter(plot_train_t, plot_train_u, label = 'Train points', linewidths=0.5, marker = '*')
    
    plt.ylabel('u(t)')
    plt.xlabel('time')
    
    plt.legend()
    plt.savefig('./Network prediction.png')
