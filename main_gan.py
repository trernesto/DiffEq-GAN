from data.exp_decay import ExpDecay
from data.coupled_oscillators import CoupOsc
from models.deqgan import Generator, Discriminator

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt

# Loss Section
def diff_t(network: nn.Module, C:torch.tensor, t: torch.tensor):
    input = torch.cat((C, t) , dim = 1)
    u = network(input)
    du_dt = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    #du_dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    return du_dt

# Main section
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_exp_decay = ExpDecay(C = 1, N = 121, a = 0, b = 10)
    train_exp_decay.set_points_to_random(C_start=1, C_finish=1)
    train_u = torch.tensor(train_exp_decay.x, dtype = torch.float).view(-1, 1)
    train_t = torch.tensor(train_exp_decay.t, dtype = torch.float).view(-1, 1)
    train_C = torch.tensor(train_exp_decay.C, dtype = torch.float).view(-1, 1)
    #exp_decay_const = train_exp_decay.C
    train_dataset = TensorDataset(train_u, train_t, train_C)

    # raw data
    exp_decay = ExpDecay(C = 1, N = 101)
    test_u = torch.tensor(exp_decay.x, dtype = torch.float).view(-1, 1)
    test_t = torch.tensor(exp_decay.t, dtype = torch.float).view(-1, 1)
    exp_decay_const = exp_decay.C
    test_dataset = TensorDataset(test_u, test_t)

    
    g_cpu = torch.Generator()
    g_cpu = g_cpu.manual_seed(42)
    
    train_set, val_set = torch.utils.data.random_split(dataset=train_dataset, lengths=[0.9, 0.1], generator = g_cpu)
    
    test_set = test_dataset
    
    batch_size = 64
    data_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    data_val = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    data_test = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    #exp_decay.plot()

    #coup_osc = CoupOsc(C = 1)
    #coup_osc.plot()

    # data to train val test section

    model_generator = Generator(input_size=2, number_of_hidden_layers=2, hidden_layer_size=16)
    model_discriminator = Discriminator(number_of_hidden_layers=2, hidden_layer_size=20)
    model_generator.to(device)
    model_discriminator.to(device)
    
    
    opt_generator = torch.optim.Adam(model_generator.parameters(), lr = 3e-4)
    opt_discriminator = torch.optim.Adam(model_discriminator.parameters(), lr = 3e-4)

    criterion = nn.BCELoss()

    #scheduler = ...
    

    epochs = 5000
    print_epoch = 500
    
    losses_generator_train = []
    losses_generator_val = []
    
    losses_discriminator_train = []
    losses_discriminator_val = []    
    
    for epoch in range(epochs):
        mean_loss_generator_batch = 0
        mean_loss_discriminator_batch = 0
        for batch_u, batch_t, batch_C in data_train:
            
            #Discriminator loss
            #Real part
            opt_discriminator.zero_grad()
            
            RHS = torch.zeros_like(batch_u) + torch.rand(1) * 1e-8
            model_discriminator_out_true = model_discriminator(RHS)
            
            ones = torch.ones_like(model_discriminator_out_true)
            loss_discriminator_real = criterion(model_discriminator_out_true, ones)
            
            loss_discriminator_real.backward()
            #fake part
            #MODEL
            bC = Variable(batch_C, requires_grad=False)
            bt = Variable(batch_t + torch.rand(1) * 1e-6, requires_grad=True)
            input = torch.cat((bC, bt), dim = 1)
            LHS = model_generator(input) + diff_t(model_generator, C=bC, t=bt)
            #model_discriminator generated loss
            model_discriminator_out_g = model_discriminator(LHS.detach())
            
            zeros = torch.zeros_like(model_discriminator_out_g)
            loss_discriminator_fake = criterion(model_discriminator_out_g, zeros)
            
            loss_discriminator_fake.backward()
            
            loss_discriminator = loss_discriminator_fake + loss_discriminator_real
            opt_discriminator.step()

            #generator loss section
            opt_generator.zero_grad()
            
            model_discriminator_out_g = model_discriminator(LHS)
            ones = torch.ones_like(model_discriminator_out_g)
            loss_generator = criterion(model_discriminator_out_g, ones)
            
            loss_generator.backward()
            opt_generator.step()
            
            mean_loss_generator_batch += loss_generator/len(data_train)
            mean_loss_discriminator_batch += loss_discriminator/len(data_train)
        
        if (epoch + 1) % print_epoch == 0:
            print(f'epoch: {epoch + 1}')
            print('train generator loss:', loss_generator.item())
            print('train discriminator loss:', loss_discriminator.item())
            
        losses_generator_train.append(loss_generator.item())
        losses_discriminator_train.append(loss_discriminator.item())

        for batch_u, batch_t, batch_C in data_val:
            #Discriminator loss
            #Real part
            
            RHS = torch.zeros_like(batch_u)
            model_discriminator_out_true = model_discriminator(RHS)
            
            ones = torch.ones_like(model_discriminator_out_true)
            loss_discriminator_real = criterion(model_discriminator_out_true, ones)
            
            loss_discriminator_real.backward()
            #fake part
            #MODEL
            bC = Variable(batch_C, requires_grad=False)
            bt = Variable(batch_t, requires_grad=True)
            input = torch.cat((bC, bt), dim = 1)
            LHS = model_generator(input) + diff_t(model_generator, C=bC, t=bt)
            #model_discriminator generated loss
            model_discriminator_out_g = model_discriminator(LHS.detach())
            
            zeros = torch.zeros_like(model_discriminator_out_g)
            loss_discriminator_fake = criterion(model_discriminator_out_g, zeros)
            
            loss_discriminator_fake.backward()
            
            loss_discriminator = loss_discriminator_fake + loss_discriminator_real

            #generator loss section
            
            model_discriminator_out_g = model_discriminator(LHS)
            ones = torch.ones_like(model_discriminator_out_g)
            loss_generator = criterion(model_discriminator_out_g, ones)

        if (epoch + 1) % print_epoch == 0:    
            print('val generator loss:', loss_generator.item())
            print('val discriminator loss:', loss_discriminator.item())
            
        losses_generator_val.append(loss_generator.item())
        losses_discriminator_val.append(loss_discriminator.item())

    # Test secti
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    fig.suptitle('Loss function')

    x = np.arange(epochs)

    axs[0, 0].set_title('generator loss')
    axs[0, 0].plot(x, losses_generator_train, label = 'train')
    axs[0, 0].plot(x, losses_generator_val, label = 'val')
    axs[0, 0].legend()

    
    axs[0, 1].set_title('discriminator loss')
    axs[0, 1].plot(x, losses_discriminator_train, label = 'train')
    axs[0, 1].plot(x, losses_discriminator_val, label = 'val')
    axs[0, 1].legend()

    
    axs[1, 0].set_title('Train losses')
    axs[1, 0].plot(x, losses_generator_train, label = 'train generator')
    axs[1, 0].plot(x, losses_discriminator_train, label = 'train discriminator')
    axs[1, 0].legend()
    
    axs[1, 1].set_title('Val losses')
    axs[1, 1].plot(x, losses_generator_val, label = 'val generator')
    axs[1, 1].plot(x, losses_discriminator_val, label = 'val discriminator')
    axs[1, 1].legend()



    plt.savefig('./losses_gan.png')
    plt.close()

    plt.clf()
    plt.cla()
    
    plot_true_u = np.array([])
    plot_u = np.array([])
    plot_t = np.array([])

    for true_u, t in data_test:
        c = exp_decay_const * torch.ones_like(t)
        input = torch.cat((c, t), dim = 1)
        u = model_generator(input)
        plot_true_u = np.concatenate((plot_true_u, true_u.detach().numpy().flatten()), axis = None)
        plot_u = np.concatenate((plot_u, u.detach().numpy().flatten()), axis = None)
        plot_t = np.concatenate((plot_t, t.detach().numpy().flatten()), axis = None)
        
    plot_train_u = np.array([])
    plot_train_t = np.array([])
        
    for train_u, train_t, _ in data_train:
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
