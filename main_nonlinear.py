from data.nonlinear_eq_1 import Nonlinear

from models.pinn import PINN

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt

# Loss Section
def diff_t(network: nn.Module, x:torch.tensor, y:torch.tensor, t: torch.tensor):
    input = torch.cat((x, y, t) , dim = 1)
    u = network(input)
    du_dt = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    return du_dt

def diff_xx(network: nn.Module, x:torch.tensor, y:torch.tensor, t: torch.tensor):
    input = torch.cat((x, y, t) , dim = 1)
    u = network(input)
    du_dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    du_dx = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0]
    return du_dx

def diff_yy(network: nn.Module, x:torch.tensor, y:torch.tensor, t: torch.tensor):
    input = torch.cat((x, y, t) , dim = 1)
    u = network(input)
    du_dy = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    du_dy = torch.autograd.grad(du_dy.sum(), y, create_graph=True)[0]
    return du_dy

# Main section
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_equation = Nonlinear()
    #train_exp_decay.set_points_to_random(C_start=0, C_finish=4)
    train_u = torch.tensor(train_equation.u, dtype = torch.float).view(-1, 1)
    train_x = torch.tensor(train_equation.x, dtype = torch.float).view(-1, 1)
    train_y = torch.tensor(train_equation.y, dtype = torch.float).view(-1, 1)
    train_t = torch.tensor(train_equation.t, dtype = torch.float).view(-1, 1)
    train_const = train_equation.C
    train_dataset = TensorDataset(train_u, train_x, train_y, train_t)
    

    # raw data
    test_equation = Nonlinear(C = 1, grid_size = 101)
    u, x, y, t = test_equation.get_points_at_time(t = 0)
    test_u = torch.tensor(u, dtype = torch.float).view(-1, 1)
    test_x = torch.tensor(x, dtype = torch.float).view(-1, 1)
    test_y = torch.tensor(y, dtype = torch.float).view(-1, 1)
    test_t = torch.tensor(t, dtype = torch.float).view(-1, 1)
    test_const = test_equation.C
    test_dataset = TensorDataset(test_u, test_x, test_y, test_t)

    
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

    model = PINN(input_size=3, number_of_hidden_layers=3, hidden_layer_size=16)
    model.to(device)
    opt = torch.optim.Adam(model.parameters())

    criterion_mse = nn.MSELoss()

    #scheduler = ...
    

    epochs = 25
    print_epoch = 5
    
    losses_train = []
    losses_val = []
    losses_mse_train = []
    losses_mse_val = []
    losses_phys_train = []
    losses_phys_val = []    
    
    print('Training started')
    for epoch in range(epochs):
        mean_loss_batch = 0
        mean_loss_mse = 0
        mean_loss_phys = 0
        for batch_u, batch_x, batch_y, batch_t in data_train:
            opt.zero_grad()
            input = torch.cat((batch_x, batch_y, batch_t), dim = 1)
            model_out = model(input)

            #mse loss section
            loss_mse = criterion_mse(model_out, batch_u)

            #pde loss section
            bx = Variable(batch_x, requires_grad=True)
            by = Variable(batch_y, requires_grad=True)
            bt = Variable(batch_t, requires_grad=True)
            
            dudt = diff_t(model, bx, by, bt)
            dudxx = diff_xx(model, bx, by, bt)
            dudyy = diff_yy(model, bx, by, bt)

            eq = dudt * model_out ** 2 - dudxx + dudyy + model_out ** 3 
            zeros = torch.zeros_like(eq)
            loss_ph = criterion_mse(eq, zeros)

            #Boundary conditions
            #x_right_boundary = torch.tensor([20], dtype = torch.float, requires_grad =  True)
            #c_right_boundary = torch.tensor([np.random.uniform(0, 10)], dtype = torch.float, requires_grad = True)
            #input = torch.cat((c_right_boundary, x_right_boundary), dim = 0)
            #u_predicted_boundary = model(input)
            #You can use this, but lim b->inf -> u_true -> 0
            #u_true = train_exp_decay.equation(x_right_boundary)
            #u_true = torch.zeros_like(u_predicted_boundary)
            
            #loss_boundary = criterion_mse(u_predicted_boundary, u_true)

            loss = loss_mse + loss_ph #+ loss_boundary
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
        for batch_u, batch_x, batch_y, batch_t in data_val:
            input = torch.cat((batch_x, batch_y, batch_t), dim = 1)
            model_out = model(input)

            #mse loss section
            loss_mse = criterion_mse(model_out, batch_u)

            #pde loss section
            bx = Variable(batch_x, requires_grad=True)
            by = Variable(batch_y, requires_grad=True)
            bt = Variable(batch_t, requires_grad=True)
            
            dudt = diff_t(model, bx, by, bt)
            dudxx = diff_xx(model, bx, by, bt)
            dudyy = diff_yy(model, bx, by, bt)

            eq = dudt * model_out ** 2 - dudxx + dudyy + model_out ** 3 
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


    plt.savefig('./losses_nonlinear.png')
    plt.close()

    plt.clf()
    plt.cla()
    
    plot_true_u = np.array([])
    plot_u = np.array([])
    plot_x = np.array([])
    plot_y = np.array([])
    plot_t = np.array([])
    loss_test = 0

    for true_u, x, y, t in data_test:
        c = test_const * torch.ones_like(t)
        input = torch.cat((x, y, t), dim = 1)
        u = model(input)
        plot_true_u = np.concatenate((plot_true_u, true_u.detach().numpy().flatten()), axis = None)
        loss_test += criterion_mse(u, true_u)
        plot_u = np.concatenate((plot_u, u.detach().numpy().flatten()), axis = None)
        plot_x = np.concatenate((plot_x, x.detach().numpy().flatten()), axis = None)
        plot_y = np.concatenate((plot_y, y.detach().numpy().flatten()), axis = None)
        plot_t = np.concatenate((plot_t, t.detach().numpy().flatten()), axis = None)
        
    print('loss test:', loss_test)
        
    
    # for training points visualization
    plot_train_u = np.array([])
    plot_train_x = np.array([])
    plot_train_y = np.array([])
    plot_train_t = np.array([])
        
    for train_u, train_x, train_y, train_t in data_train:
        plot_train_u = np.concatenate((plot_train_u, train_u.detach().numpy().flatten()), axis = None)
        plot_train_x = np.concatenate((plot_train_x, train_x.detach().numpy().flatten()), axis = None)
        plot_train_y = np.concatenate((plot_train_y, train_y.detach().numpy().flatten()), axis = None)
        #plot_train_t = np.concatenate((plot_train_t, train_t.detach().numpy().flatten()), axis = None)
        
    plot_u = plot_u.reshape(test_equation.x_dots, test_equation.y_dots)
    plot_x = plot_x.reshape(test_equation.x_dots, test_equation.y_dots)
    plot_y = plot_y.reshape(test_equation.x_dots, test_equation.y_dots)
    
    #Plot solution from model
    
    plt.figure(figsize=(8, 6))
    plt.contourf(plot_x, plot_y, plot_u, levels=100, cmap='viridis')
    plt.colorbar(label="u(x, y, t)")
    plt.title(f"Heatmap аналитического решения при t = {0}", fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.grid(False)
    
    #plt.legend()
    plt.savefig('./Network prediction nonlinear.png')
    
    #Plot error 
    plot_true_u = plot_true_u.reshape(test_equation.x_dots, test_equation.y_dots)
    plot_u = np.abs(plot_u - plot_true_u)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(plot_x, plot_y, plot_u, levels=100, cmap='binary')
    plt.colorbar(label="u(x, y, t)")
    plt.title(f"Heatmap аналитического решения при t = {0}", fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.grid(False)
    
    #plt.legend()
    plt.savefig('./Network error nonlinear.png')
    
    #Real data plot
    plt.figure(figsize=(8, 6))
    plt.contourf(plot_x, plot_y, plot_true_u, levels=100, cmap='viridis')
    plt.colorbar(label="u(x, y, t)")
    plt.title(f"Heatmap аналитического решения при t = {0}", fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.grid(False)
    
    #plt.legend()
    plt.savefig('./real data nonlinear.png')
    
    torch.save( {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
    }, './model')