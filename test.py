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
    # raw data
    grid_size = 64
    test_time = 1
    
    test_equation = Nonlinear(C = 1, grid_size = 101)
    u, x, y, t = test_equation.get_points_at_time(t = test_time)
    test_u = torch.tensor(u, dtype = torch.float).view(-1, 1)
    test_x = torch.tensor(x, dtype = torch.float).view(-1, 1)
    test_y = torch.tensor(y, dtype = torch.float).view(-1, 1)
    test_t = torch.tensor(t, dtype = torch.float).view(-1, 1)
    test_const = test_equation.C
    test_dataset = TensorDataset(test_u, test_x, test_y, test_t)

    
    g_cpu = torch.Generator()
    g_cpu = g_cpu.manual_seed(42)

    test_set = test_dataset
    
    batch_size = 64
    data_test = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # data to train val test section
    model = PINN(input_size=3, number_of_hidden_layers=4, hidden_layer_size=64)
    model_dict = torch.load('./nn_models/model_gridsize_64_epochs_1')
    model.load_state_dict(model_dict['model_state_dict'])
    model.to(device)

    criterion_mse = nn.MSELoss()
    
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
        
    plot_u = plot_u.reshape(test_equation.x_dots, test_equation.y_dots)
    plot_x = plot_x.reshape(test_equation.x_dots, test_equation.y_dots)
    plot_y = plot_y.reshape(test_equation.x_dots, test_equation.y_dots)
    
    #Plot solution from model
    
    plt.figure(figsize=(8, 6))
    plt.contourf(plot_x, plot_y, plot_u, levels=100, cmap='viridis')
    plt.colorbar(label="u(x, y, t)")
    plt.title(f"Heatmap предсказанного решения при t = {test_time}", fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.grid(False)
    
    #plt.legend()
    plt.savefig(f'./Network predictions/Network prediction nonlinear grid size {grid_size} time {test_time}.png')
    
    #Plot error 
    plot_true_u = plot_true_u.reshape(test_equation.x_dots, test_equation.y_dots)
    plot_u = np.abs(plot_u - plot_true_u)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(plot_x, plot_y, plot_u, levels=100, cmap='binary')
    plt.colorbar(label="u(x, y, t)")
    plt.title(f"Heatmap ошибки решения при t = {test_time}", fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.grid(False)
    
    #plt.legend()
    plt.savefig(f'./Network predictions/Network error nonlinear grid size {grid_size} time {test_time}.png')
    
    #Real data plot
    plt.figure(figsize=(8, 6))
    plt.contourf(plot_x, plot_y, plot_true_u, levels=100, cmap='viridis')
    plt.colorbar(label="u(x, y, t)")
    plt.title(f"Heatmap аналитического решения при t = {test_time}", fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.grid(False)
    
    #plt.legend()
    plt.savefig(f'./real data nonlinear at time {test_time}.png')