from data.exp_decay import ExpDecay
from data.coupled_oscillators import CoupOsc
import torch
import torch.nn as nn

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

