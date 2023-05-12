import torch
import torch.nn as nn

def train_loop(model: nn.Module,
               trainloader,
               optimizer,
               criterion,
               device
               ):

    # train step
    model.train()
    train_loss = 0
    # Loop over train dataset
    for x, y in trainloader:
        optimizer.zero_grad()
        # move inputs to device
        x = x.to(device)
        y  = y.squeeze().to(device)
        # Forward Pass
        preds = model(x).squeeze()
        loss = criterion(preds, y) # compute batch loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_loss = train_loss / len(trainloader)
    return epoch_loss
      

def test_loop(model,
              testloader,
              criterion,
              valid_loss,
              device,
              v_losses):
      # validation step
      model.eval()
      # Loop over validation dataset
      for x, y in testloader:
        with torch.no_grad():
          x, y = x.to(device), y.squeeze().to(device)
          preds = model(x).squeeze()
          error = criterion(preds, y)
        valid_loss += error.item()
      valid_loss = valid_loss / len(testloader)
      v_losses.append(valid_loss)
          

def train_test_loop(model: nn.Module,
                    n_epochs,
                    trainloader,
                    testloader,
                    optimizer,
                    criterion,
                    device):

    # Lists to store training and validation losses
    t_losses, v_losses = [], []
    # Loop over epochs
    for epoch in range(n_epochs):
      train_loss, valid_loss = 0.0, 0.0
      epoch_loss = train_loop(model,
                              trainloader,
                              optimizer,
                              criterion,
                              device)

      print(f'{epoch} - train: {epoch_loss}, valid: {valid_loss}')

      v_losses.append(valid_loss)
      plot_losses(t_losses, v_losses)



      

    
