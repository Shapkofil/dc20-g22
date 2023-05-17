import torch
import torch.nn as nn
from dvclive import Live

import shutil
from typing import Tuple

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
              device)->Tuple[float, float]:
      # validation step
      model.eval()
      # Loop over validation dataset
      valid_loss = 0.0 
      total = 0
      correct = 0
      for x, y in testloader:
        with torch.no_grad():
          x, y = x.to(device), y.squeeze().to(device)
          preds = model(x).squeeze()
          error = criterion(preds, y)
          valid_loss += error.item()
          _, predicted = torch.max(preds.data, 1)
          total += y.size(0)
          correct += (preds == y).sum().item()
      valid_loss = valid_loss / len(testloader)
      acc = total / correct
      return valid_loss, acc
          

def train_test_loop(model: nn.Module,
                    n_epochs,
                    trainloader,
                    testloader,
                    optimizer,
                    criterion,
                    device,
                    args = None):

    # Loop over epochs
    live = Live()
    best_acc = 0
    for epoch in range(n_epochs):
      train_loss, valid_loss = 0.0, 0.0
      epoch_loss = train_loop(model,
                              trainloader,
                              optimizer,
                              criterion,
                              device)
      live.log("train_loss", epoch_loss)

      valid_loss, acc = test_loop(model,
                                  testloader,
                                  criterion,
                                  device)
      live.log("test_loss", valid_loss)
      live.log("acc", acc)

      # Save model checkpoint
      is_best = acc > best_acc
      best_acc = max(acc, best_acc)

      save_checkpoint({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'best_acc1': best_acc,
          'optimizer' : optimizer.state_dict(),
      }, is_best)

      live.next_step()

    
