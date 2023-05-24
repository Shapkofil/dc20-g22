import torch
import torch.nn as nn
from tqdm import tqdm

import shutil
from typing import Tuple, Dict

from dc2_g22.lstm.dataset_loaders import BatchSampler, BatchLoader

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from dvclive import Live
	

def save_checkpoint(state, is_best, filename='model_checkpoints/checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_checkpoints/model_best.pth')


def log_experiment(
        live: Live,
        args: Dict[str, float],
        prefix:str = "train") -> None:
    """
    Logs an experiment
    """

    for key, value in args.items():
      live.log_metric(f"{prefix}/{key}", value)

def train_loop(model: nn.Module,
               trainloader,
               optimizer,
               criterion,
               device
               ) -> Dict[str, float]:

    # train step
    model.train()
    train_loss = 0
    # Loop over train dataset
    for x, y in tqdm(trainloader, desc="Training"):
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

    result = {}
    result["loss"] = train_loss / len(trainloader)
    return result
      

def test_loop(model,
              testloader,
              criterion,
              device)->Dict[str, float]:
      # validation step
      model.eval()
      # Loop over validation dataset
      valid_loss = 0.0 
      y_true = []
      y_pred = []


      for x, y in tqdm(testloader, desc="Testing"):
          with torch.no_grad():
              x, y = x.to(device), y.squeeze().to(device)
              preds = model(x)
              error = criterion(preds, y)
              valid_loss += error.item()
              _, predicted = torch.max(preds.data, 1)
              y_true.extend(y.cpu().numpy())
              y_pred.extend(preds.cpu().numpy())


      result = {}
      result["loss"] = valid_loss / len(testloader)
      result["mse"] = mean_squared_error(y_true, y_pred)
      result["r2"] = r2_score(y_true, y_pred)
      result["percentage_error"] = mean_absolute_percentage_error(y_true, y_pred)
      
      return result
          

def train_test_loop(model: nn.Module,
                    n_epochs: int,
                    sampler: BatchSampler,
                    optimizer,
                    criterion,
                    live: Live,
                    device,
                    args = None):

    live.log_param("epochs", n_epochs)
    best_loss = 1_000_000
    trainloader = BatchLoader(sampler)
    testloader = BatchLoader(sampler, test=True)
    for epoch in range(n_epochs):
        train_loss, valid_loss = 0.0, 0.0
        print(f"Epoch {epoch}:")
        train_metrics = train_loop(model,
                                trainloader,
                                optimizer,
                                criterion,
                                device)
        log_experiment(live, train_metrics)

        metrics = test_loop(model,
                            testloader,
                            criterion,
                            device)
        log_experiment(live, metrics, "test")

        # Save model checkpoint
        is_best = metrics["loss"] < best_loss
        best_acc = max(metrics["loss"], best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

        live.next_step()
    live.log_artifact('model_checkpoints/model_best.pth',
                      type="model")

    
