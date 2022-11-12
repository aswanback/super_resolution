
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import lr_scheduler, Adam
import random
from tqdm.auto import tqdm
import copy
from metrics import AverageMeter, compute_metrics
from typing import NamedTuple, Tuple, Union

class EvalPrediction(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    labels: np.ndarray

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        epochs: int,
        batch_size: int,
        device,
        weights_name = None,
        state_dict_file = None,
    ):
        seed = 123
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.learning_rate = 1e-4
        self.gamma = 0.5
        self.device = device

        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights_name = weights_name

        self.best_epoch = 0
        self.best_metric = 0.0
        if state_dict_file is not None:
          state_dict = torch.load(state_dict_file, map_location="cpu")
          self.model.load_state_dict(state_dict, strict=False)

    def train(self):
        epochs_trained = 0
        
        train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        step_size = int(len(self.train_dataset) / self.batch_size * 200)
        
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=self.gamma)

        for epoch in range(epochs_trained, self.epochs):
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate * (0.1 ** (epoch // int(self.epochs * 0.8)))

            self.model.train()
            epoch_losses = AverageMeter()

            with tqdm(total=(len(self.train_dataset) - len(self.train_dataset) % self.batch_size)) as t:
                t.set_description(f'epoch: {epoch}/{self.epochs - 1}')

                for data in train_dataloader:
                    inputs, labels = data

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    preds = self.model(inputs)
                    criterion = nn.L1Loss()
                    loss = criterion(preds, labels)

                    epoch_losses.update(loss.item(), len(inputs))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    t.set_postfix(loss=f'{epoch_losses.avg:.6f}')
                    t.update(len(inputs))

            self.eval(epoch)

    def eval(self, epoch):

        eval_dataloader = DataLoader(dataset=self.eval_dataset, batch_size=1,)
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()

        self.model.eval()

        for data in eval_dataloader:
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                preds = self.model(inputs)

            metrics = compute_metrics(EvalPrediction(predictions=preds, labels=labels), scale=self.model.upsample_factor)

            epoch_psnr.update(metrics['psnr'], len(inputs))
            epoch_ssim.update(metrics['ssim'], len(inputs))

        print(f'scale:{str(self.model.upsample_factor)}      eval psnr: {epoch_psnr.avg:.2f}     ssim: {epoch_ssim.avg:.4f}')

        if epoch_psnr.avg > self.best_metric:
            self.best_epoch = epoch
            self.best_metric = epoch_psnr.avg

            print(f'best epoch: {epoch}, psnr: {epoch_psnr.avg:.6f}, ssim: {epoch_ssim.avg:.6f}')
            self.save_model()
          
    def save_model(self):
        
        weights = copy.deepcopy(self.model.state_dict())
        torch.save(weights, self.weights_name)
