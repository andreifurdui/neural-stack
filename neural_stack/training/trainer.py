import os

import torch
import torch.nn as nn
import wandb

from tqdm import tqdm

class Trainer:
    def __init__(
            self, 
            model,
            dataloader_train,
            dataloader_val,
            optimizer,
            criterion,
            lr_scheduler
            ):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler

    def train(self, device, wandb_logger, num_epochs):
        best_val_accuracy = 0.0

        for epoch in tqdm(range(num_epochs), total=num_epochs):
            self.train_epoch(
                epoch=epoch,
                device=device,
                wandb_logger=wandb_logger
            )

            val_loss, val_accuracy = self.validate(
                device=device
            )

            print(f"Epoch {epoch+1}/{num_epochs}: Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), os.path.join(wandb_logger.dir, "best.pth"))
                print(f"New best model saved with Val Acc={best_val_accuracy:.4f}")

            wandb_logger.log({
                "val/loss": val_loss,
                "val/accuracy": val_accuracy,
            }, step=(epoch + 1) * len(self.dataloader_train))

        wandb_logger.save(os.path.join(wandb_logger.dir, "best.pth"))

    def train_epoch(self, epoch, device, wandb_logger: wandb.Run, print_freq = 50):
        self.model.train()

        for idx, (img, target) in enumerate(self.dataloader_train):
            img = img.to(device)
            target = target.to(device)

            pred = self.model(img)
            loss = self.criterion(pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            accuracy = (pred.argmax(dim=1) == target).float().mean().item()

            wandb_logger.log({
                "train/loss": loss.item(),
                "train/accuracy": accuracy,
                "train/lr": self.optimizer.param_groups[0]['lr'],
            }, step=epoch * len(self.dataloader_train) + idx)
                
        self.lr_scheduler.step()

    def validate(self, device):
        self.model.eval()

        loss_avg = 0.0
        accuracy_avg = 0.0

        for img, target in self.dataloader_val:
            with torch.no_grad():
                img = img.to(device)
                target = target.to(device)

                pred = self.model(img)
                loss = self.criterion(pred, target)

                accuracy = (pred.argmax(dim=1) == target).float().mean().item()

                loss_avg += loss.item()
                accuracy_avg += accuracy
        
        loss_avg /= len(self.dataloader_val)
        accuracy_avg /= len(self.dataloader_val)

        return loss_avg, accuracy_avg
