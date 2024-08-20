import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from logger import Logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        opt: argparse.Namespace,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        logger: Logger,
    ):
        """初始化訓練器"""
        self.epochs_run = 0
        # 解析 opt 的必要參數
        self.lr = opt.lr
        self.device = opt.device
        self.eta_min = opt.eta_min
        self.max_epochs = opt.max_epochs
        self.log_dir = opt.log_dir

        self.model = model.to(self.device)  # 模型
        self.train_loader = train_loader  # 訓練資料
        self.val_loader = val_loader  # 驗證資料
        self.test_loader = test_loader  # 測試資料

        # self.criterion = nn.SmoothL1Loss()  # 損失函數
        self.criterion = nn.MSELoss()  # 損失函數
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-5,
        )
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=self.eta_min, last_epoch=-1
        )
        # self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode="min", factor=0.9, patience=5
        # )
        # 指標計算器
        self.mse_metric = torchmetrics.MeanSquaredError().to(self.device)
        self.r2_metric = torchmetrics.R2Score().to(self.device)
        self.mae_metric = torchmetrics.MeanAbsoluteError().to(self.device)

        self.writer = logger

    def log_metrics(self, prefix: str, epoch: int):
        self.writer.add_scalar(f"{prefix}/MSE", self.mse_metric.compute().item(), epoch)
        self.writer.add_scalar(f"{prefix}/R2", self.r2_metric.compute().item(), epoch)
        self.writer.add_scalar(f"{prefix}/MAE", self.mae_metric.compute().item(), epoch)

        self.mse_metric.reset()
        self.r2_metric.reset()
        self.mae_metric.reset()

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"[{self.device}] Train Epoch {epoch:2d}")
        for src, tgt_hist, tgt in pbar:
            src, tgt = src.to(self.device), tgt.to(self.device)
            tgt_hist = tgt_hist.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(src, tgt_hist)
            loss = self.criterion(outputs, tgt)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            self.mse_metric(outputs, tgt)
            self.mae_metric(outputs, tgt)
            self.r2_metric(outputs.view(-1), tgt.view(-1))

        self.writer.add_scalar("Train/Loss", total_loss, epoch)
        self.writer.add_scalar(
            "Learning Rate", self.optimizer.param_groups[0]["lr"], epoch
        )
        self.log_metrics("Train", epoch)

    def val_epoch(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"[{self.device}] Val Epoch {epoch:2d}")
            for src, tgt_hist, tgt in pbar:
                src, tgt = src.to(self.device), tgt.to(self.device)
                tgt_hist = tgt_hist.to(self.device)
                outputs = self.model(src, tgt_hist)
                loss = self.criterion(outputs, tgt)
                total_loss += loss.item()

                self.mse_metric(outputs, tgt)
                self.mae_metric(outputs, tgt)
                self.r2_metric(outputs.view(-1), tgt.view(-1))

        self.writer.add_scalar("Val/Loss", total_loss, epoch)
        self.log_metrics("Val", epoch)

    def test_epoch(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f"[{self.device}] Test Epoch {epoch:2d}")
            for src, tgt_hist, tgt in pbar:
                src, tgt = src.to(self.device), tgt.to(self.device)
                tgt_hist = tgt_hist.to(self.device)
                outputs = self.model(src, tgt_hist)
                loss = self.criterion(outputs, tgt)
                total_loss += loss.item()

                self.mse_metric(outputs, tgt)
                self.mae_metric(outputs, tgt)
                self.r2_metric(outputs.view(-1), tgt.view(-1))

        # self.lr_scheduler.step(total_loss)

        self.writer.add_scalar("Test/Loss", total_loss, epoch)
        self.log_metrics("Test", epoch)

    def run(self):
        """主訓練循環"""
        for epoch in range(self.max_epochs):
            self.train_epoch(epoch)  # 訓練一個epoch
            self.lr_scheduler.step()  # 更新學習率調整器
            self.val_epoch(epoch)
            self.test_epoch(epoch)
