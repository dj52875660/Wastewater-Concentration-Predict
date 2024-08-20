import argparse

from dataset import PH_Dataset
from logger import Logger
from models import *
from opts import Opts
from torch.utils.data import DataLoader
from trainer import Trainer
from utils import seed_everything
from plot import Plotter


def main(opt: argparse.Namespace):
    seed_everything(opt.seed)  # 設定隨機種子
    logger = Logger(opt)  # 設定 logger
    train_dataset = PH_Dataset(
        root=opt.file_path,
        history_steps=opt.history_steps,
        forecast_steps=opt.forecast_steps,
        seed=opt.seed,
        mode="train",
    )
    val_dataset = PH_Dataset(
        root=opt.file_path,
        history_steps=opt.history_steps,
        forecast_steps=opt.forecast_steps,
        seed=opt.seed,
        mode="val",
    )
    test_dataset = PH_Dataset(
        root=opt.file_path,
        history_steps=opt.history_steps,
        forecast_steps=opt.forecast_steps,
        seed=opt.seed,
        mode="test",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False,
    )
    # lr = 1e-3
    # model = LSTMModel(
    #     input_size=len(train_dataset.x_cols),
    #     output_size=len(train_dataset.y_cols),
    #     forecast_steps=opt.forecast_steps,
    #     hidden_size=128,
    #     num_layers=2,
    # )
    model = Seq2Seq(
        input_size=len(train_dataset.x_cols),
        output_size=len(train_dataset.y_cols),
        hidden_size=512,
        num_layers=2,
        num_heads=4,
    )
    # model = ResSeq2Seq(
    #     input_size=len(train_dataset.x_cols),
    #     output_size=len(train_dataset.y_cols),
    #     hidden_size=512,
    #     num_layers=2,
    #     num_heads=4,
    # )
    # model = ResLSTMModel(
    #     input_size=len(train_dataset.x_cols),
    #     output_size=len(train_dataset.y_cols),
    #     forecast_steps=opt.forecast_steps,
    #     hidden_size=256,
    #     num_layers=1,
    # )
    # model = LSTMPeephole(
    #     input_size=len(train_dataset.x_cols),
    #     output_size=len(train_dataset.y_cols),
    #     forecast_steps=opt.forecast_steps,
    #     hidden_size=512,
    # )
    # model = LSTMPeepholebid(
    #     input_size=len(train_dataset.x_cols),
    #     output_size=len(train_dataset.y_cols),
    #     forecast_steps=opt.forecast_steps,
    #     hidden_size=512,
    #     num_layers=2,
    #     bidirectional=True,
    # )
    # model = ModernTCNModel(
    #     input_size=len(train_dataset.x_cols),
    #     output_size=len(train_dataset.y_cols),
    #     hidden_size=128,
    #     num_layers=2,
    # )
    # lr = 1e-4
    # model = TransformerModel(
    #     input_size=len(train_dataset.x_cols),
    #     output_size=len(train_dataset.y_cols),
    #     hidden_size=128,
    # )
    logger.write_model_summary(model)
    trainer = Trainer(
        opt=opt,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        logger=logger,
    )
    trainer.run()
    plotter = Plotter(model, trainer, train_loader, val_loader, test_loader, opt)
    plotter.plot_predictions(mode="test")
    plotter.plot_predictions(mode="train")
    logger.close()


if __name__ == "__main__":
    opt = Opts().parse(
        [
            "--history_steps",
            "60",
            "--forecast_steps",
            "30",
            "--gpu_id",
            "1",
            "--lr",
            "1e-3",
            "--max_epochs",
            "100",
        ]
    )
    main(opt)
