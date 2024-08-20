import matplotlib.pyplot as plt
import numpy as np
import os


class Plotter:
    def __init__(self, model, trainer, train_loader, val_loader, test_loader, opt):
        self.model = model
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = opt.log_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir + "/plot"):
            os.makedirs(self.output_dir + "/plot")

    def plot_predictions(self, mode="train"):
        if mode == "train":
            data_loader = self.train_loader
        elif mode == "val":
            data_loader = self.val_loader
        elif mode == "test":
            data_loader = self.test_loader
        else:
            raise ValueError("Invalid mode. Expected 'train', 'val', or 'test'.")

        for i, (src, hist, tgt) in enumerate(data_loader):
            src, hist, tgt = (
                src.to(self.trainer.device),
                hist.to(self.trainer.device),
                tgt.to(self.trainer.device),
            )
            outputs = self.model(src, hist)

            src = src.cpu().numpy()
            hist = hist.cpu().numpy()
            tgt = tgt.cpu().numpy()
            outputs = outputs.cpu().detach().numpy()

            for j in range(3):
                # tgt and outputs
                random_idx = np.random.randint(0, len(src))
                t_past = len(src[random_idx, :, 0])
                t_future = len(tgt[random_idx, :, 0])
                plt.plot(
                    range(t_past),
                    src[random_idx, :, 0],
                    label="src",
                )
                plt.plot(
                    range(t_past - 1, t_past + t_future),
                    np.concatenate([[src[random_idx, -1, 0]], tgt[random_idx, :, 0]]),
                    label="tgt",
                )
                plt.plot(
                    range(t_past - 1, t_past + t_future),
                    np.concatenate(
                        [[src[random_idx, -1, 0]], outputs[random_idx, :, 0]]
                    ),
                    label="outputs",
                )
                # y 軸的範圍
                plt.ylim(-4, 4)
                plt.legend()
                # Save the plot as an image file
                output_file = os.path.join(
                    self.output_dir, "plot", f"{mode}_plot_{i}_{j}.png"
                )
                plt.savefig(output_file)
                plt.close()  # Close the plot to free memory
