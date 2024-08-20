import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class PH_Dataset(Dataset):
    def __init__(
        self,
        root: str,
        history_steps: int = 1,
        forecast_steps: int = 1,
        seed: int = 42,
        mode: str = "train",  # 可以是 'train', 'val', 或 'test'
        test_ratio: float = 0.1,
        val_ratio: float = 0.2,
    ):
        self.root = root
        self.history_steps = history_steps
        self.forecast_steps = forecast_steps
        self.seed = seed
        self.mode = mode
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.x_cols = [
            "放流槽採樣箱1 pH",
            "酸鹼pH反應槽1 pH計",
            "C-B3F-1103 酸鹼pH計1(清洗比對)",
            "T-1102硫酸供應泵 流量計 實際值",
            "T-1103硫酸供應泵 流量計 實際值",
            "RO-CIP Tank 流量A NaOH",
            "RO-CIP Tank 流量B HCL",
            "酸鹼pH反應槽1(T-1102) 自動閥2 狀態",
            "酸鹼pH反應槽2(T-1103) 自動閥2 狀態",
            "F5UPW.UPW_CB1F_ACF_T_ABC",
            "F5UPW.UPW_CB1F_DB_P_ABC",
            "F5UPW2.UPW2-CB2F-2B3T_123",
        ]
        self.y_cols = ["放流槽採樣箱1 pH"]
        self.x_scaler, self.y_scaler = StandardScaler(), StandardScaler()

        self.data = pd.read_csv(root)
        self.data = self._cutoff(self.data, 37440)  # TODO: 超參數暫定內定
        self.data = self._fillna(self.data)

        # self.data["point_time"] = pd.to_datetime(self.data["point_time"])
        # self.data = self.data.set_index("point_time").resample("10T").mean()
        # self.data = self.data.reset_index()

        x_data = self.x_scaler.fit_transform(self.data[self.x_cols].values)
        y_data = self.y_scaler.fit_transform(self.data[self.y_cols].values)

        train_val_data, test_data = self._split_test_data(
            x_data, y_data, test_ratio=self.test_ratio
        )

        if mode == "test":
            self.x_data, self.y_data = test_data
            self.samples = self._sliding_window(
                self.x_data, self.y_data, history_steps, forecast_steps
            )
        else:
            self.x_data, self.y_data = train_val_data
            self.samples = self._sliding_window(
                self.x_data, self.y_data, history_steps, forecast_steps
            )
            train_samples, val_samples = train_test_split(
                self.samples, test_size=self.val_ratio, random_state=seed, shuffle=True
            )
            self.samples = train_samples if mode == "train" else val_samples

    @staticmethod
    def _cutoff(data, idx):
        return data[:idx]

    @staticmethod
    def _fillna(data):
        return data.interpolate(method="linear")

    @staticmethod
    def _split_test_data(x_data, y_data, test_ratio):
        test_size = int(len(x_data) * test_ratio)
        train_val_data_x, test_data_x = x_data[:-test_size], x_data[-test_size:]
        train_val_data_y, test_data_y = y_data[:-test_size], y_data[-test_size:]
        return (train_val_data_x, train_val_data_y), (test_data_x, test_data_y)

    def _sliding_window(self, x_data, y_data, history_steps, forecast_steps):
        samples = []
        for idx in range(len(x_data) - history_steps - forecast_steps + 1):
            x = x_data[idx : idx + history_steps]
            y_hist = y_data[
                idx + history_steps - forecast_steps : idx + history_steps
            ]  # 使用歷史步驟的 y
            y_target = y_data[
                idx + history_steps : idx + history_steps + forecast_steps
            ]
            samples.append((x, y_hist, y_target))
        return samples

    def inverse_transform(self, x, y):
        x = self.x_scaler.inverse_transform(x)
        y = self.y_scaler.inverse_transform(y)
        return x, y

    def to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y_hist, y_target = self.samples[idx]
        return self.to_tensor(x), self.to_tensor(y_hist), self.to_tensor(y_target)
