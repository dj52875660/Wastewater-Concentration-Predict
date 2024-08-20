# mx_ph

歡迎來到 `mx_ph` 專案，這裡提供了詳細的專案架構、初始化指南以及使用說明，旨在協助團隊成員快速熟悉並投入開發。

## 專案結構概覽

本專案採用清晰且模組化的結構，以支持靈活的開發與擴展。結構如下：

```
mx_ph/              # 根目錄
├── .vscode/                                # VSCode 設定，已預設 isort 和 black 格式化
├── mx_ph/          # 主要源碼
│   ├── data/                               # 數據目錄，隔離代碼和數據
│   ├── module/                             # 核心模組
│   │   ├── __init__.py                     # 模組初始化
│   │   ├── utils.py                        # 工具模組
│   ├── logger.py                           # 日誌管理
│   ├── main.py                             # 程序入口
│   ├── opts.py                             # 參數配置
│   └── utils.py                            # 通用工具函數
├── data/                                   # 專案數據存儲
├── notebook/                               # jupyter檔案，用數字-加功能，e.g.'1.0-jqp-initial-data-exploration`.
├── runs/                                   # 存放日誌
├── tests/                                  # 測試腳本集
├── .gitignore                              # Git 忽略設定
├── Makefile                                # 自動化指令集
├── README.md                               # 專案說明文檔
├── requirements.txt                        # 依賴包列表
├── setup.cfg                               # 安裝配置
├── setup.py                                # 安裝腳本
└── use_example.py                          # 使用示例
```

## 初始化指南

為了確保環境的一致性，請按照以下步驟初始化專案：

1. **克隆專案：**
   ```bash
   git clone [專案URL]
   ```

2. **打開專案（使用 VSCode）：**
   - 「檔案」->「開啟資料夾」-> 選擇 `mx_ph`

3. **創建並啟動虛擬環境：**
   ```bash
   conda create -n mx_ph python=3.X
   conda activate mx_ph
   ```

4. **安裝依賴包：**
   ```bash
   pip install -r requirements.txt
   ```
   請隨時更新 `requirements.txt` 以反映新的依賴。

## 使用說明

本專案支持靈活的命令行互動，以下是一些基本的使用方式：

- **查看幫助信息：**
  ```bash
  ~/mx_ph$ python mx_ph/main.py --help
  ```

為了讓團隊更方便快捷地使用這些自動化命令，以下是進一步精簡和清晰化的說明：

### 快速命令指南

- **清理工作區**  
  清除臨時文件和建構產物，保持工作環境整潔。
  ```bash
  make clean
  ```

- **運行測試**  
  利用 pytest 執行單元測試，確保代碼質量。
  ```bash
  make test
  ```

- **安裝專案**  
  將專案安裝到當前的 Python 環境，便於任何地方使用。
  ```bash
  make install
  ```

- **版本控制**  
  簡化提升版本的流程，包括自動提交更改。
  - 提升小修正版本（Patch）：
    ```bash
    make patch
    ```
  - 添加新功能時提升次版本（Minor）：
    ```bash
    make minor
    ```
  - 重大修改提升主版本（Major）：
    ```bash
    make major
    ```

希望這份指南能夠幫助您更輕鬆地融入專案，如果有任何疑問或建議，歡迎隨時提出。讓我們共同打造一個更好的 `mx_ph`！