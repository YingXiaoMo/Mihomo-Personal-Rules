# 🤖 Mihomo 智能权重模型训练与自动化部署

本项目旨在利用 LightGBM 回归模型，基于 Clash.Meta/Mihomo 产生的历史连接日志数据，训练出能够预测代理节点最佳权重的模型（`Model.bin`），并通过 GitHub Actions 实现自动化训练、构建和部署。

## ✨ 项目功能一览

1. **Go 源码特征解析：** Python 脚本自动解析 Mihomo Go 源码中定义的特征顺序，确保模型训练的特征与实际运行时使用的特征一致。

2. **数据清洗与加载：** 自动遍历数据文件，执行数据加载、缺失值处理和清洗。

3. **LightGBM 模型训练：** 使用 LightGBM 框架进行高效的模型训练。

4. **GitHub Actions 自动化：** 自动执行整个训练流程。

5. **最终产物发布（Release）：** 训练成功的模型文件（`Model.bin`）将通过 GitHub Release 发布，**不会**提交回仓库的代码或 `models/` 目录中。

---

## ⚠️ 关键操作警告

### 数据来源与 Runner 清理机制

* **数据来源：** 本工作流**不存储数据**，假定历史数据是从**外部云存储**（例如 Google Cloud Storage 或 Google Drive）下载到 Runner 上的 `data/` 目录中。

* **Runner 清理机制：** 在工作流执行完毕后，托管它的 Runner 虚拟机会被销毁，因此 `data/` 目录中的所有文件将随之**自动删除**。

* **重要提示：** 请勿将原始、不可替代的历史数据文件直接存储在您的 Git 仓库的 `data/` 目录中！您的数据应始终保存在外部可靠的存储服务上。

---

## 📂 文件结构与依赖

| 路径 | 描述 | 状态 | 
| ----- | ----- | ----- | 
| `data/` | **Action 自动创建。** CSV 数据的临时目标目录。 | **本地必填** | 
| `models/` | **脚本自动创建。** 存放最终生成的模型文件 `Model.bin`。 | **本地必填** | 
| `Smart/go_transform/transform.go` | **必需。** Go 语言特征定义文件。`train_smart.py` 依赖此文件来获取特征顺序。 | **必填** | 
| `Smart/scripts/train_smart.py` | 训练主脚本。负责解析特征、数据处理、训练和编码。 | **必填** | 
| `required_dependencies.txt` | **必需。** 严格锁定的 Python 依赖列表文件。 | **必填** | 
| `.github/workflows/train.yml` | GitHub Actions 自动化工作流定义文件。 | **必填** | 

---

## 🛠️ 新用户使用指南：需要修改的关键位置

### 1. 文件夹创建和数据准备

如果在本地使用，必须在项目根目录下创建以下文件夹并准备相应文件：

| 文件夹/文件 | 内容要求 | 
| ----- | ----- | 
| `data/` | **创建空文件夹。** CSV 数据的目标目录。 | 
| `models/` | **脚本自动创建。** 训练脚本的输出位置。 | 
| `Smart/go_transform/transform.go` | **放入 Mihomo 源码中的 Go 特征定义文件。** | 

### 2. 训练脚本 (`Smart/scripts/train_smart.py`) 修改

| 变量/代码段 | 建议修改内容 | 
| ----- | ----- | 
| `DATA_FILE`, `GO_FILE`, `MODEL_FILE` | 修改路径变量，确保脚本找到 Go 文件、数据目录和模型输出位置。 | 
| `LGBM_PARAMS` | 可调整 LightGBM 超参数，如 `learning_rate`。 | 
| `STD_SCALER_FEATURES` / `ROBUST_SCALER_FEATURES` | 根据 Go 源码定义的特征类型，调整特征标准化方法。 | 

### 3. 自动化工作流 (`.github/workflows/train.yml`) 修改

| 配置项 | 建议修改内容 | 
| ----- | ----- | 
| `on:` 触发器 | 默认 `push`，可改为定时训练：<br>`schedule: [ { cron: '0 0 * * *' } ]` | 
| **数据下载步骤** | 添加步骤，将外部 CSV 数据下载到 `./data/` 目录。 | 
| `python-version` | 检查 Python 版本（如 3.10 或 3.11）。 | 
| **Python 依赖安装** | `pip install -r required_dependencies.txt` | 
| **Telegram Secrets** | 更新 `secrets.TG_BOT_TOKEN` 和 `secrets.TG_CHAT_ID`（如使用 Telegram 通知）。 | 


### 4. 可选：配置 Telegram 部署通知

如果您希望在模型训练和部署成功后收到 Telegram 通知，您需要在 GitHub Secrets 中设置必要的环境变量，这样可以将完整训练日志推送到您的通知机器人。

  **准备 Secrets**

在 GitHub 仓库中进入 **Settings -> Secrets and variables -> Actions**，添加以下 Secrets：

| Secret 名称 | 内容 | 必填 | 说明 |
| ----------- | ---- | ---- | ---- |
| `TG_BOT_TOKEN` | 您的 Telegram 机器人的 Token | ✅ | 用于发送消息到您的机器人 |
| `TG_CHAT_ID` | 您希望接收通知的聊天 ID 或频道 ID | ✅ | 可以是个人聊天或频道 ID |

> ⚠️ 注意：此步骤可选，如果不需要通知，可以跳过。


### 5. 本地运行说明 (Local Execution)

1. **环境准备**  

   - 安装 Python 3.10 或更高版本  
   - 安装依赖：
   ```bash
   pip install pandas==2.2.3 scikit-learn==1.7.0 lightgbm==3.3.5 joblib==1.5.1 numpy==2.3.1
   ```

2. **文件准备**  

   - 数据文件：手动将所有历史 CSV 数据文件放置到项目根目录下的 `/data/`  
   - Go 源码文件：确保 `Smart/go_transform/transform.go` 已存在

3. **执行训练**  

   在项目根目录下运行：
   ```bash
   python Smart/scripts/train_smart.py
   ```

   执行成功后，模型文件 `Model.bin` 将生成在 `/models/` 目录下

---

## 🚀 关键步骤详解：数据下载配置示例（GCS / Rclone）  

由于自动化运行的数据需要从外部云存储下载，您需要在 `train.yml` 的 `train` Job 中添加相应步骤。

---

### 选项 A：使用 Google Cloud Storage (GCS)

1. 准备 Secret：Google 服务账户密钥  
   - **Secret 名称：** `GCP_SA_KEY`  
   - **Secret 内容：** Google 服务账户密钥 JSON 文件内容  
     > 注意：此密钥必须拥有访问 GCS 存储桶中数据的权限

2. YAML 示例：
```yaml
# 1. 认证 Google Cloud
- name: 认证 Google Cloud
  uses: google-github-actions/auth@v2 
  with:
    credentials_json: ${{ secrets.GCP_SA_KEY }}

# 2. 下载历史数据
- name: 从 GCS 下载历史数据 (gsutil)
  run: |
    echo "创建数据目录..."
    mkdir -p data
    
    # 替换为实际 GCS 路径
    gsutil cp gs://your-gcs-bucket-name/smart_data/*.csv ./data/
    
    echo "数据下载完成！"
```

---

### 选项 B：使用 Rclone (适用于 Google Drive / OneDrive / 其他云存储)

1. 准备 Secret：Rclone 配置文件  
   - **Secret 名称：** `RCLONE_CONFIG`  
   - **Secret 内容：** `rclone.conf` 文件中 **除了 `[remote_name]` 之外的所有配置内容**

2. YAML 示例：
```yaml
# 1. 安装 rclone 工具
- name: 安装 Rclone
  run: sudo apt-get install rclone -y

# 2. 配置 rclone
- name: 配置 Rclone 认证
  run: |
    echo "创建数据目录..."
    mkdir -p data
    
    # 替换 [gdrive_remote_name] 为实际远程名称
    mkdir -p ~/.config/rclone
    echo "[gdrive_remote_name]" > ~/.config/rclone/rclone.conf
    echo "${{ secrets.RCLONE_CONFIG }}" >> ~/.config/rclone/rclone.conf

# 3. 下载历史数据
- name: 从云存储下载历史数据 (rclone)
  run: |
    # 替换 gdrive_remote_name:path/to/data 为实际路径
    rclone copy gdrive_remote_name:path/to/data ./data/ --include "*.csv"
    echo "数据下载完成！"
```

> 注意：  
> - 替换 `[gdrive_remote_name]` 和 `gdrive_remote_name:path/to/data`  
> - `~/.config/rclone/` 在 Runner 中默认可能不存在，脚本中已创建

