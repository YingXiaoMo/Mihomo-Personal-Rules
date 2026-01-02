# Mihomo Smart 训练脚本使用教程

## 准备工作

1. 安装 Python 3.10+ 和依赖：
   ```bash
   cd smart_trainer
   pip install -r requirements.txt
   ```

2. 准备 Mihomo 数据：启用 Smart 模式，导出 CSV 文件到项目根目录。

   **可选**: 将 Mihomo 源码中的 `transform.go` 文件复制到项目根目录，脚本会优先使用本地文件，避免网络下载。

## 使用步骤 

1. 运行脚本：
   ```bash
   python smart_trainer/train_smart.py
   ```

2. 脚本自动执行：
   - **检查本地 transform.go**: 优先使用项目根目录的 `transform.go` 文件（如果存在）
   - 下载 Go 源码解析特征（如果没有本地文件）
   - 加载最近 15 天数据
   - 清洗和标准化数据
   - 训练 LightGBM 模型
   - 保存模型文件

3. 查看结果：脚本输出 MAE、R² 和评级。

## 配置变量 (本地运行)

- `--data_dir`: 数据目录路径 (默认: 项目根目录)
- `--output`: 输出模型路径 (默认: 项目根目录/Model.bin)


## Rclone 安装与配置 (Github自动化前置要求)

为了让 GitHub Actions 自动拉取云端数据，你需要配置 Rclone 并获取配置文件内容。  
如果只本地训练，下面的都不用管。

1. **安装 Rclone**:
   - 前往 [Rclone Downloads](https://rclone.org/downloads/) 下载对应系统的版本并安装。
   - Windows 用户可下载 `.zip` 解压并将 `rclone.exe` 所在目录添加到系统环境变量 `PATH` 中。

2. **配置连接**:
   在终端运行：
   ```bash
   rclone config
   ```
   - 输入 `n` 新建配置。
   - 输入名称 (例如 `gdrive`)。**请记住这个名称**，后续在 Secret `REMOTE` 中会用到 (格式如 `gdrive:/mihomo-data`)。
   - 根据提示选择存储服务商 (如 Google Drive, OneDrive 等) 并完成授权流程。
   - 可以让AI写个脚本，每天把 Openwrt 上生成的 CSV  文件自动上传到你设置的网盘。

3. **获取配置 Base64 编码**:
   配置完成后，需要将配置文件内容编码为 Base64 字符串，填入 GitHub Secret 的 `RCLONE_CONFIG_B64`。

   首先查找配置文件位置：
   ```bash
   rclone config file
   ```

   然后将该文件内容转换为 Base64：
   
   - **Linux/macOS**:
     ```bash
     # 将 <path> 替换为实际配置文件路径
     cat <path/to/rclone.conf> | base64
     ```
   - **Windows (PowerShell)**:
     ```powershell
     # 将 <path> 替换为实际配置文件路径
     [Convert]::ToBase64String([IO.File]::ReadAllBytes("<C:\path\to\rclone.conf>"))
     ```
   
   > 复制输出的一长串字符串 (不要包含换行符)，保存备用。

## 自动化运行 (GitHub Actions)

本项目包含一个 GitHub Actions 工作流 (`.github/workflows/train.yml`)，可每天自动拉取数据、训练模型并发布 Release。

### 必需的 GitHub Secrets

在项目仓库的 `Settings` -> `Secrets and variables` -> `Actions` 中添加以下 Repository secrets：

| 变量名 | 描述 | 获取方式 |
| :--- | :--- | :--- |
| `RCLONE_CONFIG_B64` | Base64 编码的 rclone 配置文件内容 | 运行 `cat rclone.conf \| base64` (Linux/Mac) 或在 Windows Powershell 运行 `[Convert]::ToBase64String([IO.File]::ReadAllBytes("rclone.conf"))` |
| `REMOTE` | rclone 远程存储路径 | 例如: `gdrive:/mihomo-data` (对应 rclone 配置中的名称和路径) |
| `TG_BOT_TOKEN` | Telegram 机器人 Token | 从 [@BotFather](https://t.me/BotFather) 获取 |
| `TG_CHAT_ID` | 接收通知的 Telegram 用户 ID | 从 [@userinfobot](https://t.me/userinfobot) 获取 |

### 工作流逻辑
1. **拉取数据**: 使用 `rclone` 从配置的 `REMOTE` 下载 CSV 数据文件。
2. **训练模型**: 运行训练脚本生成 `Model.bin`。
3. **发布版本**: 自动删除旧的 `smart-model` 标签并发布新的 Release。
4. **发送通知**: 通过 Telegram 发送训练结果（成功/失败）及日志摘要。


