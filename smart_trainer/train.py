import argparse
import os
import re
import sys
import glob
import time
import traceback
from pathlib import Path
import requests
import io

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler


TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "") 
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "")
LOG_FILENAME = "training.log"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT
DEFAULT_MODEL_PATH = PROJECT_ROOT / "Model.bin"
GO_LOCAL_PATH = SCRIPT_DIR / "transform.go"
GO_REMOTE_URL = "https://raw.githubusercontent.com/vernesong/mihomo/Alpha/component/smart/lightgbm/transform.go"

class TeeLogger(object):
    def __init__(self, filename=LOG_FILENAME):
        self.terminal = sys.stdout
        self.filename = filename
        with open(self.filename, "w", encoding='utf-8') as f:
            f.write(f"--- 训练开始: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        self.log = open(self.filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger()
sys.stderr = sys.stdout

STD_FEATURES = [
    'connect_time', 'latency', 'download_mb', 'history_download_mb',
    'maxdownloadrate_kb', 'history_maxdownloadrate_kb', 'duration_minutes', 
    'last_used_seconds', 'upload_mb', 'history_upload_mb', 
    'maxuploadrate_kb', 'history_maxuploadrate_kb', 'traffic_ratio', 'traffic_density',
    'asn_hash', 'host_hash', 'ip_hash', 'geoip_hash' 
]

ROB_FEATURES = ['success', 'failure']

LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 10000,       
    'learning_rate': 0.03,       
    'num_leaves': 31,            
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
}

def print_separator(title=None):
    print("=" * 60)
    if title:
        print(f"{title}")
        print("=" * 60)

def send_telegram_msg(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        requests.post(url, json=data, timeout=10)
    except Exception as e:
        sys.stdout.terminal.write(f"⚠️ TG 发送失败: {e}\n")

def send_telegram_logs(header_msg):
    if not TG_BOT_TOKEN or not TG_CHAT_ID: return

    send_telegram_msg(header_msg)

    try:
        with open(LOG_FILENAME, "r", encoding='utf-8') as f:
            content = f.read()
    except:
        content = "无法读取日志文件"

    chunk_size = 3500
    total_len = len(content)
    
    if total_len == 0:
        send_telegram_msg("<i>(日志为空)</i>")
        return

    for i in range(0, total_len, chunk_size):
        chunk = content[i : i + chunk_size]
        formatted_msg = f"<pre>{chunk}</pre>"
        send_telegram_msg(formatted_msg)
        time.sleep(1)

    sys.stdout.terminal.write("📨 Telegram 日志已推送\n")

def get_feature_order():
    if GO_LOCAL_PATH.exists():
        print(f"📂 [本地模式] 检测到源码: {GO_LOCAL_PATH}")
        try:
            content = GO_LOCAL_PATH.read_text(encoding='utf-8')
            return parse_go_content(content)
        except Exception as e:
            print(f"⚠️ 本地读取失败 ({e})，切换至在线模式...")
    else:
        print("ℹ️ 未检测到本地 transform.go，切换至在线模式...")

    print(f"☁️ [在线模式] 正在下载: {GO_REMOTE_URL}")
    try:
        resp = requests.get(GO_REMOTE_URL, timeout=15)
        resp.raise_for_status()
        print("✅ 下载成功")
        return parse_go_content(resp.text)
    except Exception as e:
        print(f"❌ [错误] 下载失败: {e}")
        raise RuntimeError("无法获取特征定义")

def parse_go_content(content):
    pattern = r'func getDefaultFeatureOrder\(\) map\[int\]string\s*\{(.*?)\}'
    match = re.search(pattern, content, re.DOTALL)
    if not match: raise ValueError("源码中未找到 getDefaultFeatureOrder 函数")
    body = match.group(1)
    pairs = re.findall(r'(\d+):\s*"([^"]+)"', body)
    if not pairs: raise ValueError("函数内未提取到任何特征定义")
    feature_map = {int(k): v for k, v in pairs}
    print(f"✨ 成功解析出 {len(feature_map)} 个特征")
    return feature_map

def load_data(data_dir, days=30):
    print("\n[步骤1] 加载原始数据")
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    all_files = glob.glob(str(data_dir / "*.csv"))
    cutoff_time = time.time() - (days * 86400)
    recent_files = []

    for f in all_files:
        try:
            fname = os.path.basename(f)
            match = re.search(r'smart_(\d{8}_\d{4})', fname)
            if match:
                file_ts = time.mktime(time.strptime(match.group(1), "%Y%m%d_%H%M"))
            else:
                file_ts = os.path.getmtime(f)
            if file_ts > cutoff_time:
                recent_files.append(f)
        except:
            pass

    if not recent_files:
        print("⚠️ 未发现近期数据，将加载所有可用数据...")
        recent_files = all_files

    if not recent_files:
        raise FileNotFoundError("没有找到任何 CSV 数据文件")

    print(f"📥 选中 {len(recent_files)} 个数据文件")
    
    dfs = []
    for f in recent_files:
        fname = os.path.basename(f)
        try:
            df = pd.read_csv(f, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            print(f"🔧 正在修复损坏文件: {fname} ...", end=" ")
            try:
                with open(f, 'rb') as raw_file:
                    content_bytes = raw_file.read()
                content_str = content_bytes.decode('utf-8', errors='replace')
                df = pd.read_csv(io.StringIO(content_str), on_bad_lines='skip')
                print(f"✅ 成功抢救 {len(df)} 条数据")
            except Exception as e:
                print(f"❌ 修复失败: {str(e)}")
                continue
        except pd.errors.EmptyDataError:
            print(f"⚠️ 跳过空文件: {fname}")
            continue
        except Exception as e:
            print(f"⚠️ 跳过文件 {fname}: {str(e)}")
            continue

        try:
            match = re.search(r'smart_(\d{8}_\d{4})', fname)
            if match:
                file_ts = time.mktime(time.strptime(match.group(1), "%Y%m%d_%H%M"))
            else:
                file_ts = os.path.getmtime(f)
            df['__age_hours'] = max(0, (time.time() - file_ts) / 3600)
            dfs.append(df)
        except:
            pass

    if not dfs:
        raise ValueError("没有可用的数据被加载")

    merged_df = pd.concat([d.dropna(axis=1, how='all') for d in dfs], ignore_index=True)
    merged_df = merged_df.sort_values('__age_hours', ascending=False).reset_index(drop=True)
    print(f"📊 数据加载完成，共 {len(merged_df)} 条记录")
    return merged_df

def preprocess_data(df, feature_order):
    print("\n[步骤2] 特征工程与目标构建")
    
    if 'maxdownloadrate_kb' not in df.columns:
         if 'download_mbps' in df.columns:
            df['maxdownloadrate_kb'] = df['download_mbps'] * 1000
         else:
            raise ValueError("数据中缺少速度列 (maxdownloadrate_kb)")

    raw_speed = df['maxdownloadrate_kb'].fillna(0).clip(lower=0) * 1000
    
# 提高速度的权重系数 (1.0 -> 1.5)，优先保证大带宽节点的评分
    speed_score = np.log1p(raw_speed) * 1.5
    
    failure_penalty = 0.5 ** df['failure'].fillna(0)
    latency = df['latency'].fillna(5000)
    latency_penalty = 1.0 / (1.0 + np.exp((latency - 500) / 100))

    df['target_y'] = speed_score * failure_penalty * latency_penalty
    
    print("🎯 目标构建示例:")
    print(df[['maxdownloadrate_kb', 'failure', 'latency', 'target_y']].head(3).to_string(index=False, justify='left'))

    feature_cols = [feature_order[i] for i in sorted(feature_order.keys())]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    X = df[feature_cols].copy().fillna(0)
    y = df['target_y']

    scalers = {}
    
    std_cols = [c for c in STD_FEATURES if c in X.columns]
    if std_cols:
        s_std = StandardScaler()
        X[std_cols] = s_std.fit_transform(X[std_cols])
        scalers['standard'] = s_std
        scalers['std_features'] = std_cols
        print(f"📏 StandardScaler 应用于 {len(std_cols)} 个特征")

    rob_cols = [c for c in ROB_FEATURES if c in X.columns]
    if rob_cols:
        s_rob = RobustScaler()
        X[rob_cols] = s_rob.fit_transform(X[rob_cols])
        scalers['robust'] = s_rob
        scalers['rob_features'] = rob_cols
        print(f"🛡️ RobustScaler 应用于 {len(rob_cols)} 个特征")

    w_time = np.exp(-0.01 * df['__age_hours'])
    w_speed = 1.0 + np.log1p(raw_speed) / 20.0
    weights = w_time * w_speed

    return X, y, weights, scalers

def save_model_with_config(model, scalers, feature_order, output_path):
    print("\n[步骤4] 导出模型")
    
    model.booster_.save_model(str(output_path))
    
    lines = ["", "", "[transforms]", "[order]"]
    for i in sorted(feature_order.keys()):
        lines.append(f"{i}={feature_order[i]}")
    lines.append("[/order]")
    
    lines.append("[definitions]")
    
    s_std = scalers.get('standard')
    f_std = scalers.get('std_features', [])
    if s_std and f_std:
        name_to_idx = {v:k for k,v in feature_order.items()}
        indices = [str(name_to_idx[name]) for name in f_std if name in name_to_idx]
        lines.append("std_type=StandardScaler")
        lines.append(f"std_features={','.join(indices)}")
        lines.append(f"std_mean={','.join([f'{v:.6f}' for v in s_std.mean_])}")
        lines.append(f"std_scale={','.join([f'{v:.6f}' for v in s_std.scale_])}")
        
    s_rob = scalers.get('robust')
    f_rob = scalers.get('rob_features', [])
    if s_rob and f_rob:
        lines.append("")
        name_to_idx = {v:k for k,v in feature_order.items()}
        indices = [str(name_to_idx[name]) for name in f_rob if name in name_to_idx]
        lines.append("robust_type=RobustScaler")
        lines.append(f"robust_features={','.join(indices)}")
        lines.append(f"robust_center={','.join([f'{v:.6f}' for v in s_rob.center_])}")
        lines.append(f"robust_scale={','.join([f'{v:.6f}' for v in s_rob.scale_])}")

    lines.append("[/definitions]")
    lines.append("")
    lines.append("transform=true")
    lines.append("[/transforms]")
    
    with open(output_path, "ab") as f:
        f.write("\n".join(lines).encode('utf-8'))
        
    print(f"💾 模型已保存至: {output_path}")

def run_training():

    print_separator("Smart 训练器启动")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    feature_order = get_feature_order()
    
    df = load_data(args.data_dir, days=15)

    X, y, w, scalers = preprocess_data(df, feature_order)
    
    print("\n[步骤3] 模型训练")
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    w_train, w_val = w.iloc[:split_idx], w.iloc[split_idx:]
    
    print(f"🧠 训练集: {len(X_train)} 条 | 🧪 验证集: {len(X_val)} 条")
    print(f"🛑 早停机制已启用: 如果验证集分数在 100 轮内没有提升，将自动停止训练。")
    
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=100)
    ]
    
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        callbacks=callbacks
    )
    
    if model.best_iteration_ < LGBM_PARAMS['n_estimators']:
        print(f"⏹️ 触发早停！在第 {model.best_iteration_} 轮达到最佳效果。")
    else:
        print(f"🏁 训练跑满全程，在第 {model.best_iteration_} 轮达到最佳效果。")

    print("\n[评估报告]")
    preds = model.predict(X_val)
    r2 = r2_score(y_val, preds)
    final_score = max(0, r2 * 10)
    
    print(f"📈 验证集 R2 Score: {r2:.4f}")
    print(f"🌟 模型综合评分: {final_score:.1f} / 10.0")
    
    if final_score > 8.0:
        print("🏆 评级: S级 (极佳) - 极速节点识别精准")
    elif final_score > 6.0:
        print("🟢 评级: A级 (良好) - 模型可用性高")
    elif final_score > 4.0:
        print("🟡 评级: B级 (及格) - 正常水平")
    else:
        print("🟠 评级: C级 (一般) - 需积累更多数据")

    if args.output.exists():
        args.output.unlink()
    save_model_with_config(model, scalers, feature_order, args.output)
    
    print_separator("训练完成")
    return final_score, len(df), model.best_iteration_

if __name__ == "__main__":
    try:
        score, count, rounds = run_training()
        
        header = (
            f"✅ <b>Mihomo 训练成功</b>\n"
            f"📊 数据量: <code>{count}</code> 条\n"
            f"🔄 训练轮数: <code>{rounds}</code>\n"
            f"🎯 最终评分: <code>{score:.2f} / 10.0</code> (R2)"
        )
        send_telegram_logs(header)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"\n❌ 发生严重错误:\n{error_trace}")
        
        header = (
            f"❌ <b>Mihomo 训练失败</b>\n"
            f"⚠️ 错误原因: <code>{str(e)}</code>"
        )
        send_telegram_logs(header)
        sys.exit(1)