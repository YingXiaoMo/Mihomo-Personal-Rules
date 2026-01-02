import argparse
import os
import re
import sys
import glob
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_DATA_DIR = PROJECT_ROOT
DEFAULT_MODEL_PATH = PROJECT_ROOT / "Model.bin"
CACHE_DIR = SCRIPT_DIR / "cache"
GO_SOURCE_CACHE_PATH = CACHE_DIR / "transform.go.cache"
GO_SOURCE_URL = "https://raw.githubusercontent.com/vernesong/mihomo/Alpha/component/smart/lightgbm/transform.go"

IGNORED_FEATURES = [
    'upload_mb', 
    'history_upload_mb',
    'maxuploadrate_kb',         
    'history_maxuploadrate_kb',
    
    'asn_feature', 
    'country_feature', 
    'address_feature', 
    'port_feature', 
    'connection_type_feature',
    
    'traffic_density', 
    'traffic_ratio'
]

CONTINUOUS_FEATURES = [
    'connect_time', 'latency', 
    'download_mb', 'history_download_mb', 
    'maxdownloadrate_kb', 'history_maxdownloadrate_kb', 
    'duration_minutes', 'last_used_seconds',
    'asn_hash', 'host_hash', 'ip_hash', 'geoip_hash',
    'upload_mb', 'history_upload_mb', 'maxuploadrate_kb', 'history_maxuploadrate_kb',
    'traffic_density', 'traffic_ratio'
]

COUNT_FEATURES = ['success', 'failure']


LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 10000,       
    'learning_rate': 0.03,
    'num_leaves': 63,
    'max_depth': -1,
    'min_child_samples': 10,    
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
}


def print_separator(title=None):
    if title:
        print("=" * 60)
        print(f"{title}")
        print("=" * 60)
    else:
        print("=" * 60)

class GoTransformParser:
    """
    Go 源码解析器 (增强版)
    """
    def __init__(self, content: str):
        self.content = content
        self.feature_order = self._parse_feature_order()

    def _parse_feature_order(self):
        print("开始解析 getDefaultFeatureOrder 函数...")
        pattern = (
            r'func getDefaultFeatureOrder\(\) map\[int\]string \{\s*'
            r'return map\[int\]string\{(.*?)\}\s*\}'
        )
        match = re.search(pattern, self.content, re.DOTALL)
        
        if not match:
            print("警告: 未能在源码中找到特征定义块，使用内置后备配置")
            return self._get_fallback_config()
        
        body = match.group(1)
        pairs = re.findall(r'(\d+):\s*"([^"]+)"', body)
        
        if not pairs:
            print("警告: 解析到的特征列表为空，使用后备配置")
            return self._get_fallback_config()
            
        feature_map = {int(idx): name for idx, name in pairs}
        print(f"成功解析 {len(feature_map)} 个特征")
        return feature_map

    def _get_fallback_config(self):
        features = [
            'success', 'failure', 'connect_time', 'latency', 'upload_mb', 
            'history_upload_mb', 'maxuploadrate_kb', 'history_maxuploadrate_kb',
            'download_mb', 'history_download_mb', 'maxdownloadrate_kb', 
            'history_maxdownloadrate_kb', 'duration_minutes', 'last_used_seconds', 
            'is_udp', 'is_tcp', 'asn_feature', 'country_feature', 'address_feature', 
            'port_feature', 'traffic_ratio', 'traffic_density', 
            'connection_type_feature', 'asn_hash', 'host_hash', 'ip_hash', 'geoip_hash'
        ]
        return {i: f for i, f in enumerate(features)}

    def get_order(self):
        return self.feature_order

# ==============================================================================
# 核心逻辑
# ==============================================================================

def fetch_go_source():
    print("\n[步骤1] Go 源码解析")
    
    local_go_path = PROJECT_ROOT / "transform.go"
    if local_go_path.exists():
        print(f"发现本地 transform.go 文件: {local_go_path}")
        return local_go_path.read_text(encoding='utf-8')
    
    if GO_SOURCE_CACHE_PATH.exists():
        if (time.time() - GO_SOURCE_CACHE_PATH.stat().st_mtime) < 86400:
            print(f"成功加载本地缓存: {GO_SOURCE_CACHE_PATH}")
            return GO_SOURCE_CACHE_PATH.read_text(encoding='utf-8')

    print(f"正在下载 Go 源文件: {GO_SOURCE_URL}")
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        response = requests.get(GO_SOURCE_URL, timeout=10)
        response.raise_for_status()
        content = response.text
        GO_SOURCE_CACHE_PATH.write_text(content, encoding='utf-8')
        print("下载并缓存成功")
        return content
    except Exception as e:
        if GO_SOURCE_CACHE_PATH.exists():
            print(f"下载失败 ({e})，使用旧缓存")
            return GO_SOURCE_CACHE_PATH.read_text(encoding='utf-8')
        raise RuntimeError(f"无法获取 Go 源码: {e}")

def load_data(data_dir, days=90):
    print("\n[步骤2] 数据加载与清洗")
    print(f"开始从数据目录加载 CSV 文件: {data_dir}")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_files = glob.glob(str(data_dir / "*.csv"))
    
    # 筛选最近 N 天的数据
    cutoff_time = time.time() - (days * 86400)
    recent_files = []
    
    for f in all_files:
        try:
            mtime = os.path.getmtime(f)
            # 尝试从文件名解析日期 (smart_20250101_1200.csv)
            fname = os.path.basename(f)
            date_match = re.search(r'smart_(\d{8}_\d{4})', fname)
            if date_match:
                file_time = time.mktime(time.strptime(date_match.group(1), "%Y%m%d_%H%M"))
                if file_time > cutoff_time:
                    recent_files.append(f)
            elif mtime > cutoff_time:
                recent_files.append(f)
        except:
            pass

    if not recent_files:
        print("警告: 未发现近期数据，将使用所有可用数据...")
        recent_files = all_files
    
    if not recent_files:
        raise FileNotFoundError("没有找到任何 CSV 数据文件")

    print(f"--- 选中 {len(recent_files)} 个数据文件 ---")
    
    dfs = []
    for f in recent_files:
        try:
            # 宽容模式读取
            df = pd.read_csv(f, encoding='utf-8', on_bad_lines='skip')
            
            # 计算文件年龄（天），用于后续权重衰减
            fname = os.path.basename(f)
            date_match = re.search(r'smart_(\d{8}_\d{4})', fname)
            if date_match:
                file_time = time.mktime(time.strptime(date_match.group(1), "%Y%m%d_%H%M"))
                age_days = (time.time() - file_time) / 86400
            else:
                age_days = (time.time() - os.path.getmtime(f)) / 86400
            
            df['__file_age_days'] = max(0, age_days)
            dfs.append(df)
        except Exception as e:
            print(f"跳过文件 {os.path.basename(f)}: {e}")
            continue
    
    if not dfs:
        raise ValueError("无法加载任何有效数据")
    
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"数据合并完成，原始记录数: {len(merged_df)}")
    return merged_df

def preprocess_data(df, feature_order):
    print("\n[步骤3] 特征提取与目标构建 (极速模式)")

    # 1. 确定目标列 (我们只看 MaxDownloadRate)
    target_col = 'maxdownloadrate_kb'
    if target_col not in df.columns:
        # 兼容旧版本数据列名
        if 'download_mbps' in df.columns:
            df[target_col] = df['download_mbps'] * 1024 # 转换为 kb
        else:
            raise ValueError("严重错误: 数据中缺少 maxdownloadrate_kb 列，无法训练速度模型")

    # 填充缺失值
    df[target_col] = df[target_col].fillna(0)
    
    # --------------------------------------------------------------------------
    # 核心黑科技：构建 "惩罚性" 目标变量 (Punished Target)
    # --------------------------------------------------------------------------
    # 目标：让模型预测的值不仅仅是速度，而是 "稳定速度"。
    # 手段：如果节点有丢包或高延迟，我们在训练时人为把它的 Target 值打低。
    # 结果：模型在推理时，会给那些 "快但丢包" 的节点打出很低的预测分，从而避开它们。
    # --------------------------------------------------------------------------
    
    raw_speed = df[target_col]
    
    # 惩罚因子 1: 丢包惩罚
    # failure > 0 时，惩罚极其严厉。failure=1 -> 分数变为 1/3; failure=2 -> 分数变为 1/5
    failure_penalty = 1.0 / (1.0 + df['failure'].fillna(0) * 2.0)
    
    # 惩罚因子 2: 延迟惩罚
    # 延迟越高，分数越低。每 1000ms 延迟，分数打 8 折。
    # 主要是为了剔除那些 2000ms+ 的假死节点
    latency_val = df['latency'].fillna(10000)
    latency_penalty = 1.0 / (1.0 + (latency_val / 4000.0)) 
    
    # 最终训练目标：(物理速度) * (丢包惩罚) * (延迟惩罚)
    # 这样训练出来的模型，预测值越高，代表节点 "既快又稳"
    y = raw_speed * failure_penalty * latency_penalty
    
    # 记录日志看看效果
    print(f"目标构建示例 (前5条):")
    for i in range(min(5, len(df))):
        print(f"  原始速度: {raw_speed.iloc[i]:.0f} kbps, 失败数: {df['failure'].iloc[i]}, "
              f"延迟: {latency_val.iloc[i]:.0f} ms -> 训练目标值: {y.iloc[i]:.2f}")

    # --------------------------------------------------------------------------
    # 策略优化：新节点探索机制 (Exploration Strategy)
    # 问题：如果完全依赖历史数据，新节点(历史为0)会被永远打入冷宫。
    # 解决：随机将 25% 数据的"历史特征"强行置为 0。
    # 效果：教会模型 "即使没有历史数据，只要延迟低，也有可能是个好节点"。
    # --------------------------------------------------------------------------
    history_cols = ['history_maxdownloadrate_kb', 'history_download_mb', 'last_used_seconds']
    # 创建一个随机掩码，25% 的概率为 True
    exploration_mask = np.random.rand(len(df)) < 0.25
    
    for col in history_cols:
        if col in df.columns:
            # 对于选中的行，将历史特征抹去 (模拟成新节点)
            df.loc[exploration_mask, col] = 0.0

    # 2. 特征屏蔽 (Masking)
    # 将不需要的特征置为 0，防止噪声干扰
    for col in IGNORED_FEATURES:
        if col in df.columns:
            df[col] = 0.0

    # 3. 按顺序提取特征矩阵 X
    ordered_cols = [feature_order[i] for i in sorted(feature_order.keys())]
    
    # 确保所有列都存在
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    X = df[ordered_cols].copy()
    
    # 只保留数值类型
    X = X.select_dtypes(include=np.number)
    
    # 4. 特征标准化 (Standardization)
    print("\n[步骤4] 特征标准化")
    scalers = {}
    
    # 数值型特征 -> StandardScaler
    std_cols = [c for c in CONTINUOUS_FEATURES if c in X.columns]
    if std_cols:
        scaler_std = StandardScaler()
        X[std_cols] = scaler_std.fit_transform(X[std_cols])
        scalers['standard'] = scaler_std
        scalers['std_features'] = std_cols
        print(f"StandardScaler 应用于 {len(std_cols)} 个特征")

    # 计数型特征 -> RobustScaler
    rob_cols = [c for c in COUNT_FEATURES if c in X.columns]
    if rob_cols:
        scaler_rob = RobustScaler()
        X[rob_cols] = scaler_rob.fit_transform(X[rob_cols])
        scalers['robust'] = scaler_rob
        scalers['rob_features'] = rob_cols
        print(f"RobustScaler 应用于 {len(rob_cols)} 个特征")

    # 5. 样本权重 (Sample Weights) - 优化：时间主导的乘法权重
    # --------------------------------------------------------------------------
    # 策略：指数级时间衰减
    # Day 0: 100%, Day 1: 82%, Day 3: 55%, Day 7: 25%, Day 14: 6%
    # 越旧的数据，对模型的影响力呈断崖式下跌。
    # --------------------------------------------------------------------------
    time_decay = np.exp(-0.2 * df['__file_age_days'])
    
    # 速度加成：依然保留对高速样本的关注，但必须受制于时间衰减
    # 速度越快，权重会有 1.0 ~ 2.0 倍的加成
    speed_bonus = np.log1p(raw_speed) / 12.0  
    
    # 最终权重 = 时间衰减系数 * (基础分 + 速度加成)
    # 使用乘法：确保旧数据即使速度再快，总权重也被时间系数强行压低
    sample_weights = time_decay * (1.0 + speed_bonus)

    return X, y, sample_weights, scalers

def save_model_and_params(model, scalers, feature_order, output_path):
    print("\n[步骤7] 模型保存与参数注入")
    
    # 保存原始 LightGBM 模型
    model.booster_.save_model(str(output_path))
    
    # 构建 INI 格式的变换参数
    feature_name_to_idx = {v: k for k, v in feature_order.items()}
    
    ini_content = ["", "", "[transforms]"]
    
    # 1. Order 区块
    ini_content.append("[order]")
    for i in sorted(feature_order.keys()):
        ini_content.append(f"{i}={feature_order[i]}")
    ini_content.append("[/order]")
    
    # 2. Definitions 区块 (标准化参数)
    ini_content.append("[definitions]")
    
    # StandardScaler 参数写入
    s_std = scalers.get('standard')
    f_std = scalers.get('std_features', [])
    if s_std and f_std:
        indices = []
        valid_idx = []
        for i, name in enumerate(f_std):
            if name in feature_name_to_idx:
                indices.append(str(feature_name_to_idx[name]))
                valid_idx.append(i)
        
        if indices:
            ini_content.append("std_type=StandardScaler")
            ini_content.append("std_features=" + ",".join(indices))
            
            means = [f"{x:.6f}" for x in s_std.mean_[valid_idx]]
            ini_content.append("std_mean=" + ",".join(means))
            
            scales = [f"{x:.6f}" for x in s_std.scale_[valid_idx]]
            ini_content.append("std_scale=" + ",".join(scales))

    # RobustScaler 参数写入
    s_rob = scalers.get('robust')
    f_rob = scalers.get('rob_features', [])
    if s_rob and f_rob:
        indices = []
        valid_idx = []
        for i, name in enumerate(f_rob):
            if name in feature_name_to_idx:
                indices.append(str(feature_name_to_idx[name]))
                valid_idx.append(i)
        
        if indices:
            ini_content.append("") # 空行分隔
            ini_content.append("robust_type=RobustScaler")
            ini_content.append("robust_features=" + ",".join(indices))
            
            centers = [f"{x:.6f}" for x in s_rob.center_[valid_idx]]
            ini_content.append("robust_center=" + ",".join(centers))
            
            scales = [f"{x:.6f}" for x in s_rob.scale_[valid_idx]]
            ini_content.append("robust_scale=" + ",".join(scales))

    ini_content.append("[/definitions]")
    
    # 3. 启用变换
    ini_content.append("")
    ini_content.append("transform=true")
    ini_content.append("[/transforms]")
    
    # 追加到文件末尾
    with open(output_path, "ab") as f:
        f.write("\n".join(ini_content).encode('utf-8'))
    
    print(f"模型已保存至: {output_path} (包含完整预处理参数)")

def training_logger(period=100):
    def _callback(env):
        if period > 0 and (env.iteration + 1) % period == 0:
            msg = f"[迭代 {env.iteration + 1:5d}]"
            for data_name, eval_name, result, *rest in env.evaluation_result_list:
                msg += f" {data_name}-{eval_name}: {result:.4f}"
            print(msg)
    _callback.order = 10
    return _callback

def main():
    print_separator("Mihomo 极速权重模型训练器 (Speed & Stability First)")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    # 1. 解析特征
    try:
        go_content = fetch_go_source()
        parser_obj = GoTransformParser(go_content)
        feature_order = parser_obj.get_order()
    except Exception as e:
        print(f"错误: Go 源码解析失败: {e}")
        sys.exit(1)

    # 2. 加载数据
    try:
        # 默认只加载最近 14 天的数据，保证时效性
        df = load_data(args.data_dir, days=14)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

    # 3. 预处理 (应用极速策略)
    try:
        X, y, weights, scalers = preprocess_data(df, feature_order)
    except Exception as e:
        print(f"预处理失败: {e}")
        sys.exit(1)

    # 4. 划分数据集
    print("\n[步骤5] 划分训练集与验证集")
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, weights, test_size=0.15, random_state=42
    )
    print(f"训练集: {X_train.shape[0]} 条, 验证集: {X_val.shape[0]} 条")

    # 5. 训练
    print("\n[步骤6] 模型训练 (LightGBM)")
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        training_logger(period=200)
    ]

    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        callbacks=callbacks
    )

    if model.best_iteration_ < LGBM_PARAMS['n_estimators']:
         print(f"训练状态: 触发早停。最佳迭代轮数: [{model.best_iteration_}]")
    else:
         print(f"训练状态: 未触发早停 (跑满全量)。最佳迭代轮数: [{model.best_iteration_}]")

    # 6. 评估
    preds = model.predict(X_val)
    r2 = r2_score(y_val, preds)
    
    # 简单的线性映射评分 (0.0 - 10.0)
    # R2=0.5 -> 5.0分, R2=0.8 -> 8.0分
    final_score = max(0, r2 * 10)

    print(f"\n训练结束. 最佳迭代: {model.best_iteration_}")
    print(f"验证集 R2 得分: {r2:.4f}")
    print(f"模型最终评分: {final_score:.3f} / 10.0")
    
    if final_score > 8.0:
        print("✨ 评级: S级 (极佳) - 极速节点识别精准")
    elif final_score > 6.0:
        print("🟢 评级: A级 (良好) - 模型可用性高")
    elif final_score > 4.0:
        print("🟡 评级: B级 (及格) - 正常水平")
    elif final_score > 2.0:
        print("🟠 评级: C级 (一般) - 需积累更多数据")
    else:
        print("🔴 评级: D级 (不合格) - 噪声过大或数据不足")

    # 7. 保存
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink() # 删除旧文件
        
    save_model_and_params(model, scalers, feature_order, args.output)
    
    print_separator("完成")

if __name__ == "__main__":
    main()
