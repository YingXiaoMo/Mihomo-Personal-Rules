# train_smart.py
# ==============================================================================
# Mihomo 智能权重模型训练
# 功能：基于历史数据训练 LightGBM 回归模型，用于预测代理节点权重
# ==============================================================================



import re
import os
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, List, Optional

class GoTransformParser:
    
    def __init__(self, go_file_path: str):
        try:
            with open(go_file_path, 'r', encoding='utf-8') as f:
                self.content = f.read()
            print(f"成功加载 Go 源文件: {go_file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Go 源文件 '{go_file_path}' 没找到。请确保文件存在于正确路径中。"
            )
        
        self.feature_order = self._parse_feature_order()
    
    def _parse_feature_order(self) -> List[str]:
        print("开始解析 getDefaultFeatureOrder 函数...")
        
        function_pattern = r'func getDefaultFeatureOrder\(\) map\[int\]string \{\s*return map\[int\]string\{(.*?)\}\s*\}'
        match = re.search(function_pattern, self.content, re.DOTALL)
        
        if not match:
            print("警告: 没找到 getDefaultFeatureOrder 函数，使用预定义特征顺序")
            return self._get_fallback_feature_order()
        
        function_body = match.group(1)
        feature_pairs = re.findall(r'(\d+):\s*"([^"]+)"', function_body)
        
        if not feature_pairs:
            print("警告: 函数体中无有效特征定义，使用预定义特征顺序")
            return self._get_fallback_feature_order()
        
        feature_dict = {int(index): name for index, name in feature_pairs}
        sorted_features = [feature_dict[i] for i in sorted(feature_dict.keys())]
        
        print(f"成功解析 {len(sorted_features)} 个特征的顺序定义")
        return sorted_features
    
    def get_feature_order(self) -> List[str]:
        return self.feature_order
    
    def _get_fallback_feature_order(self) -> List[str]:
        return [
            'success', 'failure', 'connect_time', 'latency', 'upload_mb', 'download_mb', 
            'duration_minutes', 'last_used_seconds', 'is_udp', 'is_tcp', 'asn_feature', 
            'country_feature', 'address_feature', 'port_feature', 'traffic_ratio', 
            'traffic_density', 'connection_type_feature', 'asn_hash', 'host_hash', 
            'ip_hash', 'geoip_hash'
        ]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GO_FILE = os.path.join(BASE_DIR, "../go_transform/transform.go")
MODEL_FILE = os.path.join(BASE_DIR, "../../models/Model.bin")
DATA_DIR = os.path.join(BASE_DIR, "../../data") 

STD_SCALER_FEATURES = [
    'connect_time', 'latency', 'upload_mb', 'download_mb', 'duration_minutes', 
    'last_used_seconds', 'traffic_density'
]

ROBUST_SCALER_FEATURES = ['success', 'failure']

LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'random_state': 42,
    'n_jobs': -1
}

EARLY_STOPPING_ROUNDS = 100


def load_and_clean_data(file_path: str) -> Optional[pd.DataFrame]:
    print(f"尝试加载文件: {os.path.basename(file_path)}...")
    
    try:
        data = pd.read_csv(file_path, on_bad_lines='skip')
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(file_path, on_bad_lines='skip', encoding='gbk')
            print(f"警告: 文件 '{os.path.basename(file_path)}' 使用 GBK 编码成功加载。")
        except UnicodeDecodeError:
            try:
                data = pd.read_csv(file_path, on_bad_lines='skip', encoding='latin-1')
                print(f"警告: 文件 '{os.path.basename(file_path)}' 使用 latin-1 编码成功加载。")
            except Exception as e:
                print(f"数据加载失败 (多种编码尝试后仍失败): {e}")
                return None
        except Exception as e:
            print(f"数据加载失败 (尝试 GBK 编码后仍失败): {e}")
            return None
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

    original_count = len(data)
    
    data.dropna(subset=['weight'], inplace=True)
    
    data = data[data['weight'] > 0].copy()
    
    final_count = len(data)
    filtered_count = original_count - final_count
    
    print(f"文件处理完成: {original_count} → {final_count} 条记录 (过滤 {filtered_count} 条)")
    return data

def load_all_data_from_directory(data_dir: str) -> Optional[pd.DataFrame]:
    print(f"开始从数据目录加载所有 CSV 文件: {data_dir}")
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if not csv_files:
        print("错误: 在指定目录中没有找到任何 CSV 文件。")
        return None
    
    print(f"--- 找到 {len(csv_files)} 个数据文件 ---")
    
    all_data = []
    
    for file_path in csv_files:
        df = load_and_clean_data(file_path)
        if df is not None:
            all_data.append(df)
            
    if not all_data:
        print("错误: 所有数据文件加载或清洗失败，无法继续训练。")
        return None
        
    print("\n合并所有数据文件...")
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"数据合并完成，总记录数: {len(combined_data)}")
    return combined_data

def extract_features_from_preprocessed(data: pd.DataFrame, feature_order: List[str]) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    print("开始构建特征矩阵和目标变量...")
    
    try:
        X = data[feature_order]
        y = data['weight']
        
        print(f"特征提取完成 - 特征矩阵形状: {X.shape}, 目标变量形状: {y.shape}")
        return X, y
        
    except KeyError as e:
        print(f"特征提取失败: 缺少必要的特征列 {e}")
        available_columns = list(data.columns)
        print(f"数据中可用的列: {available_columns}")
        return None, None

def apply_feature_transforms(X: pd.DataFrame, feature_order: List[str]) -> Tuple[pd.DataFrame, StandardScaler, RobustScaler]:
    print("开始特征标准化处理...")
    X_transformed = X.copy()
    
    std_scaler = StandardScaler()
    std_features_available = [f for f in STD_SCALER_FEATURES if f in X_transformed.columns]
    
    if std_features_available:
        X_transformed[std_features_available] = std_scaler.fit_transform(X_transformed[std_features_available])
        print(f"StandardScaler 处理完成，影响特征数: {len(std_features_available)}")
    
    robust_scaler = RobustScaler()
    robust_features_available = [f for f in ROBUST_SCALER_FEATURES if f in X_transformed.columns]
    
    if robust_features_available:
        X_transformed[robust_features_available] = robust_scaler.fit_transform(X_transformed[robust_features_available])
        print(f"RobustScaler 处理完成，影响特征数: {len(robust_features_available)}")
    
    return X_transformed, std_scaler, robust_scaler

def train_lgbm_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> lgb.LGBMRegressor:
    print("开始 LightGBM 模型训练...")
    
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
    )
    
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    
    print(f"模型训练完成")
    print(f"训练集R²得分: {train_r2:.4f}")
    print(f"测试集R²得分: {test_r2:.4f}")
    
    if test_r2 > 0.8:
        print("模型性能评估: 优秀")
    elif test_r2 > 0.6:
        print("模型性能评估: 良好")
    else:
        print("模型性能评估: 需要改进")
    
    return model

def save_model(model: lgb.LGBMRegressor, model_file: str) -> None:
    print(f"开始保存模型至: {model_file}")
    
    output_dir = os.path.dirname(model_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"目标目录已创建: {output_dir}")
    
    try:
        model.booster_.save_model(model_file)
        print("模型保存成功，可以直接部署")
    except Exception as e:
        print(f"模型保存失败: {e}")

def main() -> None:
    print("=" * 60)
    print("Mihomo 智能权重模型训练")
    print("=" * 60)
    
    print("\n[步骤1] Go 源码解析")
    try:
        parser = GoTransformParser(GO_FILE)
        feature_order = parser.get_feature_order()
        print(f"特征顺序解析完成，共 {len(feature_order)} 个特征")
    except Exception as e:
        print(f"Go 源码解析失败: {e}")
        print("程序终止")
        return
    
    print("\n[步骤2] 数据加载与清洗")
    dataset = load_all_data_from_directory(DATA_DIR)
    if dataset is None:
        print("数据加载失败，程序终止")
        return
    
    print("\n[步骤3] 特征提取")
    extraction_result = extract_features_from_preprocessed(dataset, feature_order)
    if extraction_result[0] is None:
        print("特征提取失败，程序终止")
        return
    
    X, y = extraction_result
    
    print("\n[步骤4] 特征标准化")
    X_processed, std_scaler, robust_scaler = apply_feature_transforms(X, feature_order)
    
    print("\n[步骤5] 训练测试集划分")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, 
        test_size=0.2,
        random_state=42
    )
    print(f"数据划分完成 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    print("\n[步骤6] 模型训练")
    trained_model = train_lgbm_model(X_train, y_train, X_test, y_test)
    
    print("\n[步骤7] 模型保存")
    save_model(trained_model, MODEL_FILE)
    
    print("\n" + "=" * 60)
    print("模型训练流程完成")
    print(f"输出文件: {MODEL_FILE}")
    print("模型可进行生产环境部署")
    print("=" * 60)

if __name__ == "__main__":
    main()
