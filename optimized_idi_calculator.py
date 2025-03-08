import pandas as pd
import numpy as np
from keras.models import load_model
import random
from tqdm import tqdm

def load_dataset(dataset_name):
    """加载数据集"""
    print(f"\n加载数据集: {dataset_name}")
    if dataset_name == 'credit':
        return pd.read_csv('lab4数据/dataset/processed_credit_with_numerical.csv')
    return pd.read_csv(f'lab4数据/dataset/processed_{dataset_name}.csv')

def load_dnn_model(dataset_name):
    """加载对应的DNN模型"""
    print(f"加载模型: model_processed_{dataset_name}.h5")
    return load_model(f'lab4数据/DNN/model_processed_{dataset_name}.h5')

def get_sensitive_attributes(dataset_name):
    """获取数据集的敏感属性"""
    sensitive_attrs = {
        'adult': ['gender', 'race', 'age'],
        'compas': ['Sex', 'Race'],
        'law_school': ['male', 'race'],
        'kdd': ['sex', 'race'],
        'dutch': ['sex', 'age'],
        'credit': ['SEX', 'EDUCATION', 'MARRIAGE'],
        'communities_crime': ['Black', 'femalePctDiv'],
        'german': ['PersonStatusSex', 'AgeInYears']
    }
    return sensitive_attrs.get(dataset_name.lower(), [])

def get_target_label(dataset_name):
    """获取数据集的目标标签"""
    target_labels = {
        'adult': 'Class-label',
        'compas': 'Recidivism',
        'law_school': 'pass_bar',
        'kdd': 'income',
        'dutch': 'occupation',
        'credit': 'class',
        'communities_crime': 'class',
        'german': 'CREDITRATING'
    }
    return target_labels.get(dataset_name.lower())

def modify_continuous_attribute(df, attr, original_value, strategy='quantile'):
    """针对连续型属性的修改策略，确保修改后的值与原值不同"""
    max_attempts = 10  # 最大尝试次数
    attempts = 0
    
    while attempts < max_attempts:
        if strategy == 'quantile':
            quantiles = [0.25, 0.5, 0.75]
            new_value = df[attr].quantile(random.choice(quantiles))
        else:  # extreme
            if random.random() < 0.5:
                new_value = df[attr].min()
            else:
                new_value = df[attr].max()
        
        # 检查新值是否与原值不同（考虑浮点数精度）
        if not np.isclose(new_value, original_value, rtol=1e-05, atol=1e-08):
            return new_value
        attempts += 1
    
    # 如果多次尝试都未能找到不同的值，则返回最远的值
    all_values = df[attr].unique()
    distances = np.abs(all_values - original_value)
    return all_values[np.argmax(distances)]

def modify_discrete_attribute(df, attr, original_value):
    """针对离散型属性的修改策略，确保修改后的值与原值不同"""
    possible_values = df[attr].unique()
    
    # 如果只有一个可能的值，无法修改
    if len(possible_values) <= 1:
        return original_value
        
    # 移除原始值
    other_values = possible_values[possible_values != original_value]
    
    # 如果没有其他可选值，返回原值
    if len(other_values) == 0:
        return original_value
        
    # 随机选择一个不同的值
    return random.choice(other_values)

def get_prediction_probability(model, sample):
    """获取模型预测概率，确保结果在[0,1]范围内"""
    prob = model.predict(sample.reshape(1, -1), verbose=0)[0][0]
    # 确保预测概率在[0,1]范围内
    return np.clip(prob, 0, 1)

def is_prediction_changed(original_prob, modified_prob, threshold=0.5):
    """判断预测是否发生变化，包含预测概率的有效性检查"""
    # 确保预测概率在有效范围内
    if not (0 <= original_prob <= 1 and 0 <= modified_prob <= 1):
        print(f"警告：检测到无效的预测概率值 - 原始值: {original_prob}, 修改后: {modified_prob}")
        return False
    
    original_label = 1 if original_prob >= threshold else 0
    modified_label = 1 if modified_prob >= threshold else 0
    return original_label != modified_label

def calculate_optimized_discrimination_rate(dataset_name, target_samples=500):
    """计算优化后的歧视率"""
    try:
        # 设置随机种子
        random.seed(42)
        np.random.seed(42)
        
        # 加载数据集和模型
        df = load_dataset(dataset_name)
        model = load_dnn_model(dataset_name)
        
        # 获取敏感属性和目标标签
        sensitive_attrs = get_sensitive_attributes(dataset_name)
        target_label = get_target_label(dataset_name).lower()
        
        # 将列名转换为小写
        df.columns = df.columns.str.lower()
        sensitive_attrs = [attr.lower() for attr in sensitive_attrs]
        
        print(f"处理数据集: {dataset_name}")
        print(f"敏感属性: {', '.join(sensitive_attrs)}")
        
        changes_count = 0
        samples_processed = 0
        invalid_predictions = 0  # 记录无效预测的数量
        
        # 预处理：计算所有样本的预测概率
        X = df.drop(target_label, axis=1)
        probabilities = model.predict(X, verbose=0).flatten()
        probabilities = np.clip(probabilities, 0, 1)  # 确保所有概率值在[0,1]范围内
        
        # 选择预测概率在指定范围内的样本
        prob_mask_tight = (probabilities >= 0.4) & (probabilities <= 0.6)
        candidate_indices = np.where(prob_mask_tight)[0]
        
        # 如果样本不够，扩大范围
        if len(candidate_indices) < target_samples:
            prob_mask_wide = (probabilities >= 0.3) & (probabilities <= 0.7)
            candidate_indices = np.where(prob_mask_wide)[0]
        
        # 确保有足够的样本
        num_samples = min(len(candidate_indices), target_samples)
        
        progress_bar = tqdm(total=num_samples, desc="计算预测变化")
        
        for idx in np.random.choice(candidate_indices, num_samples, replace=False):
            original_sample = df.iloc[idx].drop(target_label)
            original_prob = probabilities[idx]
            
            # 修改所有敏感属性
            modified_sample = original_sample.copy()
            for attr in sensitive_attrs:
                current_value = modified_sample[attr]
                if df[attr].dtype in ['int64', 'float64']:
                    # 对连续型属性使用优化策略
                    modified_sample[attr] = modify_continuous_attribute(df, attr, current_value)
                else:
                    # 对离散型属性使用优化策略
                    modified_sample[attr] = modify_discrete_attribute(df, attr, current_value)
            
            # 获取修改后的预测概率
            modified_prob = get_prediction_probability(model, modified_sample.values.astype(float))
            
            # 检查预测是否发生变化
            if is_prediction_changed(original_prob, modified_prob):
                changes_count += 1
            
            samples_processed += 1
            progress_bar.update(1)
        
        progress_bar.close()
        
        # 打印无效预测的统计信息
        if invalid_predictions > 0:
            print(f"\n警告：检测到 {invalid_predictions} 个无效的预测结果")
        
        # 计算歧视率
        discrimination_rate = changes_count / samples_processed
        return discrimination_rate, changes_count, samples_processed
    
    except Exception as e:
        print(f"错误: {str(e)}")
        return None, None, None

def main():
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 测试所有数据集
    datasets = ['adult', 'compas', 'law_school', 'kdd', 'dutch', 
               'credit', 'communities_crime', 'german']
    
    results = []
    
    for dataset in datasets:
        print(f"\n{'='*20} 处理数据集: {dataset} {'='*20}")
        discrimination_rate, changes_count, total_samples = calculate_optimized_discrimination_rate(dataset)
        sensitive_attrs = get_sensitive_attributes(dataset)
        results.append({
            'dataset': dataset,
            'discrimination_rate': discrimination_rate,
            'changes_count': changes_count,
            'total_samples': total_samples,
            'sensitive_attrs': sensitive_attrs
        })
    
    print("\n\n优化后的结果汇总：")
    print("=" * 100)
    print(f"{'数据集':<15} {'歧视率':<10} {'变化样本数':<12} {'总样本数':<10} {'敏感属性'}")
    print("-" * 100)
    
    for result in results:
        if result['discrimination_rate'] is not None:
            print(f"{result['dataset']:<15} {result['discrimination_rate']:>8.2%} {result['changes_count']:>12} {result['total_samples']:>10} {', '.join(result['sensitive_attrs'])}")
        else:
            print(f"{result['dataset']:<15} {'计算失败':<10} {'N/A':<12} {'N/A':<10} {', '.join(result['sensitive_attrs'])}")
    
    print("=" * 100)
    print("\n注：")
    print("1. 优化后的歧视率基于以下改进：")
    print("   - 选择预测概率在[0.4,0.6]范围内的样本（如不够则扩展到[0.3,0.7]）")
    print("   - 对连续型敏感属性使用分位数或极值修改策略")
    print("   - 每个数据集使用最多500个精选样本")
    print("2. 预测标签发生变化表示模型对同一样本在修改敏感属性后的预测结果发生了改变")
    print("3. 歧视率越高，表示模型对敏感属性的依赖程度越高，公平性越差")

if __name__ == "__main__":
    main() 