import pandas as pd
import numpy as np
from keras.models import load_model
from tqdm import tqdm

def load_dataset(dataset_name):
    """加载数据集"""
    print(f"\n加载数据集: {dataset_name}")
    if dataset_name == 'credit':
        return pd.read_csv('lab4data/dataset/processed_credit_with_numerical.csv')
    return pd.read_csv(f'lab4data/dataset/processed_{dataset_name}.csv')

def load_dnn_model(dataset_name):
    """加载对应的DNN模型"""
    print(f"加载模型: model_processed_{dataset_name}.h5")
    return load_model(f'lab4data/DNN/model_processed_{dataset_name}.h5')

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
            new_value = df[attr].quantile(np.random.choice(quantiles))
        else:  # extreme
            if np.random.random() < 0.5:
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
    return np.random.choice(other_values)

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
        # 加载数据集和模型
        df = load_dataset(dataset_name)
        model = load_dnn_model(dataset_name)
        
        # 获取敏感属性和目标标签
        sensitive_attrs = get_sensitive_attributes(dataset_name)
        
        # 将列名转换为小写
        df.columns = df.columns.str.lower()
        sensitive_attrs = [attr.lower() for attr in sensitive_attrs]
        
        # 将目标标签转为小写，确保与列名匹配
        target_label = get_target_label(dataset_name).lower()
        
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
        modified_samples_list = []
        original_probs_list = []
        candidate_selected = np.random.choice(candidate_indices, num_samples, replace=False)
        for idx in candidate_selected:
            original_sample = df.iloc[idx].drop(target_label)
            original_prob = probabilities[idx]
            modified_sample = original_sample.copy()
            for attr in sensitive_attrs:
                current_value = modified_sample[attr]
                if df[attr].dtype in ['int64', 'float64']:
                    # 随机选择极值修改或分位数修改策略
                    strategy = np.random.choice(['quantile', 'extreme'])
                    modified_sample[attr] = modify_continuous_attribute(df, attr, current_value, strategy=strategy)
                else:
                    modified_sample[attr] = modify_discrete_attribute(df, attr, current_value)
            modified_samples_list.append(modified_sample.values.astype(float))
            original_probs_list.append(original_prob)
            progress_bar.update(1)
        progress_bar.close()

        modified_samples_array = np.vstack(modified_samples_list)
        modified_preds = model.predict(modified_samples_array, verbose=0, batch_size=num_samples).flatten()
        modified_preds = np.clip(modified_preds, 0, 1)
        original_probs_array = np.array(original_probs_list)

        original_labels = (original_probs_array >= 0.5).astype(int)
        modified_labels = (modified_preds >= 0.5).astype(int)
        changes_count = int((original_labels != modified_labels).sum())
        samples_processed = num_samples
        
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
    # 使用多个随机种子，减小误差
    seeds = [42, 2023, 10086, 888, 999]
    
    # 测试所有数据集
    datasets = ['adult', 'compas', 'law_school', 'kdd', 'dutch', 
               'credit', 'communities_crime', 'german']
    
    # 用于存储每个随机种子的结果
    all_results = {}
    
    # 用于存储平均结果
    avg_results = []
    
    # 用于存储每个随机种子的表格输出
    seed_tables = {}
    
    # 对每个随机种子进行一次完整实验
    for seed in seeds:
        print(f"\n\n{'#'*30} 使用随机种子: {seed} {'#'*30}")
        
        # 设置随机种子
        np.random.seed(seed)
        
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
        
        # 为当前随机种子存储结果
        all_results[seed] = results
        
        # 生成结果表格但不立即显示
        table_lines = []
        table_lines.append(f"\n\n随机种子 {seed} 的优化结果汇总：")
        table_lines.append("=" * 100)
        table_lines.append(f"{'数据集':<15} {'歧视率':<10} {'变化样本数':<12} {'总样本数':<10} {'敏感属性'}")
        table_lines.append("-" * 100)
        
        for result in results:
            if result['discrimination_rate'] is not None:
                table_lines.append(f"{result['dataset']:<15} {result['discrimination_rate']:>8.2%} {result['changes_count']:>12} {result['total_samples']:>10} {', '.join(result['sensitive_attrs'])}")
            else:
                table_lines.append(f"{result['dataset']:<15} {'计算失败':<10} {'N/A':<12} {'N/A':<10} {', '.join(result['sensitive_attrs'])}")
        
        table_lines.append("=" * 100)
        seed_tables[seed] = table_lines
    
    # 计算平均结果
    for dataset in datasets:
        valid_rates = []
        valid_changes = []
        valid_samples = []
        sensitive_attrs = None
        
        for seed in seeds:
            for result in all_results[seed]:
                if result['dataset'] == dataset and result['discrimination_rate'] is not None:
                    valid_rates.append(result['discrimination_rate'])
                    valid_changes.append(result['changes_count'])
                    valid_samples.append(result['total_samples'])
                    sensitive_attrs = result['sensitive_attrs']
        
        if valid_rates:
            avg_rate = sum(valid_rates) / len(valid_rates)
            avg_changes = sum(valid_changes) / len(valid_changes)
            avg_samples = sum(valid_samples) / len(valid_samples)
            
            avg_results.append({
                'dataset': dataset,
                'discrimination_rate': avg_rate,
                'changes_count': int(avg_changes),
                'total_samples': int(avg_samples),
                'sensitive_attrs': sensitive_attrs
            })
        else:
            avg_results.append({
                'dataset': dataset,
                'discrimination_rate': None,
                'changes_count': None,
                'total_samples': None,
                'sensitive_attrs': sensitive_attrs if sensitive_attrs else get_sensitive_attributes(dataset)
            })
    
    # 在所有测试完成后一起显示结果表格
    print("\n\n" + "="*40 + " 所有优化测试结果 " + "="*40)
    
    # 显示每个随机种子的结果表格
    for seed in seeds:
        for line in seed_tables[seed]:
            print(line)
        print("\n")
    
    # 显示平均结果表格
    print("\n\n多随机种子平均优化结果汇总：")
    print("=" * 100)
    print(f"{'数据集':<15} {'平均歧视率':<10} {'平均变化样本数':<15} {'平均总样本数':<10} {'敏感属性'}")
    print("-" * 100)
    
    for result in avg_results:
        if result['discrimination_rate'] is not None:
            print(f"{result['dataset']:<15} {result['discrimination_rate']:>8.2%} {result['changes_count']:>15} {result['total_samples']:>12} {', '.join(result['sensitive_attrs'])}")
        else:
            print(f"{result['dataset']:<15} {'计算失败':<10} {'N/A':<15} {'N/A':<12} {', '.join(result['sensitive_attrs'])}")
    
    print("=" * 100)
    print("\n注：")
    print("1. 优化后的歧视率基于以下改进：")
    print("   - 选择预测概率在[0.4,0.6]范围内的样本（如不够则扩展到[0.3,0.7]）")
    print("   - 对连续型敏感属性使用分位数或极值修改策略")
    print("   - 每个数据集使用最多500个精选样本")
    print("2. 预测标签发生变化表示模型对同一样本在修改敏感属性后的预测结果发生了改变")
    print("3. 歧视率越高，表示模型对敏感属性的依赖程度越高，公平性越差")
    print(f"4. 上述结果是基于 {len(seeds)} 个随机种子 ({', '.join(map(str, seeds))}) 的平均值")

if __name__ == "__main__":
    main() 