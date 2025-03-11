import pandas as pd
import numpy as np
from keras.models import load_model


# 向量化版本反映的是"单次修改敏感属性后的平均影响"

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

def calculate_prediction_change(model, original_sample, modified_sample):
    """计算预测是否发生变化"""
    original_pred = model.predict(original_sample.reshape(1, -1), verbose=0)[0]
    modified_pred = model.predict(modified_sample.reshape(1, -1), verbose=0)[0]
    
    # 对预测结果进行二值化（阈值0.5）
    original_label = 1 if original_pred >= 0.5 else 0
    modified_label = 1 if modified_pred >= 0.5 else 0
    
    # 返回是否发生变化
    return original_label != modified_label  # 确保这里返回的是布尔值

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

def calculate_discrimination_rate(dataset_name, num_samples=500):
    """计算歧视率（预测标签发生变化的比例） - 保证样本数量"""
    try:
        # 加载数据集和模型
        df = load_dataset(dataset_name)
        model = load_dnn_model(dataset_name)
        
        # 获取敏感属性
        sensitive_attrs = get_sensitive_attributes(dataset_name)
        print(f"敏感属性: {', '.join(sensitive_attrs)}")
        print(f"目标标签: {get_target_label(dataset_name)}")
        
        # 将数据集的列名转换为小写
        df.columns = df.columns.str.lower()
        sensitive_attrs = [attr.lower() for attr in sensitive_attrs]
        
        # 打印数据集列名和敏感属性列表
        print("数据集列名:", df.columns.tolist())
        print("敏感属性列表:", sensitive_attrs)
        
        for attr in sensitive_attrs:
            if attr in df.columns:
                print(f"可以访问: {attr}")
            else:
                print(f"访问失败: {attr}")
        
        # 获取目标标签名称并将其转为小写，确保与列名匹配
        target_label = get_target_label(dataset_name).lower()
        
        # 使用向量化方式采样所有样本
        # 是否需要放回抽样（replace=True）！！！！
        sample_df = df.sample(n=num_samples, replace=False)
        original_df = sample_df.drop(columns=[target_label])
        modified_df = original_df.copy()
        
        # 定义向量化函数用于修改敏感属性，确保新取值与原值不同
        def modify_column(series, possible_values):
            choices = np.random.choice(possible_values, size=len(series))
            mask = (choices == series.values)
            while mask.any():
                choices[mask] = np.random.choice(possible_values, size=mask.sum())
                mask = (choices == series.values)
            return choices
        
        # 对每个敏感属性进行向量化修改
        for attr in sensitive_attrs:
            if attr in modified_df.columns:
                possible_values = df[attr].unique()
                modified_df[attr] = modify_column(modified_df[attr], possible_values)
        
        original_array = original_df.values.astype(float)
        modified_array = modified_df.values.astype(float)
        
        original_preds = model.predict(original_array, verbose=0, batch_size=num_samples)
        modified_preds = model.predict(modified_array, verbose=0, batch_size=num_samples)
        
        original_labels = (original_preds >= 0.5).astype(int)
        modified_labels = (modified_preds >= 0.5).astype(int)
        if original_labels.ndim == 2:
            original_labels = original_labels[:, 0]
        if modified_labels.ndim == 2:
            modified_labels = modified_labels[:, 0]
        changes_count = int((original_labels != modified_labels).sum())

        # 计算歧视率
        discrimination_rate = changes_count / num_samples
        return discrimination_rate, changes_count, num_samples
    
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
            discrimination_rate, changes_count, total_samples = calculate_discrimination_rate(dataset)
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
        table_lines.append(f"\n\n随机种子 {seed} 的结果汇总：")
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
    print("\n\n" + "="*40 + " 所有测试结果 " + "="*40)
    
    # 显示每个随机种子的结果表格
    for seed in seeds:
        for line in seed_tables[seed]:
            print(line)
        print("\n")
    
    # 显示平均结果表格
    print("\n\n多随机种子平均结果汇总：")
    print("=" * 100)
    print(f"{'数据集':<15} {'平均歧视率':<10} {'平均变化样本数':<15} {'平均总样本数':<10} {'敏感属性'}")
    print("-" * 100)
    
    for result in avg_results:
        if result['discrimination_rate'] is not None:
            print(f"{result['dataset']:<15} {result['discrimination_rate']:>8.2%} {result['changes_count']:>15} {result['total_samples']:>12} {', '.join(result['sensitive_attrs'])}")
        else:
            print(f"{result['dataset']:<15} {'计算失败':<10} {'N/A':<15} {'N/A':<12} {', '.join(result['sensitive_attrs'])}")
    
    print("=" * 100)
    print(f"注：上述结果是基于 {len(seeds)} 个随机种子 ({', '.join(map(str, seeds))}) 的平均值")

if __name__ == "__main__":
    main() 