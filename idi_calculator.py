import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
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
        # 设置随机种子
        random.seed(42)
        np.random.seed(42)
        
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
        
        # 获取目标标签名称并删除对应的列
        target_label = get_target_label(dataset_name).lower()
        
        # 存储预测变化的计数
        changes_count = 0
        unique_input_pairs = set()  # 初始化一个集合来存储唯一的输入对
        samples_collected = 0  # 记录已收集的唯一样本数量

        # 使用 while 循环，直到收集到足够数量的唯一样本
        progress_bar = tqdm(total=num_samples, desc="计算预测变化")  # 初始化进度条
        while samples_collected < num_samples:
            # 随机选择一个样本
            original_idx = random.randint(0, len(df) - 1)
            original_sample = df.iloc[original_idx].drop(target_label)

            # 修改所有敏感属性
            if sensitive_attrs:
                modified_sample = original_sample.copy()
                for attr in sensitive_attrs:  # 遍历所有敏感属性
                    if df[attr].dtype in ['int64', 'float64']:
                        possible_values = df[attr].unique()
                        current_value = modified_sample[attr]
                        new_value = current_value
                        while new_value == current_value:
                            new_value = random.choice(possible_values)
                        modified_sample[attr] = new_value
                    else:
                        possible_values = df[attr].unique()
                        current_value = modified_sample[attr]
                        new_value = current_value
                        while new_value == current_value:
                            new_value = random.choice(possible_values)
                        modified_sample[attr] = new_value

            # 检查预测是否发生变化
            if calculate_prediction_change(model, original_sample.values.astype(float),
                                        modified_sample.values.astype(float)):
                changes_count += 1
            samples_collected += 1  # 只有当是新样本时才增加计数器
            progress_bar.update(1)  # 更新进度条
        progress_bar.close()  # 关闭进度条

        # 计算歧视率
        discrimination_rate = changes_count / num_samples
        return discrimination_rate, changes_count, num_samples
    
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
        discrimination_rate, changes_count, total_samples = calculate_discrimination_rate(dataset)
        sensitive_attrs = get_sensitive_attributes(dataset)
        results.append({
            'dataset': dataset,
            'discrimination_rate': discrimination_rate,
            'changes_count': changes_count,
            'total_samples': total_samples,
            'sensitive_attrs': sensitive_attrs
        })
    
    print("\n\n最终结果汇总：")
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
    print("1. 歧视率 = 预测标签发生变化的样本数 / 唯一输入对的数量")
    print("2. 预测标签发生变化表示模型对同一样本在修改敏感属性后的预测结果发生了改变")
    print("3. 歧视率越高，表示模型对敏感属性的依赖程度越高，公平性越差")
    print("4. 每个数据集使用1000个随机样本进行测试")

if __name__ == "__main__":
    main() 