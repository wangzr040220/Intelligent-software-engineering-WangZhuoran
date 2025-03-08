#导入必要的库
#loading the necessary libraries
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import random
from tqdm import tqdm

#加载数据集
#loading the dataset
def load_dataset(dataset_name):
    """Load dataset"""
    print(f"\nLoading dataset: {dataset_name}")
    if dataset_name == 'credit':
        return pd.read_csv('lab4数据/dataset/processed_credit_with_numerical.csv')
    return pd.read_csv(f'lab4数据/dataset/processed_{dataset_name}.csv')

#加载DNN模型
#loading the DNN model
def load_dnn_model(dataset_name):
    """Load corresponding DNN model"""
    print(f"Loading model: model_processed_{dataset_name}.h5")
    return load_model(f'lab4数据/DNN/model_processed_{dataset_name}.h5')

#计算预测变化
#calculating the prediction change
def calculate_prediction_change(model, original_sample, modified_sample):
    """Calculate if prediction changes"""
    original_pred = model.predict(original_sample.reshape(1, -1), verbose=0)[0]
    modified_pred = model.predict(modified_sample.reshape(1, -1), verbose=0)[0]
    
    original_label = 1 if original_pred >= 0.5 else 0
    modified_label = 1 if modified_pred >= 0.5 else 0
    
    return original_label != modified_label

#获取敏感属性
#getting the sensitive attributes
def get_sensitive_attributes(dataset_name):
    """Get sensitive attributes of the dataset"""
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

#获取目标标签
#getting the target label
def get_target_label(dataset_name):
    """Get target label of the dataset"""
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

#计算歧视率
#calculating the discrimination rate
def calculate_discrimination_rate(dataset_name, num_samples=500):
    """Calculate discrimination rate (proportion of prediction label changes) - Ensure sample count"""
    try:
        random.seed(42)
        np.random.seed(42)
        
        df = load_dataset(dataset_name)
        model = load_dnn_model(dataset_name)
        
        sensitive_attrs = get_sensitive_attributes(dataset_name)
        print(f"Sensitive attributes: {', '.join(sensitive_attrs)}")
        print(f"Target label: {get_target_label(dataset_name)}")
        
        df.columns = df.columns.str.lower()
        sensitive_attrs = [attr.lower() for attr in sensitive_attrs]
        
        print("Dataset columns:", df.columns.tolist())
        print("Sensitive attribute list:", sensitive_attrs)
        
        for attr in sensitive_attrs:
            if attr in df.columns:
                print(f"Accessible: {attr}")
            else:
                print(f"Access failed: {attr}")
        
        target_label = get_target_label(dataset_name).lower()
        
        changes_count = 0
        unique_input_pairs = set()
        samples_collected = 0

        progress_bar = tqdm(total=num_samples, desc="Calculating prediction changes")
        while samples_collected < num_samples:
            original_idx = random.randint(0, len(df) - 1)
            original_sample = df.iloc[original_idx].drop(target_label)

            if sensitive_attrs:
                modified_sample = original_sample.copy()
                for attr in sensitive_attrs:
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

            if calculate_prediction_change(model, original_sample.values.astype(float),
                                        modified_sample.values.astype(float)):
                changes_count += 1
            samples_collected += 1
            progress_bar.update(1)
        progress_bar.close()

        discrimination_rate = changes_count / num_samples
        return discrimination_rate, changes_count, num_samples
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None, None

#主函数
#main function
def main():
    random.seed(42)
    np.random.seed(42)
    
    datasets = ['adult', 'compas', 'law_school', 'kdd', 'dutch', 'credit', 'communities_crime', 'german']
    
    results = []
    
    for dataset in datasets:
        print(f"\n{'='*20} Processing dataset: {dataset} {'='*20}")
        discrimination_rate, changes_count, total_samples = calculate_discrimination_rate(dataset)
        sensitive_attrs = get_sensitive_attributes(dataset)
        results.append({
            'dataset': dataset,
            'discrimination_rate': discrimination_rate,
            'changes_count': changes_count,
            'total_samples': total_samples,
            'sensitive_attrs': sensitive_attrs
        })
    
    print("\n\nFinal results summary:")
    print("=" * 100)
    print(f"{'Dataset':<15} {'Discrimination Rate':<10} {'Changed Samples':<12} {'Total Samples':<10} {'Sensitive Attributes'}")
    print("-" * 100)
    
    for result in results:
        if result['discrimination_rate'] is not None:
            print(f"{result['dataset']:<15} {result['discrimination_rate']:>8.2%} {result['changes_count']:>12} {result['total_samples']:>10} {', '.join(result['sensitive_attrs'])}")
        else:
            print(f"{result['dataset']:<15} {'Calculation Failed':<10} {'N/A':<12} {'N/A':<10} {', '.join(result['sensitive_attrs'])}")
    
    print("=" * 100)
    print("\nNote:")
    print("1. Discrimination Rate = Number of samples with changed prediction labels / Number of unique input pairs")
    print("2. A change in prediction label indicates that the model's prediction for the same sample changes after modifying sensitive attributes")
    print("3. A higher discrimination rate indicates a higher dependency of the model on sensitive attributes, implying lower fairness")
    print("4. Each dataset is tested with 1000 random samples")

if __name__ == "__main__":
    main()
