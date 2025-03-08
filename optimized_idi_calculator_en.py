import pandas as pd
import numpy as np
from keras.models import load_model
import random
from tqdm import tqdm

#加载数据集
#loading the dataset
def load_dataset(dataset_name):
    print(f"\nLoading dataset: {dataset_name}")
    if dataset_name == 'credit':
        return pd.read_csv('lab4数据/dataset/processed_credit_with_numerical.csv')
    return pd.read_csv(f'lab4数据/dataset/processed_{dataset_name}.csv')

#加载DNN模型
#loading the DNN model
def load_dnn_model(dataset_name):
    print(f"Loading model: model_processed_{dataset_name}.h5")
    return load_model(f'lab4数据/DNN/model_processed_{dataset_name}.h5')

#获取敏感属性
#getting the sensitive attributes
def get_sensitive_attributes(dataset_name):
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

#修改连续型敏感属性
#modifying the continuous sensitive attributes
def modify_continuous_attribute(df, attr, original_value, strategy='quantile'):
    max_attempts = 10
    attempts = 0
    
    while attempts < max_attempts:
        if strategy == 'quantile':
            quantiles = [0.25, 0.5, 0.75]
            new_value = df[attr].quantile(random.choice(quantiles))
        else:
            if random.random() < 0.5:
                new_value = df[attr].min()
            else:
                new_value = df[attr].max()
        
        if not np.isclose(new_value, original_value, rtol=1e-05, atol=1e-08):
            return new_value
        attempts += 1
    
    all_values = df[attr].unique()
    distances = np.abs(all_values - original_value)
    return all_values[np.argmax(distances)]

#修改离散型敏感属性
#modifying the discrete sensitive attributes
def modify_discrete_attribute(df, attr, original_value):
    possible_values = df[attr].unique()
    
    if len(possible_values) <= 1:
        return original_value
        
    other_values = possible_values[possible_values != original_value]
    
    if len(other_values) == 0:
        return original_value
        
    return random.choice(other_values)

#获取预测概率
#getting the prediction probability
def get_prediction_probability(model, sample):
    prob = model.predict(sample.reshape(1, -1), verbose=0)[0][0]
    return np.clip(prob, 0, 1)

#判断预测是否变化
#prediction change
def is_prediction_changed(original_prob, modified_prob, threshold=0.5):
    if not (0 <= original_prob <= 1 and 0 <= modified_prob <= 1):
        print(f"Warning: Invalid prediction probabilities detected - Original: {original_prob}, Modified: {modified_prob}")
        return False
    
    original_label = 1 if original_prob >= threshold else 0
    modified_label = 1 if modified_prob >= threshold else 0
    return original_label != modified_label

#计算优化后的歧视率
#calculating the optimized discrimination rate
def calculate_optimized_discrimination_rate(dataset_name, target_samples=500):
    try:
        random.seed(42)
        np.random.seed(42)
        
        df = load_dataset(dataset_name)
        model = load_dnn_model(dataset_name)
        
        sensitive_attrs = get_sensitive_attributes(dataset_name)
        target_label = get_target_label(dataset_name).lower()
        
        df.columns = df.columns.str.lower()
        sensitive_attrs = [attr.lower() for attr in sensitive_attrs]
        
        print(f"Processing dataset: {dataset_name}")
        print(f"Sensitive attributes: {', '.join(sensitive_attrs)}")
        
        changes_count = 0
        samples_processed = 0
        invalid_predictions = 0
        
        # 预处理：计算所有样本的预测概率
        #preprocessing: calculate the prediction probability of all samples
        X = df.drop(target_label, axis=1)
        probabilities = model.predict(X, verbose=0).flatten()
        probabilities = np.clip(probabilities, 0, 1)
        
        # 选择预测概率在[0.4,0.6]范围内的样本
        #select the samples with prediction probabilities in the range [0.4,0.6]
        prob_mask_tight = (probabilities >= 0.4) & (probabilities <= 0.6)
        candidate_indices = np.where(prob_mask_tight)[0]
        
        if len(candidate_indices) < target_samples:
            # 如果数量不足，扩展到[0.3,0.7]范围
            #if the number is insufficient, expand to the range [0.3,0.7]
            prob_mask_wide = (probabilities >= 0.3) & (probabilities <= 0.7)
            candidate_indices = np.where(prob_mask_wide)[0]
        
        num_samples = min(len(candidate_indices), target_samples)
        
        progress_bar = tqdm(total=num_samples, desc="Calculating prediction changes")
        
        for idx in np.random.choice(candidate_indices, num_samples, replace=False):
            original_sample = df.iloc[idx].drop(target_label)
            original_prob = probabilities[idx]
            
            modified_sample = original_sample.copy()
            #针对敏感属性进行修改
            #modifying the sensitive attributes
            for attr in sensitive_attrs:
                current_value = modified_sample[attr]
                if df[attr].dtype in ['int64', 'float64']:
                    #修改连续型敏感属性
                    #modifying the continuous sensitive attributes
                    modified_sample[attr] = modify_continuous_attribute(df, attr, current_value)
                else:
                    #修改离散型敏感属性
                    #modifying the discrete sensitive attributes
                    modified_sample[attr] = modify_discrete_attribute(df, attr, current_value)
            
            modified_prob = get_prediction_probability(model, modified_sample.values.astype(float))
            
            if is_prediction_changed(original_prob, modified_prob):
                changes_count += 1
            
            samples_processed += 1
            progress_bar.update(1)
        
        progress_bar.close()
        
        if invalid_predictions > 0:
            print(f"\nWarning: Detected {invalid_predictions} invalid predictions")
        
        discrimination_rate = changes_count / samples_processed
        return discrimination_rate, changes_count, samples_processed
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None, None

#主函数
#main function
def main():
    random.seed(42)
    np.random.seed(42)
    
    datasets = ['adult', 'compas', 'law_school', 'kdd', 'dutch', 
               'credit', 'communities_crime', 'german']
    
    results = []
    
    for dataset in datasets:
        print(f"\n{'='*20} Processing dataset: {dataset} {'='*20}")
        discrimination_rate, changes_count, total_samples = calculate_optimized_discrimination_rate(dataset)
        sensitive_attrs = get_sensitive_attributes(dataset)
        results.append({
            'dataset': dataset,
            'discrimination_rate': discrimination_rate,
            'changes_count': changes_count,
            'total_samples': total_samples,
            'sensitive_attrs': sensitive_attrs
        })
    
    print("\n\nOptimized results summary:")
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
    print("1. The optimized discrimination rate is based on the following improvements:")
    print("   - Selecting samples with prediction probabilities in the range [0.4,0.6] (expanded to [0.3,0.7] if insufficient)")
    print("   - Using quantile or extreme value modification strategies for continuous sensitive attributes")
    print("   - Using a maximum of 500 selected samples per dataset")
    print("2. A change in prediction label indicates that the model's prediction for the same sample changes after modifying sensitive attributes")
    print("3. A higher discrimination rate indicates a higher dependency of the model on sensitive attributes, implying lower fairness")

if __name__ == "__main__":
    main()
