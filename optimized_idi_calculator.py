import pandas as pd
import numpy as np
from keras.models import load_model
from tqdm import tqdm

def load_dataset(dataset_name):
    """Load dataset"""
    print(f"\nLoading dataset: {dataset_name}")
    if dataset_name == 'credit':
        return pd.read_csv('lab4data/dataset/processed_credit_with_numerical.csv')
    return pd.read_csv(f'lab4data/dataset/processed_{dataset_name}.csv')

def load_dnn_model(dataset_name):
    """Load corresponding DNN model"""
    print(f"Loading model: model_processed_{dataset_name}.h5")
    return load_model(f'lab4data/DNN/model_processed_{dataset_name}.h5')

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

def modify_continuous_attribute(df, attr, original_value, strategy='quantile'):
    """Modification strategy for continuous attributes, ensuring the modified value differs from the original"""
    max_attempts = 10  # Maximum number of attempts
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
        
        # Check if the new value differs from the original (considering floating-point precision)
        if not np.isclose(new_value, original_value, rtol=1e-05, atol=1e-08):
            return new_value
        attempts += 1
    
    # If multiple attempts fail to find a different value, return the furthest value
    all_values = df[attr].unique()
    distances = np.abs(all_values - original_value)
    return all_values[np.argmax(distances)]

def modify_discrete_attribute(df, attr, original_value):
    """Modification strategy for discrete attributes, ensuring the modified value differs from the original"""
    possible_values = df[attr].unique()
    
    # If there is only one possible value, cannot modify
    if len(possible_values) <= 1:
        return original_value
        
    # Remove the original value
    other_values = possible_values[possible_values != original_value]
    
    # If there are no other options, return the original value
    if len(other_values) == 0:
        return original_value
        
    # Randomly choose a different value
    return np.random.choice(other_values)

def get_prediction_probability(model, sample):
    """Get model prediction probability, ensuring the result is within [0,1]"""
    prob = model.predict(sample.reshape(1, -1), verbose=0)[0][0]
    # Ensure prediction probability is within [0,1]
    return np.clip(prob, 0, 1)

def is_prediction_changed(original_prob, modified_prob, threshold=0.5):
    """Determine if the prediction has changed, including validity check of prediction probabilities"""
    # Ensure prediction probabilities are within valid range
    if not (0 <= original_prob <= 1 and 0 <= modified_prob <= 1):
        print(f"Warning: Detected invalid prediction probability values - Original: {original_prob}, Modified: {modified_prob}")
        return False
    
    original_label = 1 if original_prob >= threshold else 0
    modified_label = 1 if modified_prob >= threshold else 0
    return original_label != modified_label

def calculate_optimized_discrimination_rate(dataset_name, target_samples=500):
    """Calculate optimized discrimination rate"""
    try:
        # Load dataset and model
        df = load_dataset(dataset_name)
        model = load_dnn_model(dataset_name)
        
        # Get sensitive attributes and target label
        sensitive_attrs = get_sensitive_attributes(dataset_name)
        
        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
        sensitive_attrs = [attr.lower() for attr in sensitive_attrs]
        
        # Convert target label to lowercase to ensure it matches the column name
        target_label = get_target_label(dataset_name).lower()
        
        print(f"Processing dataset: {dataset_name}")
        print(f"Sensitive attributes: {', '.join(sensitive_attrs)}")
        
        changes_count = 0
        samples_processed = 0
        invalid_predictions = 0  # Record the number of invalid predictions
        
        # Preprocessing: Calculate prediction probabilities for all samples
        X = df.drop(target_label, axis=1)
        probabilities = model.predict(X, verbose=0).flatten()
        probabilities = np.clip(probabilities, 0, 1)  # Ensure all probability values are within [0,1]
        
        # Select samples with prediction probabilities within the specified range
        prob_mask_tight = (probabilities >= 0.4) & (probabilities <= 0.6)
        candidate_indices = np.where(prob_mask_tight)[0]
        
        # If samples are insufficient, expand the range
        if len(candidate_indices) < target_samples:
            prob_mask_wide = (probabilities >= 0.3) & (probabilities <= 0.7)
            candidate_indices = np.where(prob_mask_wide)[0]
        
        # Ensure there are enough samples
        num_samples = min(len(candidate_indices), target_samples)
        
        progress_bar = tqdm(total=num_samples, desc="Calculating prediction changes")
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
                    # Randomly choose between extreme modification or quantile modification strategy
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
        
        # Print statistics of invalid predictions
        if invalid_predictions > 0:
            print(f"\nWarning: Detected {invalid_predictions} invalid prediction results")
        
        # Calculate discrimination rate
        discrimination_rate = changes_count / samples_processed
        return discrimination_rate, changes_count, samples_processed
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None, None

def main():
    # Use multiple random seeds to reduce error
    seeds = [42, 2023, 10086, 888, 999]
    
    # Test all datasets
    datasets = ['adult', 'compas', 'law_school', 'kdd', 'dutch', 
               'credit', 'communities_crime', 'german']
    
    # Store results for each random seed
    all_results = {}
    
    # Store average results
    avg_results = []
    
    # Store table output for each random seed
    seed_tables = {}
    
    # Conduct a complete experiment for each random seed
    for seed in seeds:
        print(f"\n\n{'#'*30} Using random seed: {seed} {'#'*30}")
        
        # Set random seed
        np.random.seed(seed)
        
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
        
        # Store results for the current random seed
        all_results[seed] = results
        
        # Generate result table but do not display immediately
        table_lines = []
        table_lines.append(f"\n\nSummary of optimized results for random seed {seed}:")
        table_lines.append("=" * 100)
        table_lines.append(f"{'Dataset':<15} {'Discrimination Rate':<10} {'Changed Samples':<12} {'Total Samples':<10} {'Sensitive Attributes'}")
        table_lines.append("-" * 100)
        
        for result in results:
            if result['discrimination_rate'] is not None:
                table_lines.append(f"{result['dataset']:<15} {result['discrimination_rate']:>8.2%} {result['changes_count']:>12} {result['total_samples']:>10} {', '.join(result['sensitive_attrs'])}")
            else:
                table_lines.append(f"{result['dataset']:<15} {'Calculation Failed':<10} {'N/A':<12} {'N/A':<10} {', '.join(result['sensitive_attrs'])}")
        
        table_lines.append("=" * 100)
        seed_tables[seed] = table_lines
    
    # Calculate average results
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
    
    # Display result tables after all tests are completed
    print("\n\n" + "="*40 + " All Optimized Test Results " + "="*40)
    
    # Display result table for each random seed
    for seed in seeds:
        for line in seed_tables[seed]:
            print(line)
        print("\n")
    
    # Display average result table
    print("\n\nSummary of average optimized results for multiple random seeds:")
    print("=" * 100)
    print(f"{'Dataset':<15} {'Average Discrimination Rate':<10} {'Average Changed Samples':<15} {'Average Total Samples':<10} {'Sensitive Attributes'}")
    print("-" * 100)
    
    for result in avg_results:
        if result['discrimination_rate'] is not None:
            print(f"{result['dataset']:<15} {result['discrimination_rate']:>8.2%} {result['changes_count']:>15} {result['total_samples']:>12} {', '.join(result['sensitive_attrs'])}")
        else:
            print(f"{result['dataset']:<15} {'Calculation Failed':<10} {'N/A':<15} {'N/A':<12} {', '.join(result['sensitive_attrs'])}")
    
    print("=" * 100)
    print("\nNote:")
    print("1. The optimized discrimination rate is based on the following improvements:")
    print("   - Select samples with prediction probabilities in the range [0.4,0.6] (expand to [0.3,0.7] if insufficient)")
    print("   - Use quantile or extreme modification strategy for continuous sensitive attributes")
    print("   - Use up to 500 selected samples per dataset")
    print("2. A change in prediction label indicates that the model's prediction result for the same sample changed after modifying sensitive attributes")
    print("3. The higher the discrimination rate, the more the model depends on sensitive attributes, indicating poorer fairness")
    print(f"4. The above results are averages based on {len(seeds)} random seeds ({', '.join(map(str, seeds))})")

if __name__ == "__main__":
    main() 