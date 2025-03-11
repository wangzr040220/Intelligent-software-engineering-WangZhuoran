import pandas as pd
import numpy as np
from keras.models import load_model


# The vectorized version reflects the "average impact after a single modification of sensitive attributes"

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

def calculate_prediction_change(model, original_sample, modified_sample):
    """Calculate whether the prediction changes"""
    original_pred = model.predict(original_sample.reshape(1, -1), verbose=0)[0]
    modified_pred = model.predict(modified_sample.reshape(1, -1), verbose=0)[0]
    
    # Binarize the prediction result (threshold 0.5)
    original_label = 1 if original_pred >= 0.5 else 0
    modified_label = 1 if modified_pred >= 0.5 else 0
    
    # Return whether there is a change
    return original_label != modified_label  # Ensure this returns a boolean value

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

def calculate_discrimination_rate(dataset_name, num_samples=500):
    """Calculate discrimination rate (proportion of prediction label changes) - Ensure sample size"""
    try:
        # Load dataset and model
        df = load_dataset(dataset_name)
        model = load_dnn_model(dataset_name)
        
        # Get sensitive attributes
        sensitive_attrs = get_sensitive_attributes(dataset_name)
        print(f"Sensitive attributes: {', '.join(sensitive_attrs)}")
        print(f"Target label: {get_target_label(dataset_name)}")
        
        # Convert dataset column names to lowercase
        df.columns = df.columns.str.lower()
        sensitive_attrs = [attr.lower() for attr in sensitive_attrs]
        
        # Print dataset column names and sensitive attribute list
        print("Dataset column names:", df.columns.tolist())
        print("Sensitive attribute list:", sensitive_attrs)
        
        for attr in sensitive_attrs:
            if attr in df.columns:
                print(f"Accessible: {attr}")
            else:
                print(f"Access failed: {attr}")
        
        # Get target label name and convert it to lowercase to ensure it matches the column name
        target_label = get_target_label(dataset_name).lower()
        
        # Use vectorized method to sample all samples
        # Whether to sample with replacement (replace=True)!!!!
        sample_df = df.sample(n=num_samples, replace=False)
        original_df = sample_df.drop(columns=[target_label])
        modified_df = original_df.copy()
        
        # Define a vectorized function to modify sensitive attributes, ensuring new values differ from original values
        def modify_column(series, possible_values):
            choices = np.random.choice(possible_values, size=len(series))
            mask = (choices == series.values)
            while mask.any():
                choices[mask] = np.random.choice(possible_values, size=mask.sum())
                mask = (choices == series.values)
            return choices
        
        # Vectorized modification for each sensitive attribute
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

        # Calculate discrimination rate
        discrimination_rate = changes_count / num_samples
        return discrimination_rate, changes_count, num_samples
    
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
            discrimination_rate, changes_count, total_samples = calculate_discrimination_rate(dataset)
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
        table_lines.append(f"\n\nSummary of results for random seed {seed}:")
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
    print("\n\n" + "="*40 + " All Test Results " + "="*40)
    
    # Display result table for each random seed
    for seed in seeds:
        for line in seed_tables[seed]:
            print(line)
        print("\n")
    
    # Display average result table
    print("\n\nSummary of average results for multiple random seeds:")
    print("=" * 100)
    print(f"{'Dataset':<15} {'Average Discrimination Rate':<10} {'Average Changed Samples':<15} {'Average Total Samples':<10} {'Sensitive Attributes'}")
    print("-" * 100)
    
    for result in avg_results:
        if result['discrimination_rate'] is not None:
            print(f"{result['dataset']:<15} {result['discrimination_rate']:>8.2%} {result['changes_count']:>15} {result['total_samples']:>12} {', '.join(result['sensitive_attrs'])}")
        else:
            print(f"{result['dataset']:<15} {'Calculation Failed':<10} {'N/A':<15} {'N/A':<12} {', '.join(result['sensitive_attrs'])}")
    
    print("=" * 100)
    print(f"Note: The above results are averages based on {len(seeds)} random seeds ({', '.join(map(str, seeds))})")

if __name__ == "__main__":
    main() 