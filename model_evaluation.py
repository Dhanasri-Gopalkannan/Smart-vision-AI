"""
Transfer Learning - Model Comparison & Selection
Step 2.5: Compare all 4 models and select the best one
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from scipy import stats

import tensorflow as tf
from tensorflow import keras
from tensorflow import keras
import tensorflow as tf

# Configuration
class Config:
    TEST_DIR = "data/cropped_images/test"
    NUM_CLASSES = 25
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    SAVE_DIR = "saved_models"
    RESULTS_DIR = "results"
    CLASS_NAMES = [f"Class_{i}" for i in range(25)]  # Replace with actual class names

# Create results directory
os.makedirs(Config.RESULTS_DIR, exist_ok=True)

# Load test data
def load_test_data():
    """Load test dataset"""
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    test_data = test_datagen.flow_from_directory(
        Config.TEST_DIR,
        target_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Important for evaluation
    )
    
    return test_data

# Model loading functions
def load_model(model_name):
    """Load a specific model"""
    model_path = os.path.join(Config.SAVE_DIR, f"{model_name.lower()}_final.h5")
    
    if not os.path.exists(model_path):
        model_path = os.path.join(Config.SAVE_DIR, f"{model_name.lower()}_best.h5")
    
    if os.path.exists(model_path):
        print(f"Loading {model_name} from {model_path}")
        model = keras.models.load_model(model_path)
        return model
    else:
        print(f"Model file not found: {model_path}")
        return None

# Evaluation metrics
def evaluate_model(model, test_data, model_name):
    """Comprehensive model evaluation"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    # Get true labels
    y_true = test_data.classes
    y_true_one_hot = test_data.labels
    
    # Predict
    start_time = time.time()
    y_pred_proba = model.predict(test_data, verbose=1)
    inference_time = time.time() - start_time
    
    # Get predicted classes
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_true)
    
    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Top-5 accuracy
    top5_accuracy = tf.keras.metrics.top_k_categorical_accuracy(
        y_true_one_hot, y_pred_proba, k=5
    ).numpy().mean()
    
    # Calculate inference time per image
    num_images = len(y_true)
    avg_inference_time = inference_time / num_images
    
    # Model size
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=Config.CLASS_NAMES, output_dict=True)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'top5_accuracy': top5_accuracy,
        'inference_time_total': inference_time,
        'avg_inference_time': avg_inference_time,
        'model_size_mb': model_size,
        'num_params': model.count_params(),
        'confusion_matrix': cm,
        'classification_report': report,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

# Visualization functions
def plot_confusion_matrices(results_dict, test_data):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.ravel()
    
    class_names = list(test_data.class_indices.keys())
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        cm = results['confusion_matrix']
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'{model_name} - Confusion Matrix\nAccuracy: {results["accuracy"]:.3f}')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add labels
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=90, fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                       horizontalalignment="center",
                       color="white" if cm_normalized[i, j] > thresh else "black",
                       fontsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_metrics_comparison(results_dict):
    """Create comparison visualizations"""
    # Extract metrics
    models = list(results_dict.keys())
    accuracies = [results_dict[m]['accuracy'] for m in models]
    f1_scores = [results_dict[m]['f1_score'] for m in models]
    top5_acc = [results_dict[m]['top5_accuracy'] for m in models]
    inference_times = [results_dict[m]['avg_inference_time'] for m in models]
    model_sizes = [results_dict[m]['model_size_mb'] for m in models]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Accuracy Comparison
    ax = axes[0, 0]
    bars = ax.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: F1-Score Comparison
    ax = axes[0, 1]
    bars = ax.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1-Score')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Top-5 Accuracy
    ax = axes[0, 2]
    bars = ax.bar(models, top5_acc, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_title('Top-5 Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Top-5 Accuracy')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, top5 in zip(bars, top5_acc):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{top5:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Inference Time Comparison
    ax = axes[1, 0]
    bars = ax.bar(models, inference_times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_title('Average Inference Time per Image', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time (seconds)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, time_val in zip(bars, inference_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{time_val:.4f}s', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Model Size Comparison
    ax = axes[1, 1]
    bars = ax.bar(models, model_sizes, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Size (MB)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, size in zip(bars, model_sizes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{size:.1f}MB', ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Accuracy vs Inference Time (Scatter plot)
    ax = axes[1, 2]
    scatter = ax.scatter(inference_times, accuracies, s=model_sizes, 
                        c=range(len(models)), cmap='viridis', alpha=0.6, edgecolors='black')
    
    # Add model labels
    for i, model in enumerate(models):
        ax.annotate(model, (inference_times[i], accuracies[i]), 
                   xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    ax.set_xlabel('Inference Time (seconds)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Inference Time Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create radar chart for comprehensive comparison
    plot_radar_chart(results_dict)

def plot_radar_chart(results_dict):
    """Create radar chart for model comparison"""
    metrics = ['accuracy', 'f1_score', 'top5_accuracy', 'avg_inference_time', 'model_size_mb']
    metric_labels = ['Accuracy', 'F1-Score', 'Top-5 Acc', 'Inference\nTime', 'Model\nSize']
    
    # Normalize metrics (lower is better for inference time and model size)
    normalized_data = {}
    for model_name, results in results_dict.items():
        normalized = []
        for metric in metrics:
            value = results[metric]
            if metric in ['avg_inference_time', 'model_size_mb']:
                # Invert for radar chart (lower is better)
                normalized.append(1 / (value + 1e-6))
            else:
                normalized.append(value)
        normalized_data[model_name] = normalized
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (model_name, values) in enumerate(normalized_data.items()):
        values += values[:1]  # Close the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim([0, 1])
    ax.set_title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, 'radar_chart.png'), dpi=300, bbox_inches='tight')
    plt.show()

# Statistical analysis
def perform_statistical_analysis(results_dict, test_data):
    """Perform statistical tests to determine significant differences"""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    # Collect accuracy scores for statistical tests
    accuracy_scores = {}
    for model_name, results in results_dict.items():
        # Calculate per-sample accuracy
        y_true = results['y_true']
        y_pred = results['y_pred']
        accuracy_scores[model_name] = (y_pred == y_true).astype(int)
    
    # Perform McNemar's test for paired comparisons
    models = list(results_dict.keys())
    n_models = len(models)
    
    print("\nMcNemar's Test Results (p-values):")
    mcnemar_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            # Create contingency table
            model1_correct = accuracy_scores[models[i]]
            model2_correct = accuracy_scores[models[j]]
            
            both_correct = np.sum((model1_correct == 1) & (model2_correct == 1))
            both_wrong = np.sum((model1_correct == 0) & (model2_correct == 0))
            model1_only = np.sum((model1_correct == 1) & (model2_correct == 0))
            model2_only = np.sum((model1_correct == 0) & (model2_correct == 1))
            
            # Calculate McNemar's test statistic
            chi2 = (abs(model1_only - model2_only) - 1)**2 / (model1_only + model2_only)
            p_value = 1 - stats.chi2.cdf(chi2, 1)
            
            mcnemar_matrix[i, j] = p_value
            mcnemar_matrix[j, i] = p_value
            
            print(f"{models[i]} vs {models[j]}: p = {p_value:.4f} {'*' if p_value < 0.05 else ''}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mcnemar_matrix, cmap='RdYlGn_r')
    
    # Add text annotations
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                text = ax.text(j, i, f'{mcnemar_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels(models, rotation=45)
    ax.set_yticklabels(models)
    ax.set_title("McNemar's Test p-values\n(Statistical Significance of Differences)", fontweight='bold')
    plt.colorbar(im, ax=ax, label='p-value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, 'statistical_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

# Model selection
def select_best_model(results_dict):
    """Select the best model based on accuracy-speed tradeoff"""
    print("\n" + "="*60)
    print("MODEL SELECTION ANALYSIS")
    print("="*60)
    
    # Calculate weighted scores
    models_data = []
    
    for model_name, results in results_dict.items():
        # Normalize metrics (0-1 scale)
        acc_norm = results['accuracy']
        f1_norm = results['f1_score']
        
        # Invert time and size (lower is better)
        time_norm = 1 / (results['avg_inference_time'] + 1e-6)
        size_norm = 1 / (results['model_size_mb'] + 1e-6)
        
        # Define weights (customize based on priorities)
        weights = {
            'accuracy': 0.4,      # Highest priority
            'f1_score': 0.3,      # Important for balanced performance
            'inference_time': 0.2, # Speed considerations
            'model_size': 0.1      # Deployment considerations
        }
        
        # Calculate composite score
        composite_score = (
            weights['accuracy'] * acc_norm +
            weights['f1_score'] * f1_norm +
            weights['inference_time'] * time_norm +
            weights['model_size'] * size_norm
        )
        
        models_data.append({
            'model': model_name,
            'accuracy': results['accuracy'],
            'f1_score': results['f1_score'],
            'inference_time': results['avg_inference_time'],
            'model_size': results['model_size_mb'],
            'composite_score': composite_score
        })
    
    # Create DataFrame and sort
    df_comparison = pd.DataFrame(models_data)
    df_comparison = df_comparison.sort_values('composite_score', ascending=False)
    
    # Display results
    print("\nModel Ranking (based on composite score):")
    print(df_comparison.to_string(index=False))
    
    # Get best model
    best_model = df_comparison.iloc[0]
    
    print(f"\n{'='*60}")
    print(f"SELECTED BEST MODEL: {best_model['model']}")
    print(f"{'='*60}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"F1-Score: {best_model['f1_score']:.4f}")
    print(f"Inference Time: {best_model['inference_time']:.4f} seconds per image")
    print(f"Model Size: {best_model['model_size']:.1f} MB")
    print(f"Composite Score: {best_model['composite_score']:.4f}")
    
    # Save selection to file
    selection_report = {
        'selected_model': best_model['model'],
        'selection_criteria': {
            'weights': {
                'accuracy': 0.4,
                'f1_score': 0.3,
                'inference_time': 0.2,
                'model_size': 0.1
            },
            'timestamp': datetime.now().isoformat()
        },
        'model_details': best_model.to_dict(),
        'all_models': df_comparison.to_dict('records')
    }
    
    with open(os.path.join(Config.RESULTS_DIR, 'model_selection.json'), 'w') as f:
        json.dump(selection_report, f, indent=4)
    
    return best_model['model']

# Generate comprehensive report
def generate_final_report(results_dict, best_model_name):
    """Generate comprehensive final report"""
    print("\n" + "="*60)
    print("GENERATING FINAL REPORT")
    print("="*60)
    
    # Create summary DataFrame
    summary_data = []
    for model_name, results in results_dict.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': f"{results['accuracy']:.4f}",
            'Precision': f"{results['precision']:.4f}",
            'Recall': f"{results['recall']:.4f}",
            'F1-Score': f"{results['f1_score']:.4f}",
            'Top-5 Acc': f"{results['top5_accuracy']:.4f}",
            'Inf Time (s)': f"{results['avg_inference_time']:.6f}",
            'Model Size (MB)': f"{results['model_size_mb']:.1f}",
            'Params': f"{results['num_params']:,}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Mark best model
    df_summary['Best'] = ['✓' if row['Model'] == best_model_name else '' for _, row in df_summary.iterrows()]
    
    # Save to CSV
    csv_path = os.path.join(Config.RESULTS_DIR, 'model_comparison_summary.csv')
    df_summary.to_csv(csv_path, index=False)
    
    # Create markdown report
    md_report = f"""# Transfer Learning Model Comparison Report

## Executive Summary

This report compares four transfer learning models for image classification on 25 classes.
The best model selected is **{best_model_name}** based on a composite score considering accuracy, F1-score, inference time, and model size.

## Model Comparison Summary

{df_summary.to_markdown(index=False)}

## Key Findings

1. **Accuracy Performance**: 
   - Highest accuracy: {max([r['accuracy'] for r in results_dict.values()]):.4f}
   - Lowest accuracy: {min([r['accuracy'] for r in results_dict.values()]):.4f}
   - Range: {max([r['accuracy'] for r in results_dict.values()]) - min([r['accuracy'] for r in results_dict.values()]):.4f}

2. **Inference Speed**:
   - Fastest model: {min(results_dict.items(), key=lambda x: x[1]['avg_inference_time'])[0]} ({min([r['avg_inference_time'] for r in results_dict.values()]):.6f}s per image)
   - Slowest model: {max(results_dict.items(), key=lambda x: x[1]['avg_inference_time'])[0]} ({max([r['avg_inference_time'] for r in results_dict.values()]):.6f}s per image)

3. **Model Size**:
   - Smallest model: {min(results_dict.items(), key=lambda x: x[1]['model_size_mb'])[0]} ({min([r['model_size_mb'] for r in results_dict.values()]):.1f} MB)
   - Largest model: {max(results_dict.items(), key=lambda x: x[1]['model_size_mb'])[0]} ({max([r['model_size_mb'] for r in results_dict.values()]):.1f} MB)

## Recommendations

1. **For maximum accuracy**: Use {max(results_dict.items(), key=lambda x: x[1]['accuracy'])[0]}
2. **For deployment on mobile/edge devices**: Use {min(results_dict.items(), key=lambda x: x[1]['model_size_mb'])[0]}
3. **For real-time applications**: Use {min(results_dict.items(), key=lambda x: x[1]['avg_inference_time'])[0]}
4. **Best overall compromise**: **{best_model_name}**

## Next Steps

1. Fine-tune the selected model on more data
2. Implement ensemble methods combining top models
3. Optimize the selected model for production deployment
4. Test on additional datasets for generalization assessment

---

*Report generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    # Save markdown report
    md_path = os.path.join(Config.RESULTS_DIR, 'final_report.md')
    with open(md_path, 'w') as f:
        f.write(md_report)
    
    print(f"\nReports saved to:")
    print(f"1. CSV Summary: {csv_path}")
    print(f"2. Markdown Report: {md_path}")
    print(f"3. Visualizations: {Config.RESULTS_DIR}/")
    
    return df_summary

# Main comparison function
def compare_all_models():
    """Main function to compare all models"""
    print("="*60)
    print("TRANSFER LEARNING MODEL COMPARISON")
    print("="*60)
    
    # Load test data
    print("\nLoading test data...")
    test_data = load_test_data()
    Config.CLASS_NAMES = list(test_data.class_indices.keys())
    
    # Define models to compare
    model_names = ['VGG16', 'ResNet50', 'MobileNetV2', 'EfficientNetB0']
    
    # Evaluate each model
    results_dict = {}
    for model_name in model_names:
        model = load_model(model_name)
        if model:
            results = evaluate_model(model, test_data, model_name)
            results_dict[model_name] = results
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrices(results_dict, test_data)
    plot_metrics_comparison(results_dict)
    
    # Statistical analysis
    perform_statistical_analysis(results_dict, test_data)
    
    # Select best model
    best_model_name = select_best_model(results_dict)
    
    # Generate final report
    df_summary = generate_final_report(results_dict, best_model_name)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return results_dict, df_summary, best_model_name

if __name__ == "__main__":
    # Run comparison
    results_dict, df_summary, best_model = compare_all_models()
    
    # Display summary
    print("\nFinal Summary:")
    print(df_summary.to_string(index=False))
    print(f"\nBest Model Selected: {best_model}")