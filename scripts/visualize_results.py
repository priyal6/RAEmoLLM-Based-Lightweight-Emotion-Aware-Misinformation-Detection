import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def load_results():
    """Load your specific result files"""
    rae_file = Path("results/rae_results_tinyllama_latest.csv")
    baseline_file = Path("results/baseline_results_tinyllama_latest.csv")
    
    # Checking if files exist
    if not rae_file.exists():
        print(f" RAE file not found: {rae_file}")
        return None, None
    
    if not baseline_file.exists():
        print(f" Baseline file not found: {baseline_file}")
        return None, None
    
    print(f" Loading: {rae_file.name} & {baseline_file.name}")
    
    try:
        rae_df = pd.read_csv(rae_file)
        baseline_df = pd.read_csv(baseline_file)
        
        print(f" Loaded: RAE({len(rae_df)} samples) | Baseline({len(baseline_df)} samples)")
        return rae_df, baseline_df
        
    except Exception as e:
        print(f" Error loading files: {e}")
        return None, None

def calculate_metrics(df, pred_col, true_col='true_label'):
    """Calculate basic metrics"""
    y_true = df[true_col]
    y_pred = df[pred_col]
    
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }

def plot_comparison(rae_df, baseline_df):
    """Create simple performance comparison"""
    # Calculating metrics
    rae_metrics = calculate_metrics(rae_df, 'rae_prediction')
    baseline_metrics = calculate_metrics(baseline_df, 'baseline_prediction')
    
    # Creating plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = list(rae_metrics.keys())
    rae_values = list(rae_metrics.values())
    baseline_values = list(baseline_metrics.values())
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Creating bars
    bars1 = ax.bar(x - width/2, rae_values, width, label='RAE-Inspired', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, baseline_values, width, label='Baseline', color='orange', alpha=0.7)
    
    # Adding value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Formatting
    ax.set_ylabel('Score')
    ax.set_title('RAE vs Baseline Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotations
    for i, (rae_val, base_val) in enumerate(zip(rae_values, baseline_values)):
        improvement = rae_val - base_val
        color = 'green' if improvement > 0 else 'red'
        ax.text(i, max(rae_val, base_val) + 0.05, f'{improvement:+.3f}', 
               ha='center', fontweight='bold', color=color)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(" Plot saved: results/performance_comparison.png")
    
    return rae_metrics, baseline_metrics

def print_summary(rae_metrics, baseline_metrics, rae_df, baseline_df):
    """Print text summary"""
    print("\n" + "="*50)
    print(" PERFORMANCE SUMMARY")
    print("="*50)
    
    print(f"\n RAE-INSPIRED MODEL:")
    for metric, value in rae_metrics.items():
        print(f"   {metric}: {value:.3f}")
    
    print(f"\n BASELINE MODEL:")
    for metric, value in baseline_metrics.items():
        print(f"   {metric}: {value:.3f}")
    
    print(f"\n IMPROVEMENTS:")
    for metric in rae_metrics:
        improvement = rae_metrics[metric] - baseline_metrics[metric]
        direction = " " if improvement > 0 else ""
        print(f"   {direction} {metric}: {improvement:+.3f}")
    
    # Dataset info
    print(f"\n DATASET INFO:")
    print(f"   Samples: {len(rae_df)}")
    if 'dataset_type' in rae_df.columns:
        dataset_type = rae_df['dataset_type'].iloc[0]
        print(f"   Type: {dataset_type.upper()}")
    
    # Labeling distribution
    true_dist = rae_df['true_label'].value_counts()
    print(f"   Real: {true_dist.get(0, 0)} | Fake: {true_dist.get(1, 0)}")

def simple_emotion_plot(rae_df, baseline_df):
    """Simple emotion analysis if available"""
    if 'dominant_emotion' not in rae_df.columns:
        print("  No emotion data available")
        return
    
    # Getting emotion distribution
    emotion_dist = rae_df['dominant_emotion'].value_counts()
    
    # Simple pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(emotion_dist.values, labels=emotion_dist.index, autopct='%1.1f%%')
    plt.title('Emotion Distribution in Dataset')
    plt.savefig('results/emotion_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(" Emotion plot saved: results/emotion_distribution.png")

def main():
    """Main function"""
    print(" Simple RAEmoLLAMA Visualizer")
    print("=" * 40)
    
    # Loading results
    rae_df, baseline_df = load_results()
    if rae_df is None:
        return
    
    # Creating comparison plot
    print("\n Creating performance comparison...")
    rae_metrics, baseline_metrics = plot_comparison(rae_df, baseline_df)
    
    # Print summary
    print_summary(rae_metrics, baseline_metrics, rae_df, baseline_df)
    
    # Simple emotion plot
    print("\n Creating emotion plot...")
    simple_emotion_plot(rae_df, baseline_df)
    
    print(f"\n Visualization complete!")
    print(" Check results/ for saved plots")

if __name__ == "__main__":
    main()