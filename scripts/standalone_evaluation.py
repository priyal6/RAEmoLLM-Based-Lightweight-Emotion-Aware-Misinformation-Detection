import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from pathlib import Path
import json

class StandaloneModelEvaluator:
    def __init__(self):
        """Initialize evaluator with LIAR dataset awareness"""
        pass
    
    def evaluate_predictions(self, results_df, pred_col, true_col='true_label'):
        """Evaluate model predictions with detailed metrics"""
        y_true = results_df[true_col]
        y_pred = results_df[pred_col]
        
       
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        
        if len(np.unique(y_true)) == 2:
            metrics.update({
                'precision_binary': precision_score(y_true, y_pred, zero_division=0),
                'recall_binary': recall_score(y_true, y_pred, zero_division=0),
                'f1_binary': f1_score(y_true, y_pred, zero_division=0)
            })
        
        
        try:
            report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'], zero_division=0)
        except:
            report = "Classification report unavailable"
        
       
        try:
            cm = confusion_matrix(y_true, y_pred)
        except:
            cm = np.array([[0, 0], [0, 0]])
        
        return metrics, report, cm
    
    def analyze_by_domain(self, results_df, pred_col):
        """Analyze performance by domain (political subject)"""
        if 'domain' not in results_df.columns:
            return {}
        
        domain_metrics = {}
        
        for domain in results_df['domain'].unique():
            domain_data = results_df[results_df['domain'] == domain]
            if len(domain_data) > 3:  
                try:
                    metrics, _, _ = self.evaluate_predictions(domain_data, pred_col)
                    domain_metrics[domain] = {
                        'count': len(domain_data),
                        'accuracy': metrics['accuracy'],
                        'f1': metrics['f1']
                    }
                except:
                    continue
        
        return domain_metrics
    
    def analyze_by_emotion(self, results_df, pred_col):
        """Analyze performance by dominant emotion"""
        if 'dominant_emotion' not in results_df.columns:
            return {}
        
        emotion_metrics = {}
        
        for emotion in results_df['dominant_emotion'].unique():
            emotion_data = results_df[results_df['dominant_emotion'] == emotion]
            if len(emotion_data) > 3:
                try:
                    metrics, _, _ = self.evaluate_predictions(emotion_data, pred_col)
                    emotion_metrics[emotion] = {
                        'count': len(emotion_data),
                        'accuracy': metrics['accuracy'],
                        'f1': metrics['f1']
                    }
                except:
                    continue
        
        return emotion_metrics
    
    def compare_models(self, rae_results_file, baseline_results_file):
        """Compare RAE vs baseline performance with comprehensive analysis"""
        
        
        if not Path(rae_results_file).exists():
            print(f" RAE results file not found: {rae_results_file}")
            return None, None, None
        
        if not Path(baseline_results_file).exists():
            print(f" Baseline results file not found: {baseline_results_file}")
            return None, None, None
        
        try:
            rae_df = pd.read_csv(rae_results_file)
            baseline_df = pd.read_csv(baseline_results_file)
        except Exception as e:
            print(f" Error loading result files: {e}")
            return None, None, None
        
       
        if len(rae_df) == 0 or len(baseline_df) == 0:
            print(" One or both result files are empty")
            return None, None, None
        
       
        rae_required = ['true_label', 'rae_prediction']
        baseline_required = ['true_label', 'baseline_prediction']
        
        if not all(col in rae_df.columns for col in rae_required):
            print(f" RAE file missing columns: {[col for col in rae_required if col not in rae_df.columns]}")
            return None, None, None
        
        if not all(col in baseline_df.columns for col in baseline_required):
            print(f" Baseline file missing columns: {[col for col in baseline_required if col not in baseline_df.columns]}")
            return None, None, None
        
        dataset_type = rae_df['dataset_type'].iloc[0] if 'dataset_type' in rae_df.columns else 'unknown'
        
        print(" MODEL COMPARISON RESULTS")
        print("=" * 60)
        print(f" Dataset: {dataset_type.upper()}")
        print(f" RAE Samples: {len(rae_df)}")
        print(f" Baseline Samples: {len(baseline_df)}")
        print("=" * 60)
        
       
        try:
            rae_metrics, rae_report, rae_cm = self.evaluate_predictions(rae_df, 'rae_prediction')
            
            print("\n RAE-INSPIRED MODEL:")
            print(f" Accuracy: {rae_metrics['accuracy']:.3f}")
            print(f" Precision: {rae_metrics['precision']:.3f}")
            print(f" Recall: {rae_metrics['recall']:.3f}")
            print(f" F1-Score: {rae_metrics['f1']:.3f}")
            
            if 'inference_time' in rae_df.columns:
                avg_time = rae_df['inference_time'].mean()
                print(f" Avg Inference Time: {avg_time:.2f}s")

            
            rae_pred_dist = rae_df['rae_prediction'].value_counts()
            print(f" Predictions - Real: {rae_pred_dist.get(0, 0)}, Fake: {rae_pred_dist.get(1, 0)}")
            
        except Exception as e:
            print(f" Error evaluating RAE results: {e}")
            return None, None, None
        
        
        try:
            baseline_metrics, baseline_report, baseline_cm = self.evaluate_predictions(baseline_df, 'baseline_prediction')
            
            print("\n BASELINE MODEL:")
            print(f" Accuracy: {baseline_metrics['accuracy']:.3f}")
            print(f" Precision: {baseline_metrics['precision']:.3f}")
            print(f" Recall: {baseline_metrics['recall']:.3f}")
            print(f" F1-Score: {baseline_metrics['f1']:.3f}")

            if 'inference_time' in baseline_df.columns:
                avg_time = baseline_df['inference_time'].mean()
                print(f"  Avg Inference Time: {avg_time:.2f}s")
            
            
            baseline_pred_dist = baseline_df['baseline_prediction'].value_counts()
            print(f" Predictions - Real: {baseline_pred_dist.get(0, 0)}, Fake: {baseline_pred_dist.get(1, 0)}")
            
        except Exception as e:
            print(f" Error evaluating baseline results: {e}")
            return None, None, None
        
        
        improvement = {
            'accuracy': rae_metrics['accuracy'] - baseline_metrics['accuracy'],
            'precision': rae_metrics['precision'] - baseline_metrics['precision'],
            'recall': rae_metrics['recall'] - baseline_metrics['recall'],
            'f1': rae_metrics['f1'] - baseline_metrics['f1']
        }
        
        print("\n IMPROVEMENT (RAE vs Baseline):")
        print("-" * 40)
        for metric, value in improvement.items():
            direction = "" if value > 0 else "" if value < 0 else ""
            print(f"{direction} {metric.capitalize()}: {value:+.3f}")
        
        
        if len(rae_df) > 10:
            print(f"\n Sample Size: {len(rae_df)} (sufficient for analysis)")
        else:
            print(f"\n  Sample Size: {len(rae_df)} (consider larger dataset)")
        
        
        if dataset_type == 'liar' or 'domain' in rae_df.columns:
            print("\n  POLITICAL DOMAIN ANALYSIS:")
            print("-" * 40)
            
            try:
                rae_domain_metrics = self.analyze_by_domain(rae_df, 'rae_prediction')
                baseline_domain_metrics = self.analyze_by_domain(baseline_df, 'baseline_prediction')
                
                common_domains = set(rae_domain_metrics.keys()) & set(baseline_domain_metrics.keys())
                
                if common_domains:
                    for domain in sorted(common_domains):
                        rae_f1 = rae_domain_metrics[domain]['f1']
                        baseline_f1 = baseline_domain_metrics[domain]['f1']
                        improvement_domain = rae_f1 - baseline_f1
                        count = rae_domain_metrics[domain]['count']
                        
                        print(f" {domain.capitalize():15} ({count:2d} samples): "
                              f"RAE: {rae_f1:.3f} | Baseline: {baseline_f1:.3f} | "
                              f"Δ: {improvement_domain:+.3f}")
                else:
                    print("   No common domains found for comparison")
                    
            except Exception as e:
                print(f"   Error in domain analysis: {e}")
        
        
        if 'dominant_emotion' in rae_df.columns:
            print("\n EMOTION-BASED ANALYSIS:")
            print("-" * 40)
            
            try:
                rae_emotion_metrics = self.analyze_by_emotion(rae_df, 'rae_prediction')
                baseline_emotion_metrics = self.analyze_by_emotion(baseline_df, 'baseline_prediction')
                
                common_emotions = set(rae_emotion_metrics.keys()) & set(baseline_emotion_metrics.keys())
                
                if common_emotions:
                    for emotion in sorted(common_emotions):
                        rae_f1 = rae_emotion_metrics[emotion]['f1']
                        baseline_f1 = baseline_emotion_metrics[emotion]['f1']
                        improvement_emotion = rae_f1 - baseline_f1
                        count = rae_emotion_metrics[emotion]['count']
                        
                        print(f" {emotion.capitalize():15} ({count:2d} samples): "
                              f"RAE: {rae_f1:.3f} | Baseline: {baseline_f1:.3f} | "
                              f"Δ: {improvement_emotion:+.3f}")
                else:
                    print("   No common emotions found for comparison")
                    
            except Exception as e:
                print(f"   Error in emotion analysis: {e}")
        
        
        try:
            comparison_results = {
                'rae_metrics': rae_metrics,
                'baseline_metrics': baseline_metrics,
                'improvement': improvement,
                'dataset_type': dataset_type,
                'sample_count': len(rae_df)
            }
            
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            with open(results_dir / "comparison_summary.json", 'w') as f:
                json.dump(comparison_results, f, indent=2)
            
            print(f"\n Comparison summary saved to results/comparison_summary.json")
            
        except Exception as e:
            print(f"Warning: Could not save comparison summary: {e}")
        
        return rae_metrics, baseline_metrics, improvement


def quick_evaluate(model_name="tinyllama: latest"):
    """Quick evaluation with automatic file detection"""
    print(f" Quick Evaluation for {model_name}")
    print("=" * 40)
    
   
    results_dir = Path("results")
    if not results_dir.exists():
        print(" Results directory not found. Run inference first.")
        return
    
    model_safe = model_name.replace(':', '_').replace('/', '_')
    rae_file = results_dir / f"rae_results_{model_safe}.csv"
    baseline_file = results_dir / f"baseline_results_{model_safe}.csv"
    
    
    if not rae_file.exists():
        print(f" RAE results not found: {rae_file}")
        # Listing available files
        print("Available result files:")
        for file in results_dir.glob("*.csv"):
            print(f"  - {file.name}")
        return
    
    if not baseline_file.exists():
        print(f" Baseline results not found: {baseline_file}")
        return
    
    
    evaluator = StandaloneModelEvaluator()
    evaluator.compare_models(str(rae_file), str(baseline_file))

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        quick_evaluate(model_name)
    else:
        quick_evaluate()
