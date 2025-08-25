import argparse
import os
from pathlib import Path
import sys

def main():
    """Main execution pipeline with LIAR dataset support and standalone inference"""
    parser = argparse.ArgumentParser(description='RAEmoLLAMA-Lite: Emotion-Aware Misinformation Detection')
    parser.add_argument('--mode', choices=['preprocess', 'emotion', 'vectorstore', 'inference', 'evaluate', 'full'], 
                       default='full', help='Execution mode')
    parser.add_argument('--model', default='tinyllama:latest', help='Ollama model name')
    parser.add_argument('--data', default=None, help='Input data file (optional - will auto-detect LIAR)')
    parser.add_argument('--use-liar', action='store_true', default=True, help='Try to use LIAR dataset')
    parser.add_argument('--sample-only', action='store_true', help='Use only sample dataset')
    
    args = parser.parse_args()
    
    print(" RAEmoLLAMA-Lite: LIAR Dataset Edition")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Use LIAR: {not args.sample_only}")
    
    
    Path("data").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    if args.mode in ['preprocess', 'full']:
        print("\n Step 1: Preprocessing data...")
        sys.path.append('.')
        try:
            from scripts.preprocess_data import preprocess_dataset
            
            use_liar = not args.sample_only
            df, dataset_type = preprocess_dataset(
                input_file=args.data, 
                use_liar=use_liar
            )
            print(f" Using {dataset_type.upper()} dataset with {len(df)} samples")
        except Exception as e:
            print(f" Error in preprocessing: {e}")
            return
    
    if args.mode in ['emotion', 'full']:
        print("\n Step 2: Emotion tagging...")
        try:
            from scripts.emotion_tagging import EmotionTagger
            tagger = EmotionTagger()
            tagger.tag_dataset("data/processed_misinfo.csv", "data/emotion_tagged_misinfo.csv")
        except Exception as e:
            print(f" Error in emotion tagging: {e}")
            return
    
    if args.mode in ['vectorstore', 'full']:
        print("\n Step 3: Building vector store...")
        try:
            from scripts.build_vectorstore import VectorStoreBuilder
            builder = VectorStoreBuilder()
            builder.build_vectorstore("data/emotion_tagged_misinfo.csv")
        except Exception as e:
            print(f" Error in vector store building: {e}")
            return
    
    if args.mode in ['inference', 'full']:
        print(f"\n Step 4: Running inference with {args.model}...")
        try:
            #ollama inference
            from scripts.ollama_inference import StandaloneOllamaInference
            inference = StandaloneOllamaInference(args.model)
            
            # Check if emotion tagged file exists
            data_file = "data/emotion_tagged_misinfo.csv"
            if not Path(data_file).exists():
                print(f" Required file not found: {data_file}")
                print("Please run preprocessing and emotion tagging first:")
                print("python main.py --mode preprocess")
                print("python main.py --mode emotion")
                return
            
            # Create safe filename for model
            model_safe = args.model.replace(':', '_').replace('/', '_')
            
            # Running both RAE and baseline inference
            print(" Running RAE inference...")
            rae_results = inference.run_rae_inference(
                data_file, 
                f"results/rae_results_{model_safe}.csv"
            )
            
            print("\n Running Baseline inference...")
            baseline_results = inference.run_baseline_inference(
                data_file, 
                f"results/baseline_results_{model_safe}.csv"
            )
            
            if rae_results is None or baseline_results is None:
                print(" Inference failed. Check error messages above.")
                return
            
        except Exception as e:
            print(f" Error in inference: {e}")
            return
    
    if args.mode in ['evaluate', 'full']:
        print("\n Step 5: Evaluating results...")
        try:
            
            from scripts.standalone_evaluation import StandaloneModelEvaluator
            evaluator = StandaloneModelEvaluator()
            model_suffix = args.model.replace(':', '_').replace('/', '_')
            
            rae_file = f"results/rae_results_{model_suffix}.csv"
            baseline_file = f"results/baseline_results_{model_suffix}.csv"
            
            # Check if result files exist
            if not Path(rae_file).exists() or not Path(baseline_file).exists():
                print(f" Result files not found:")
                print(f"  RAE: {rae_file} - {'' if Path(rae_file).exists() else ''}")
                print(f"  Baseline: {baseline_file} - {'' if Path(baseline_file).exists() else ''}")
                print("Please run inference first: python main.py --mode inference")
                return
            
            evaluator.compare_models(rae_file, baseline_file)
            
        except Exception as e:
            print(f" Error in evaluation: {e}")
            print(" Try running standalone evaluation:")
            print(f"   python standalone_evaluation.py {args.model.replace(':', '_')}")
            return
    
    print(f"\n RAEmoLLAMA-Lite pipeline completed successfully!")
    print(" Check the 'results/' directory for output files")
    print(" Use model comparison script for multi-model analysis")

def quick_inference():
    """Quick inference function for testing"""
    print(" Quick Inference Test")
    print("=" * 30)
    
    
    data_file = "data/emotion_tagged_misinfo.csv"
    if not Path(data_file).exists():
        print(f" Required file not found: {data_file}")
        print("Run full pipeline first: python main.py --mode full")
        return
    
    try:
        from scripts.ollama_inference import StandaloneOllamaInference
        
        #Importing TinyLLAMA innference from ollama inference standalone function
        inference = StandaloneOllamaInference()
        
        
        print("Running quick RAE inference...")
        rae_results = inference.run_inference(data_file)
        
        print("\nRunning quick baseline inference...")
        baseline_results = inference.run_inference(data_file)
        
        if rae_results is not None and baseline_results is not None:
            print("\n Quick Comparison:")
            rae_acc = (rae_results['rae_prediction'] == rae_results['true_label']).mean()
            baseline_acc = (baseline_results['baseline_prediction'] == baseline_results['true_label']).mean()
            
            print(f"RAE Accuracy: {rae_acc:.3f}")
            print(f"Baseline Accuracy: {baseline_acc:.3f}")
            print(f"RAE Improvement: {rae_acc - baseline_acc:+.3f}")
            
          
            from scripts.standalone_evaluation  import StandaloneModelEvaluator
            evaluator = StandaloneModelEvaluator()
            
            # Get model-safe filenames
            model_safe = "tinyllama: latest"
            rae_file = f"results/rae_results_{model_safe}.csv"
            baseline_file = f"results/baseline_results_{model_safe}.csv"
            
            if Path(rae_file).exists() and Path(baseline_file).exists():
                print("\n" + "="*50)
                evaluator.compare_models(rae_file, baseline_file)
            
    except Exception as e:
        print(f" Error in quick inference: {e}")

if __name__ == "__main__":
    # Check if running in quick mode
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_inference()
    else:
        main()