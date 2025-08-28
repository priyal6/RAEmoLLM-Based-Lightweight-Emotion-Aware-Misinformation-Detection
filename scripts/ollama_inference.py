import ollama
import pandas as pd
import json
import re
from pathlib import Path
from tqdm import tqdm
import time
import chromadb
from sentence_transformers import SentenceTransformer

class StandaloneOllamaInference:
    def __init__(self, model_name="tinyllama:latest"):
        """Initialize Ollama client with all components built-in"""
        self.model_name = model_name
        self.client = ollama.Client()
        
        # Initializing vector store
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # TinyLLAMA specific settings
        self.model_settings = self._get_model_settings(model_name)
        
        print(f" Ollama client initialized with model: {model_name}")
        print(f"  Model settings: {self.model_settings}")
    
    def _get_model_settings(self, model_name):
        """Get optimized settings for different models"""
        if "tinyllama" in model_name.lower():
            return {
                'temperature': 0.1,
                'top_p': 0.9,
                'num_predict': 150,
                'top_k': 2,
                'repeat_penalty': 1.1
            }
        elif "llama2" in model_name.lower():
            return {
                'temperature': 0.1,
                'top_p': 0.9,
                'num_predict': 200,
                'top_k': 4,
                'repeat_penalty': 1.1
            }
        else:
            return {
                'temperature': 0.1,
                'top_p': 0.9,
                'num_predict': 180,
                'top_k': 3,
                'repeat_penalty': 1.1
            }
    
    def search_similar(self, query_text, top_k=3):
        """Search for similar documents in vector store"""
        try:
            #available collections
            collections = self.chroma_client.list_collections()
            if not collections:
                print(" No vector store collections found")
                return None
            
            #irst available collection
            collection_name = collections[0].name
            collection = self.chroma_client.get_collection(collection_name)
            
            # Search
            results = collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            return results
            
        except Exception as e:
            print(f"Vector search error: {e}")
            return None
    
    def construct_rae_prompt(self, target_text, search_results, emotion_scores):
        """Construct RAE-inspired prompt"""
       
        similar_examples = "No similar examples found."
        if search_results and search_results.get('documents') and search_results['documents'][0]:
            examples = []
            for i, (doc, metadata) in enumerate(zip(search_results['documents'][0], search_results['metadatas'][0])):
                label = "REAL" if metadata.get('label') == 0 else "FAKE"
                emotion = metadata.get('dominant_emotion', 'neutral')
                domain = metadata.get('domain', 'unknown')
                
                example = f"Example {i+1}: \"{doc[:150]}...\""
                example += f"\nLabel: {label} | Domain: {domain} | Emotion: {emotion}\n"
                examples.append(example)
            
            similar_examples = "\n".join(examples)
        
        # Get emotion values
        dominant_emotion = emotion_scores.get('dominant_emotion', 'neutral')
        anger_score = emotion_scores.get('emotion_anger', 0.0)
        fear_score = emotion_scores.get('emotion_fear', 0.0)
        joy_score = emotion_scores.get('emotion_joy', 0.0)
        emotion_intensity = emotion_scores.get('emotion_intensity', 0.0)
        
        prompt = f"""Analyze the following statement for truthfulness, considering both factual content and emotional context.

TARGET STATEMENT: "{target_text}"

SIMILAR EXAMPLES FROM FACT-CHECKING:
{similar_examples}

EMOTIONAL CONTEXT:
- Dominant emotion: {dominant_emotion}
- Anger level: {anger_score:.2f}
- Fear level: {fear_score:.2f}
- Joy level: {joy_score:.2f}
- Emotional intensity: {emotion_intensity:.2f}

Based on the similar examples and emotional analysis, classify the target statement as "REAL" or "FAKE" and provide brief reasoning focusing on factual accuracy."""
        
        return prompt
    
    def construct_baseline_prompt(self, target_text):
        """Construct baseline prompt without RAE"""
        return f"""You are an expert fact-checker. Analyze this statement for truthfulness:

"{target_text}"

Consider:
1. Factual accuracy based on verifiable evidence
2. Context and potential for misleading interpretation
3. Source credibility indicators
4. Logical consistency

Classify as "REAL" or "FAKE" and provide brief reasoning."""
    
    def query_ollama(self, prompt, max_retries=3):
        """Query Ollama with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=[{
                        'role': 'user',
                        'content': prompt
                    }],
                    options=self.model_settings
                )
                return response['message']['content']
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return f"ERROR: Failed to get response after {max_retries} attempts"
    
    def extract_prediction(self, response):
        """Extract REAL/FAKE prediction from response"""
        if "ERROR:" in response:
            return 0  # Default to REAL if error
        
        response_upper = response.upper()
        
        # Looking for explicit FAKE/REAL
        if "FAKE" in response_upper:
            return 1
        elif "REAL" in response_upper:
            return 0
        
        # Fallback indicators
        fake_indicators = ["MISINFORMATION", "FALSE", "INCORRECT", "MISLEADING", "UNTRUE", "PANTS-FIRE"]
        if any(word in response_upper for word in fake_indicators):
            return 1
        else:
            return 0
    
    def detect_dataset_type(self, test_data_file):
        """Detect dataset type from file"""
        info_file = test_data_file.replace('.csv', '_info.json')
        if Path(info_file).exists():
            try:
                with open(info_file, 'r') as f:
                    dataset_info = json.load(f)
                    return dataset_info.get('type', 'general')
            except:
                pass
        return 'general'
    
    def run_rae_inference(self, test_data_file="data/emotion_tagged_misinfo.csv", output_file=None):
        """Run RAE-inspired inference"""
        if output_file is None:
            model_safe = self.model_name.replace(':', '_').replace('/', '_')
            output_file = f"results/rae_results_{model_safe}.csv"
        
        # Checking if file exists
        if not Path(test_data_file).exists():
            print(f" File not found: {test_data_file}")
            return None
        
        df = pd.read_csv(test_data_file)
        dataset_type = self.detect_dataset_type(test_data_file)
        
        print(f" Running RAE inference on {len(df)} samples from {dataset_type.upper()} dataset...")
        print(f" Using model: {self.model_name}")
        
        # Verifying required columns
        required_cols = ['text', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f" Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="RAE Inference"):
            try:
                # Geting emotion scores
                emotion_scores = {}
                emotion_cols = [col for col in df.columns if col.startswith('emotion_')]
                
                if emotion_cols:
                    # Use pre-computed emotions
                    for col in emotion_cols:
                        emotion_scores[col] = float(row[col]) if pd.notna(row[col]) else 0.1
                    
                    emotion_scores['dominant_emotion'] = str(row.get('dominant_emotion', 'neutral'))
                    emotion_scores['emotion_intensity'] = float(row.get('emotion_intensity', 0.5)) if pd.notna(row.get('emotion_intensity')) else 0.5
                else:
                    # Defaulting emotions if not available
                    emotion_scores = {
                        'emotion_anger': 0.1, 'emotion_fear': 0.1, 'emotion_joy': 0.1,
                        'dominant_emotion': 'neutral', 'emotion_intensity': 0.5
                    }
                
                # Searching for similar examples
                search_results = self.search_similar(row['text'], top_k=3)
                
                # Constructing RAE prompt
                prompt = self.construct_rae_prompt(row['text'], search_results, emotion_scores)
                
                # Getting prediction
                start_time = time.time()
                response = self.query_ollama(prompt)
                inference_time = time.time() - start_time
                
                prediction = self.extract_prediction(response)
                
                results.append({
                    'text': row['text'],
                    'true_label': int(row['label']),
                    'rae_prediction': prediction,
                    'rae_response': response,
                    'dominant_emotion': emotion_scores.get('dominant_emotion', 'neutral'),
                    'emotion_intensity': emotion_scores.get('emotion_intensity', 0.0),
                    'inference_time': inference_time,
                    'dataset_type': dataset_type,
                    'domain': str(row.get('domain', 'unknown'))
                })
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                results.append({
                    'text': str(row['text']),
                    'true_label': int(row['label']),
                    'rae_prediction': 0,
                    'rae_response': f"ERROR: {str(e)}",
                    'dominant_emotion': 'neutral',
                    'emotion_intensity': 0.0,
                    'inference_time': 0.0,
                    'dataset_type': dataset_type,
                    'domain': 'unknown'
                })
        
        # Save results
        results_df = pd.DataFrame(results)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_file, index=False)
        
        print(f" RAE inference results saved to {output_file}")
        
        # Quick performance summary
        correct_predictions = (results_df['rae_prediction'] == results_df['true_label']).sum()
        total_predictions = len(results_df)
        accuracy = correct_predictions / total_predictions
        avg_time = results_df['inference_time'].mean()
        
        print(f" Quick Results:")
        print(f"   Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
        print(f"   Avg Time: {avg_time:.2f}s per sample")
        
        return results_df
    
    def run_baseline_inference(self, test_data_file="data/emotion_tagged_misinfo.csv", output_file=None):
        """Run baseline inference without RAE"""
        if output_file is None:
            model_safe = self.model_name.replace(':', '_').replace('/', '_')
            output_file = f"results/baseline_results_{model_safe}.csv"
        
       
        if not Path(test_data_file).exists():
            print(f" File not found: {test_data_file}")
            return None
        
        df = pd.read_csv(test_data_file)
        dataset_type = self.detect_dataset_type(test_data_file)
        
        print(f" Running baseline inference on {len(df)} samples from {dataset_type.upper()} dataset...")
        print(f" Using model: {self.model_name}")

        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Baseline Inference"):
            try:
                # Constructing baseline prompt
                prompt = self.construct_baseline_prompt(row['text'])
                
                #prediction
                start_time = time.time()
                response = self.query_ollama(prompt)
                inference_time = time.time() - start_time
                
                prediction = self.extract_prediction(response)
                
                results.append({
                    'text': row['text'],
                    'true_label': int(row['label']),
                    'baseline_prediction': prediction,
                    'baseline_response': response,
                    'inference_time': inference_time,
                    'dataset_type': dataset_type,
                    'domain': str(row.get('domain', 'unknown'))
                })
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                results.append({
                    'text': str(row['text']),
                    'true_label': int(row['label']),
                    'baseline_prediction': 0,
                    'baseline_response': f"ERROR: {str(e)}",
                    'inference_time': 0.0,
                    'dataset_type': dataset_type,
                    'domain': 'unknown'
                })
        
        #results
        results_df = pd.DataFrame(results)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_file, index=False)
        
        print(f" Baseline inference results saved to {output_file}")
        
        # Quick performance summary
        correct_predictions = (results_df['baseline_prediction'] == results_df['true_label']).sum()
        total_predictions = len(results_df)
        accuracy = correct_predictions / total_predictions
        avg_time = results_df['inference_time'].mean()
        
        print(f" Quick Results:")
        print(f"   Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
        print(f"   Avg Time: {avg_time:.2f}s per sample")
        
        return results_df

# Quick test and run function
def main():
    """Main function to run inference"""
    print(" Starting Ollama Inference...")
    
    # Initializing with TinyLLAMA
    inference = StandaloneOllamaInference("tinyllama:latest")
    
    # Checking if data file exists
    data_file = "data/emotion_tagged_misinfo.csv"
    if not Path(data_file).exists():
        print(f" Data file not found: {data_file}")
        print("Available files in data/:")
        data_dir = Path("data")
        if data_dir.exists():
            for file in data_dir.glob("*.csv"):
                print(f"  - {file}")
        return
    
    # Running both RAE and baseline inference
    print("\n" + "="*50)
    print(" Running RAE Inference...")
    rae_results = inference.run_rae_inference(data_file)
    
    print("\n" + "="*50)
    print(" Running Baseline Inference...")
    baseline_results = inference.run_baseline_inference(data_file)
    
    if rae_results is not None and baseline_results is not None:
        print("\n Both inferences completed successfully!")
        print(" Check the 'results/' directory for output files")
    else:
        print("\n Some inferences failed. Check error messages above.")

if __name__ == "__main__":
    main()