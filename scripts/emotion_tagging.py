import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

class EmotionTagger:
    def __init__(self):
        """Initialize emotion classification pipeline"""
        print(" Loading emotion classification model...")
        try:
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,  # Fixed: use top_k instead of return_all_scores
                device=0 if torch.cuda.is_available() else -1
            )
            print(" Emotion model loaded successfully")
            self.model_loaded = True
        except Exception as e:
            print(f" Error loading emotion model: {e}")
            print(" Will use fallback emotion detection")
            self.emotion_classifier = None
            self.model_loaded = False
    
    def get_emotions(self, text):
        """Get emotion scores for text with robust error handling"""
        # Defaulting emotion scores
        default_emotions = {
            'anger': 0.1, 'disgust': 0.1, 'fear': 0.1, 
            'joy': 0.1, 'neutral': 0.5, 'sadness': 0.1, 'surprise': 0.1
        }
        
        try:
            if not self.model_loaded or self.emotion_classifier is None:
                return default_emotions

            # Cleaning and validating text
            if not isinstance(text, str):
                text = str(text)
            
            processed_text = text.strip()[:512]  
            
            if len(processed_text) < 3:  
                return default_emotions
            
            # Geting emotion predictions
            results = self.emotion_classifier(processed_text)
            
            # Debug: Print raw results to see what we're getting
            print(f"DEBUG - Raw results type: {type(results)}")
            print(f"DEBUG - Raw results: {results}")
            
            #different result formats
            emotion_scores = {}
            
            # Case 1: Results is a list of lists (batch format)
            if isinstance(results, list) and len(results) > 0:
                # If nested list, take first element
                if isinstance(results[0], list):
                    results = results[0]
                
               
                for result in results:
                    if isinstance(result, dict):
                        label = result.get('label', '')
                        score = result.get('score', 0.0)
                        if label and isinstance(score, (int, float)):
                            emotion_scores[label] = float(score)
                            print(f"DEBUG - Found emotion: {label} = {score:.3f}")
            
            # Case 2: Results is a single dict
            elif isinstance(results, dict):
                label = results.get('label', '')
                score = results.get('score', 0.0)
                if label and isinstance(score, (int, float)):
                    emotion_scores[label] = float(score)
                    print(f"DEBUG - Found single emotion: {label} = {score:.3f}")
            
            print(f"DEBUG - Total emotions extracted: {len(emotion_scores)}")
            print(f"DEBUG - Emotion keys: {list(emotion_scores.keys())}")
            
            # If we got valid emotions, we use them
            if emotion_scores and len(emotion_scores) > 0:
                print(f"DEBUG - Successfully extracted {len(emotion_scores)} emotions")

                # Ensuring all expected emotions are present
                final_emotions = default_emotions.copy()
                for emotion, score in emotion_scores.items():
                    # Mapping any emotion names that might be different
                    emotion_lower = emotion.lower()
                    if emotion_lower in final_emotions:
                        final_emotions[emotion_lower] = score
                    elif emotion in final_emotions:
                        final_emotions[emotion] = score
                
                return final_emotions
            else:
                print(f"DEBUG - No valid emotions extracted, using defaults")
                return default_emotions
                
        except Exception as e:
            print(f"Error processing text: {e}")
            print(f"DEBUG - Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return default_emotions
    
    def analyze_political_emotions(self, df):
        """Analyze emotion patterns in political statements"""
        print(" Analyzing political emotion patterns...")
        
        # Geting emotion scores for all texts
        emotion_data = []
        successful_analyses = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Emotion analysis"):
            try:
                emotions = self.get_emotions(row['text'])
                emotions['label'] = int(row['label'])
                emotions['domain'] = str(row.get('domain', 'unknown'))
                emotions['row_index'] = idx
                emotion_data.append(emotions)
                successful_analyses += 1
                
            except Exception as e:
                print(f"Failed to analyze emotions for row {idx}: {e}")
                # default emotions for failed rows
                default_emotions = {
                    'anger': 0.1, 'disgust': 0.1, 'fear': 0.1, 
                    'joy': 0.1, 'neutral': 0.5, 'sadness': 0.1, 'surprise': 0.1,
                    'label': int(row['label']),
                    'domain': str(row.get('domain', 'unknown')),
                    'row_index': idx
                }
                emotion_data.append(default_emotions)
        
        if not emotion_data:
            print(" No emotion data collected")
            return None
        
        emotion_df = pd.DataFrame(emotion_data)
        
        # patterns
        print(f"\n Emotion Analysis Results ({successful_analyses}/{len(df)} successful):")
        print("-" * 50)
        
        try:
            # Average emotions by label
            for label in [0, 1]:
                label_name = "REAL" if label == 0 else "FAKE"
                subset = emotion_df[emotion_df['label'] == label]
                
                if len(subset) > 0:
                    print(f"\n{label_name} statements ({len(subset)} samples):")
                    for emotion in ['anger', 'fear', 'joy', 'neutral']:
                        if emotion in subset.columns:
                            avg_score = subset[emotion].mean()
                            print(f"  {emotion.capitalize()}: {avg_score:.3f}")
                else:
                    print(f"\n{label_name} statements: No samples found")
        
        except Exception as e:
            print(f"Error in emotion analysis: {e}")
        
        return emotion_df
    
    def tag_dataset(self, input_file, output_file):
        """Add emotion tags to dataset with comprehensive error handling"""
        try:
            df = pd.read_csv(input_file)
            print(f" Processing {len(df)} samples for emotion tagging...")
        except Exception as e:
            print(f" Error loading dataset: {e}")
            return None
        
        # Loading dataset info to understand what we're working with
        info_file = input_file.replace('.csv', '_info.json')
        dataset_type = "unknown"
        try:
            if Path(info_file).exists():
                with open(info_file, 'r') as f:
                    dataset_info = json.load(f)
                    dataset_type = dataset_info.get('type', 'unknown')
        except:
            pass
        
        print(f" Processing {dataset_type.upper()} dataset for emotions")
        
        # Initializing emotion columns
        emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        for emotion in emotion_labels:
            df[f'emotion_{emotion}'] = 0.0
        
        # Processing each text with progress tracking
        successful_tags = 0
        failed_tags = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Emotion tagging"):
            try:
                # Geting emotions for this text
                emotions = self.get_emotions(row['text'])
                
                # Storing emotion scores safely
                for emotion in emotion_labels:
                    if emotion in emotions and isinstance(emotions[emotion], (int, float)):
                        df.at[idx, f'emotion_{emotion}'] = float(emotions[emotion])
                    else:
                        # Seting default value if emotion not found or invalid
                        default_val = 0.5 if emotion == 'neutral' else 0.1
                        df.at[idx, f'emotion_{emotion}'] = default_val
                
                successful_tags += 1
                
            except Exception as e:
                print(f"Failed to tag row {idx}: {e}")
                failed_tags += 1
                
                # Setting default values for failed rows
                for emotion in emotion_labels:
                    default_val = 0.5 if emotion == 'neutral' else 0.1
                    df.at[idx, f'emotion_{emotion}'] = default_val
        
        # Adding dominant emotion safely
        try:
            emotion_cols = [f'emotion_{emotion}' for emotion in emotion_labels]
            
            # Ensuring all emotion columns exist and have valid values
            for col in emotion_cols:
                if col not in df.columns:
                    df[col] = 0.1
                
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.1)
            
           
            df['dominant_emotion'] = df[emotion_cols].idxmax(axis=1).str.replace('emotion_', '')
            
            # Add emotion intensity (max emotion score)
            df['emotion_intensity'] = df[emotion_cols].max(axis=1)
            
        except Exception as e:
            print(f"Error calculating dominant emotions: {e}")
            df['dominant_emotion'] = 'neutral'
            df['emotion_intensity'] = 0.5
        
        # Analyzing emotion patterns if political dataset
        if dataset_type == "liar":
            try:
                self.analyze_political_emotions(df)
            except Exception as e:
                print(f"Error in political emotion analysis: {e}")
        
        # Saving results
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)
            
            print(f" Emotion-tagged dataset saved to {output_file}")
            print(f" Successfully tagged: {successful_tags}/{len(df)} samples")
            if failed_tags > 0:
                print(f"  Failed tags: {failed_tags} (using defaults)")
            
            # Show emotion distribution
            try:
                emotion_dist = df['dominant_emotion'].value_counts().to_dict()
                print(f" Dominant emotions: {emotion_dist}")
            except:
                print(" Dominant emotions: Analysis failed")
            
        except Exception as e:
            print(f" Error saving results: {e}")
            return None
        
        return df

# Test function to verify the fix
def test_emotion_tagger():
    """Test the emotion tagger with sample texts"""
    print(" Testing Emotion Tagger...")
    
    # Quick direct test first
    print("\n Direct model test:")
    try:
        from transformers import pipeline
        direct_classifier = pipeline("text-classification", 
                                   model="j-hartmann/emotion-english-distilroberta-base", 
                                   top_k=None)
        
        test_result = direct_classifier("I am so angry!")
        print(f"Direct result: {test_result}")
        print(f"Direct result type: {type(test_result)}")
        
    except Exception as e:
        print(f"Direct test failed: {e}")
    
    print("\n Testing through our class:")
    tagger = EmotionTagger()
    
    test_texts = [
        "I am so angry about this policy!",
        "This makes me very happy and joyful.",
        "I'm scared about the future.",
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {text}")
        print(f"{'='*60}")
        emotions = tagger.get_emotions(text)
        if emotions:
            dominant = max(emotions.items(), key=lambda x: x[1])
            print(f"Final result - Dominant emotion: {dominant[0]} ({dominant[1]:.3f})")
            print(f"All emotions: {emotions}")
        else:
            print("Failed to get emotions")

if __name__ == "__main__":
    
    test_emotion_tagger()
    
    
    print("\n" + "="*50)
    tagger = EmotionTagger()
    result = tagger.tag_dataset("data/processed_misinfo.csv", "data/emotion_tagged_misinfo.csv")
    
    if result is not None:
        print(" Emotion tagging completed successfully!")
    else:
        print(" Emotion tagging failed!")