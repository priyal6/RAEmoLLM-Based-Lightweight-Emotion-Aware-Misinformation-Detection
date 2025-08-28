import pandas as pd
import numpy as np
from pathlib import Path
import re
import json
import requests
from urllib.parse import urlparse

def clean_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    
    # Removing URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Removing extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Removing special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text

def download_liar_dataset():
    """Download LIAR dataset if not available locally"""
    print(" Attempting to download LIAR dataset...")
    
    # LIAR dataset URL
    liar_urls = [
        "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv",
        ]
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    for url in liar_urls:
        try:
            print(f"Trying: {url}")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(data_dir / "liar_train.tsv", 'wb') as f:
                    f.write(response.content)
                print(" LIAR dataset downloaded successfully!")
                return True
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue
    
    print(" Could not download LIAR dataset automatically")
    print("Please download manually from: https://github.com/williamyang1991/LIAR")
    return False

def load_liar_dataset(file_path):
    """Load and parse LIAR dataset"""
    try:
        # LIAR dataset format: id, label, statement, subject, speaker, job, state, party, barely-true-counts, false-counts, half-true-counts, mostly-true-counts, pants-fire-counts, context
        df = pd.read_csv(file_path, sep='\t', header=None)
        
        # assigning column names based on LIAR format
        df.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 
                     'barely_true', 'false_count', 'half_true', 'mostly_true', 'pants_fire', 'context']
        
        print(f" Loaded LIAR dataset with {len(df)} samples")
        print(f" Label distribution: {df['label'].value_counts().to_dict()}")
        
        return df, True
        
    except Exception as e:
        print(f" Error loading LIAR dataset: {e}")
        return None, False

def convert_liar_to_binary(df):
    """Convert LIAR 6-class labels to binary classification"""
    # LIAR labels: pants-fire, false, barely-true, half-true, mostly-true, true
    # Map to: 0 (true/mostly-true/half-true) vs 1 (false/barely-true/pants-fire)
    
    label_mapping = {
        'true': 0,
        'mostly-true': 0,
        'half-true': 0,  # Could be debated, but often treated as "mostly accurate"
        'barely-true': 1,
        'false': 1,
        'pants-fire': 1
    }
    
    df['binary_label'] = df['label'].map(label_mapping)
    
    print(" Binary label conversion:")
    print(f"Real (0): {(df['binary_label'] == 0).sum()}")
    print(f"Fake (1): {(df['binary_label'] == 1).sum()}")
    
    return df

def create_sample_dataset():
    """Create sample dataset as fallback"""
    sample_data = {
        'statement': [
            "The unemployment rate has decreased by 2% this quarter according to official statistics",
            "The government is secretly planning to ban all private vehicles next month",
            "Climate change research shows increasing global temperatures over the past decade",
            "Vaccines contain microchips designed to track every citizen's location",
            "The federal budget allocates $50 billion to infrastructure development this year",
            "Politicians are hiding the fact that the earth is actually flat",
            "Recent studies indicate that renewable energy costs have dropped significantly",
            "The president personally controls gas prices through a secret button",
            "Educational funding has increased by 15% compared to last year",
            "All mainstream media is controlled by alien beings from outer space"
        ],
        'binary_label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'subject': ['economy', 'government', 'climate', 'health', 'economy', 'conspiracy', 
                   'energy', 'economy', 'education', 'conspiracy'],
        'speaker': ['economist', 'unknown', 'scientist', 'conspiracy theorist', 'politician',
                   'unknown', 'researcher', 'social media user', 'education official', 'unknown'],
        'context': ['Economic report', 'Social media post', 'Research paper', 'Blog post', 
                   'Government statement', 'Forum post', 'Academic study', 'Viral tweet', 
                   'Press release', 'Video claim']
    }
    
    return pd.DataFrame(sample_data)

def preprocess_dataset(input_file=None, output_file="data/processed_misinfo.csv", use_liar=True):
    """
    Preprocess misinformation dataset - LIAR or sample
    """
    print(" Starting dataset preprocessing...")
    
    # Trying to load LIAR dataset first
    df = None
    dataset_type = "sample"
    
    if use_liar:
        # Checking if LIAR file exists locally
        liar_files = ["data/liar_train.tsv", "data/train.tsv", input_file]
        
        for liar_file in liar_files:
            if liar_file and Path(liar_file).exists():
                df, success = load_liar_dataset(liar_file)
                if success:
                    dataset_type = "liar"
                    break
        
        # Trying to download if not found locally
        if df is None:
            if download_liar_dataset():
                df, success = load_liar_dataset("data/liar_train.tsv")
                if success:
                    dataset_type = "liar"
    
    # Fallback to sample dataset
    if df is None:
        print(" Using sample dataset as fallback...")
        df = create_sample_dataset()
        dataset_type = "sample"
    
    # Process based on dataset type
    if dataset_type == "liar":
        # Converting LIAR format
        df = convert_liar_to_binary(df)
        
        # Cleaning the statement text
        df['text'] = df['statement'].apply(clean_text)
        df['label'] = df['binary_label']
        df['domain'] = df['subject']
        
        # Selecting relevant columns
        df = df[['text', 'label', 'domain', 'speaker', 'context']].copy()
        
    else:
        # Sample dataset format
        df['text'] = df['statement'].apply(clean_text)
        df['label'] = df['binary_label']
        df['domain'] = df['subject']
    
    # Filtering out empty texts
    df = df[df['text'].str.len() > 20]  # Minimum length for meaningful analysis
    
    # Balancing dataset if heavily skewed
    label_counts = df['label'].value_counts()
    print(f" Label distribution: {label_counts.to_dict()}")
    
    # If very imbalanced, sample to balance
    if label_counts.min() / label_counts.max() < 0.3:  # Less than 30% minority class
        print("⚖️  Balancing dataset...")
        min_class_size = min(label_counts.min(), 1000)  # Cap at 1000 for efficiency
        df = df.groupby('label').apply(lambda x: x.sample(n=min(len(x), min_class_size))).reset_index(drop=True)
    
    # Saving processed data
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f" Processed {len(df)} samples saved to {output_file}")
    print(f" Dataset type: {dataset_type.upper()}")
    print(f" Final distribution: {df['label'].value_counts().to_dict()}")
    print(f" Domains: {df['domain'].unique()}")

    # Saving dataset info
    dataset_info = {
        'type': dataset_type,
        'samples': len(df),
        'label_distribution': df['label'].value_counts().to_dict(),
        'domains': df['domain'].unique().tolist(),
        'file': output_file
    }
    
    with open(output_file.replace('.csv', '_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    return df, dataset_type

if __name__ == "__main__":
    # Processing dataset with LIAR preference
    df, dataset_type = preprocess_dataset(use_liar=True)
    print(f"\n Dataset preprocessing completed using {dataset_type.upper()} dataset!")