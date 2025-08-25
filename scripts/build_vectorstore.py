import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from pathlib import Path
import json

class VectorStoreBuilder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize embedding model and vector store"""
        print(f" Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        print(" Vector store initialized")
    
    def build_vectorstore(self, input_file, collection_name="misinfo_detection"):
        """Build vector store from processed dataset"""
        df = pd.read_csv(input_file)
        
        # Load dataset info
        info_file = input_file.replace('.csv', '_info.json')
        dataset_type = "unknown"
        if Path(info_file).exists():
            with open(info_file, 'r') as f:
                dataset_info = json.load(f)
                dataset_type = dataset_info.get('type', 'unknown')
        
        print(f" Building vector store with {len(df)} samples from {dataset_type.upper()} dataset...")
        
        # Update collection name based on dataset
        collection_name = f"{dataset_type}_{collection_name}"
        
        # Create or get collection
        try:
            collection = self.chroma_client.get_collection(collection_name)
            self.chroma_client.delete_collection(collection_name)
        except:
            pass
        
        collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={
                "description": f"{dataset_type.upper()} misinformation detection dataset",
                "dataset_type": dataset_type
            }
        )
        
        # Generate embeddings with batching for efficiency
        texts = df['text'].tolist()
        batch_size = 32
        all_embeddings = []
        
        print(" Generating embeddings...")
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings)
        
        # Prepare metadata with dataset-specific fields
        metadatas = []
        for idx, row in df.iterrows():
            metadata = {
                'label': int(row['label']),
                'domain': str(row['domain']),
                'dominant_emotion': str(row['dominant_emotion']),
                'emotion_intensity': float(row['emotion_intensity']),
                'dataset_type': dataset_type
            }
            
            # Add dataset-specific metadata
            if 'speaker' in df.columns:
                metadata['speaker'] = str(row['speaker'])
            if 'context' in df.columns:
                metadata['context'] = str(row['context'])
            
            # Add emotion scores
            emotion_cols = [col for col in df.columns if col.startswith('emotion_')]
            for col in emotion_cols:
                metadata[col] = float(row[col])
            
            metadatas.append(metadata)
        
        # Add to collection in batches
        batch_size = 100
        for i in tqdm(range(0, len(texts), batch_size), desc="Adding to vector store"):
            end_idx = min(i + batch_size, len(texts))
            
            collection.add(
                embeddings=all_embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=[f"doc_{j}" for j in range(i, end_idx)]
            )
        
        print(f" Vector store built with {len(texts)} documents")
        print(f" Collection: {collection_name}")
        
        # Test retrieval
        self._test_retrieval(collection, dataset_type)
        
        return collection
    
    def _test_retrieval(self, collection, dataset_type):
        """Test the vector store retrieval"""
        if dataset_type == "liar":
            test_queries = [
                "Government spending on healthcare",
                "Climate change policies",
                "Economic statistics and unemployment"
            ]
        else:
            test_queries = [
                "Vaccine safety and effectiveness",
                "Government conspiracy theories",
                "Scientific research findings"
            ]
        
        print(f"\n Testing retrieval with {dataset_type} queries:")
        for query in test_queries[:2]:  # Test first 2
            results = collection.query(
                query_texts=[query],
                n_results=2,
                include=['documents', 'metadatas']
            )
            
            print(f"\nQuery: '{query}'")
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                label = "REAL" if meta['label'] == 0 else "FAKE"
                emotion = meta['dominant_emotion']
                print(f"  {i+1}. {doc[:80]}... [{label}, {emotion}]")
    
    def search_similar(self, query_text, collection_name="misinfo_detection", top_k=5):
        """Search for similar documents"""
        # Try to determine collection name
        collections = self.chroma_client.list_collections()
        available_collections = [c.name for c in collections]
        
        # Find the right collection
        target_collection = None
        for coll_name in available_collections:
            if collection_name in coll_name:
                target_collection = coll_name
                break
        
        if not target_collection:
            # Use first available collection
            target_collection = available_collections[0] if available_collections else collection_name
        
        try:
            collection = self.chroma_client.get_collection(target_collection)
        except:
            print(f" Collection {target_collection} not found")
            return None
        
        # Search
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        return results

if __name__ == "__main__":
    builder = VectorStoreBuilder()
    builder.build_vectorstore("data/emotion_tagged_misinfo.csv")
