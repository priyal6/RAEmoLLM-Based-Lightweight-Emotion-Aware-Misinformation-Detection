import pandas as pd
import json
from pathlib import Path

class PromptConstructor:
    def __init__(self):
        """Initialize prompt templates for political misinformation detection"""
        self.templates = {
            "system_prompt_liar": """You are an expert fact-checker specializing in political misinformation detection. 
Your task is to analyze political statements and determine their truthfulness.

Consider these factors:
1. Factual accuracy based on verifiable evidence
2. Context and potential for misleading interpretation  
3. Emotional manipulation techniques in political rhetoric
4. Source credibility and potential bias
5. Historical accuracy of similar claims

Respond with only: "REAL" or "FAKE" followed by a brief explanation focusing on factual accuracy.""",
            
            "system_prompt_health": """You are an expert misinformation detector specializing in health-related content. 
Your task is to analyze text and determine if it contains misinformation.

Consider these factors:
1. Scientific accuracy based on medical evidence
2. Emotional manipulation techniques
3. Source credibility indicators
4. Logical consistency with established health facts

Respond with only: "REAL" or "FAKE" followed by a brief explanation.""",
            
            "rae_prompt_political": """Analyze the following political statement for truthfulness, considering both factual content and emotional context.

TARGET STATEMENT: "{target_text}"

SIMILAR EXAMPLES FROM POLITICAL FACT-CHECKING:
{similar_examples}

EMOTIONAL CONTEXT:
- Dominant emotion: {dominant_emotion}
- Anger level: {anger_score:.2f} (political rhetoric often uses anger)
- Fear level: {fear_score:.2f} (fear appeals common in politics)
- Joy level: {joy_score:.2f}
- Emotional intensity: {emotion_intensity:.2f}

DOMAIN: {domain} (political subject area)

Based on the similar examples and emotional analysis, classify the target statement as "REAL" or "FAKE". 
Political statements can be misleading through selective facts, emotional manipulation, or lack of context.""",
            
            "rae_prompt_general": """Analyze the following text for misinformation, considering both factual content and emotional context.

TARGET TEXT: "{target_text}"

SIMILAR EXAMPLES:
{similar_examples}

EMOTIONAL CONTEXT:
- Dominant emotion: {dominant_emotion}
- Anger level: {anger_score:.2f}
- Fear level: {fear_score:.2f}
- Joy level: {joy_score:.2f}

Based on the similar examples and emotional analysis, classify the target text as "REAL" or "FAKE" and provide reasoning."""
        }
    
    def detect_dataset_type(self, search_results):
        """Detect if we're working with LIAR dataset based on metadata"""
        if not search_results or not search_results.get('metadatas'):
            return "general"
        
        # Check metadata for LIAR-specific fields
        first_metadata = search_results['metadatas'][0][0] if search_results['metadatas'][0] else {}
        
        if 'speaker' in first_metadata or first_metadata.get('dataset_type') == 'liar':
            return "liar"
        
        return "general"
    
    def format_similar_examples(self, search_results, dataset_type="general"):
        """Format search results into examples based on dataset type"""
        if not search_results or not search_results.get('documents'):
            return "No similar examples found."
        
        examples = []
        
        for i, (doc, metadata) in enumerate(zip(search_results['documents'][0], search_results['metadatas'][0])):
            label = "REAL" if metadata['label'] == 0 else "FAKE"
            emotion = metadata['dominant_emotion']
            domain = metadata.get('domain', 'unknown')
            
            if dataset_type == "liar":
                speaker = metadata.get('speaker', 'unknown')
                example = f"Example {i+1}: \"{doc[:150]}...\""
                example += f"\nLabel: {label} | Speaker: {speaker} | Domain: {domain} | Emotion: {emotion}\n"
            else:
                example = f"Example {i+1}: \"{doc[:150]}...\""
                example += f"\nLabel: {label} | Domain: {domain} | Emotion: {emotion}\n"
            
            examples.append(example)
        
        return "\n".join(examples)
    
    def construct_rae_prompt(self, target_text, search_results, emotion_scores):
        """Construct RAE-inspired prompt based on dataset type"""
        # Detect dataset type
        dataset_type = self.detect_dataset_type(search_results)
        
        # Format examples
        similar_examples = self.format_similar_examples(search_results, dataset_type)
        
        # Get domain from search results
        domain = "general"
        if search_results and search_results.get('metadatas') and search_results['metadatas'][0]:
            domain = search_results['metadatas'][0][0].get('domain', 'general')
        
        # Choose appropriate template
        if dataset_type == "liar":
            template = self.templates["rae_prompt_political"]
        else:
            template = self.templates["rae_prompt_general"]
        
        # Construct prompt
        prompt = template.format(
            target_text=target_text,
            similar_examples=similar_examples,
            dominant_emotion=emotion_scores.get('dominant_emotion', 'neutral'),
            anger_score=emotion_scores.get('emotion_anger', 0.0),
            fear_score=emotion_scores.get('emotion_fear', 0.0),
            joy_score=emotion_scores.get('emotion_joy', 0.0),
            emotion_intensity=emotion_scores.get('emotion_intensity', 0.0),
            domain=domain
        )
        
        return prompt
    
    def construct_baseline_prompt(self, target_text, dataset_type="general"):
        """Construct baseline prompt without RAE"""
        if dataset_type == "liar":
            system_prompt = self.templates["system_prompt_liar"]
        else:
            system_prompt = self.templates["system_prompt_health"]
        
        return f"""{system_prompt}

Analyze this statement:

"{target_text}"

Classify as "REAL" or "FAKE" and provide reasoning."""
