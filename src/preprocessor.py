import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
    def clean_text(self, text):
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.lower().strip()
    
    def process_batch(self, texts):
        cleaned = [self.clean_text(t) for t in texts]
        processed = []
        for doc in self.nlp.pipe(cleaned, batch_size=50):
            tokens = [token.lemma_ for token in doc 
                     if not token.is_stop and token.is_alpha]
            processed.append(' '.join(tokens))
        return processed

