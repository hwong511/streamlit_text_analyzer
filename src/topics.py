from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

class TopicAnalyzer:
    def __init__(self):
        self.vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=10000
        )
        
        self.topic_model = BERTopic(
            vectorizer_model=self.vectorizer,
            nr_topics="auto",
            calculate_probabilities=True
        )
        
    def fit_topics(self, texts):
        topics, probs = self.topic_model.fit_transform(texts)
        return topics, probs
    
    def get_topic_info(self):
        return self.topic_model.get_topic_info()
    
    def get_representative_docs(self, topic_id, n=5):
        return self.topic_model.get_representative_docs(topic_id)
