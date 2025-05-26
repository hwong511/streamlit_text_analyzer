from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.explainer = None
        
    def train(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
    def predict_with_explanation(self, text):
        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Extract top influential words
        feature_names = self.vectorizer.get_feature_names_out()
        if len(shap_values) == 2:  # Binary classification
            shap_vals = shap_values[1][0] if prediction == 1 else shap_values[0][0]
        else:
            shap_vals = shap_values[0]
            
        word_importance = sorted(
            zip(feature_names, shap_vals), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:10]
        return {
            'prediction': 'positive' if prediction == 1 else 'negative',
            'influential_words': word_importance
        }
