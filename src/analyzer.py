import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def main():
    st.set_page_config(page_title="Smart Text Analyzer", layout="wide")
    
    st.title("🔍 Smart Text Analyzer")
    st.markdown("Analyze sentiment, discover topics, and understand why!")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📤 Upload Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        text_column = st.selectbox("Select text column", options=[])
        
        # Model settings
        st.header("⚙️ Settings")
        domain = st.selectbox("Domain", ["General", "Reviews", "Social Media"])
        
    # Main area tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💭 Sentiment", "🎯 Topics", "🔬 Analyze Text"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Texts", "1,234")
        with col2:
            st.metric("Avg Sentiment", "65% Positive", "+5%")
        with col3:
            st.metric("Topics Found", "8")
            
    with tab2:
        # Sentiment distribution
        fig_sentiment = create_sentiment_plot(sentiment_data)
        st.plotly_chart(fig_sentiment)
        
        # Top influential words
        st.subheader("Most Influential Words")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Positive Indicators**")
            display_word_importance(positive_words)
        with col2:
            st.markdown("**Negative Indicators**")
            display_word_importance(negative_words)
            
    with tab3:
        # Interactive topic visualization
        st.subheader("Topic Distribution")
        topic_viz = topic_model.visualize_topics()
        st.plotly_chart(topic_viz)
        
        # Topic details
        selected_topic = st.selectbox("Select a topic to explore", topic_list)
        if selected_topic:
            st.write(f"### Topic {selected_topic}: {topic_labels[selected_topic]}")
            
            # Word cloud
            wordcloud = create_topic_wordcloud(topic_words[selected_topic])
            st.image(wordcloud)
            
            # Representative documents
            st.write("**Representative texts:**")
            for doc in representative_docs[selected_topic]:
                st.info(doc[:200] + "...")
                
    with tab4:
        user_text = st.text_area("Enter text to analyze:", height=200)
        
        if st.button("Analyze", type="primary"):
            with st.spinner("Analyzing..."):
                result = analyze_single_text(user_text)
                
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Sentiment", result['sentiment'], 
                         f"{result['confidence']:.1%} confident")
                st.metric("Primary Topic", result['topic'])
                
            with col2:
                st.write("**Why this prediction?**")
                fig_shap = create_shap_plot(result['shap_values'])
                st.pyplot(fig_shap)
