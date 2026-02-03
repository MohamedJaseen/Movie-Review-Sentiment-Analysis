import streamlit as st
import pickle as pk
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Custom CSS Styling (Dark Theme & Contrast) ---
# Injecting CSS to maintain the dark theme and style components
st.markdown("""
<style>
/* ---------------------------------------------------- */
/* DARK THEME STYLES - HIGH CONTRAST */
/* ---------------------------------------------------- */

/* Main app styling: Deep charcoal background */
.stApp {
    background-color: #1e1e1e; 
    color: white; 
}

/* Style the title (H1) */
h1 {
    color: #00A34A; /* Vibrant Green Accent */
    text-align: center;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    padding-bottom: 10px;
    border-bottom: 2px solid #333333; 
}

/* Style all other headers (H2, H3) */
h2, h3 {
    color: #ffffff;
}

/* Style the text input area */
.stTextInput > div > div > input {
    background-color: #2b2b2b; 
    color: white; 
    border-radius: 8px;
    border: 1px solid #00A34A; 
    padding: 12px;
    font-size: 16px;
}

/* Style the button */
.stButton>button {
    background-color: #00A34A; 
    color: black; 
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.4); 
    transition: all 0.2s ease;
    width: 100%; 
}

/* Add Hover effect to the button */
.stButton>button:hover {
    background-color: #00873a;
    transform: scale(1.02);
}

/* Style the Prediction Result containers (Success/Error) */
div[data-testid="stSuccessContent"], div[data-testid="stErrorContent"] {
    font-size: 24px;
    font-weight: 700;
    text-align: center;
    padding: 30px;
    margin-top: 20px;
    border-radius: 12px;
}

/* Style for the custom review boxes */
.review-box {
    background-color: #2b2b2b;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #00A34A;
    margin-bottom: 15px;
}

/* Ensure any normal st.write text is light */
p, label {
    color: #cccccc !important; 
}

</style>
""", unsafe_allow_html=True)

# --- 2. Load Model, Vectorizer, and Sample Data ---
try:
    model = pk.load(open('model.pkl','rb'))
    scaler = pk.load(open('scaler.pkl','rb'))
except FileNotFoundError:
    st.error("Error: Model or Scaler files not found. Ensure 'model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# Sample data for the Review Lookup page
SAMPLE_REVIEWS = [
    ("This film was an absolute masterpiece! The acting was superb and the story was gripping from start to finish.", 1), # Positive
    ("The plot was painfully slow, the dialogue was nonsensical, and the climax never arrived. A complete failure.", 0), # Negative
    ("I enjoyed the first 30 minutes, but after that, I completely lost interest. Mediocre at best.", 0), # Negative
    ("Every frame was beautiful, a visual treat. The script, however, felt like it was written by an intern.", 1) # Subtle Positive/Sarcastic (Good test)
]

# --- 3. Sidebar and Page Routing ---

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Sentiment Analyzer", "Review Lookup", "Model Insights"])
st.sidebar.markdown("---")
st.sidebar.header("About the Model")
st.sidebar.info("This tool uses a **Logistic Regression model** trained on movie review data, utilizing **TF-IDF** for text feature extraction.")


# --- 4. Define Page Functions ---

def home_page():
    """Displays the main welcome and project overview page."""
    st.title("Welcome to the AI Sentiment Analyzer 🧠")
    st.markdown("---")
    
    st.header("Project Overview")
    st.write("""
        This application uses Machine Learning to instantly determine the sentiment (Positive or Negative) of a movie review.
        Use the sidebar to explore three main sections: **Sentiment Analyzer**, **Review Lookup**, and **Model Insights**.
    """)
    
    st.subheader("How It Works:")
    st.markdown("""
        * **TF-IDF:** Text is converted into numerical features using the trained TF-IDF Vectorizer.
        * **Logistic Regression:** The model classifies the review and provides a confidence score.
    """)

    st.markdown("---")
    st.info("👈 Use the sidebar to navigate to test the model!")


def analyzer_page():
    """Displays the main sentiment analysis tool with input and prediction."""
    st.title("🎬 Movie Review Sentiment Analyzer")
    st.markdown("---") 

    st.subheader("Enter Movie Review Text Below")

    # Use columns for input box and button alignment
    col1, col2 = st.columns([4, 1])

    with col1:
        review = st.text_input('Review:', label_visibility="collapsed", placeholder="Type your movie review here...")

    # Create a placeholder for the prediction result (initially empty)
    result_placeholder = st.empty()

    with col2:
        predict_button = st.button('Predict', disabled=not review)

    # --- Prediction Logic and Styled Output ---
    if predict_button and review.strip():
        # 1. Transform the input review
        review_scale = scaler.transform([review]).toarray()
        
        # 2. Get the probability and the class prediction
        probabilities = model.predict_proba(review_scale)[0]
        negative_proba = probabilities[0]
        positive_proba = probabilities[1]
        
        # Predict the final class
        prediction_class = model.predict(review_scale)[0]
        
        # 3. Display the result in a two-column layout
        st.markdown("---")
        
        col_status, col_chart = st.columns([1, 1])
        
        with col_status:
            st.subheader("Analysis Result:")
            
            if prediction_class == 0:
                status_text = f"**Negative Review**"
                st.error(f"{status_text} 👎 (Confidence: {negative_proba*100:.2f}%)")
                st.markdown("This review is likely expressing **dissatisfaction** with the film.")
            else:
                status_text = f"**Positive Review**"
                st.success(f"{status_text} 👍 (Confidence: {positive_proba*100:.2f}%)")
                st.markdown("This review is likely expressing **satisfaction** with the film.")

        with col_chart:
            st.subheader("Confidence Breakdown")
            
            # Create the Pie Chart using Plotly
            labels = ['Negative (0)', 'Positive (1)']
            values = [negative_proba, positive_proba]
            colors = ['#ff4d4d', '#00A34A'] # Red for Neg, Green for Pos

            fig = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values, 
                hole=.3, 
                marker_colors=colors,
                textinfo='percent+label',
                insidetextorientation='horizontal'
            )])
            
            fig.update_layout(
                margin=dict(t=0, b=0, l=0, r=0),
                showlegend=False,
                paper_bgcolor="#1e1e1e", # Dark background
                font=dict(color="white")
            )
            
            st.plotly_chart(fig, use_container_width=True)


def review_lookup_page():
    """Allows users to select a predefined review and see the result."""
    st.title("🔍 Review Lookup")
    st.markdown("---") 

    st.subheader("Test the Model with Curated Examples")
    st.write("Select a review from the dropdown list to see how the model classifies it instantly.")

    # Create a list of reviews for the dropdown
    review_options = [rev[0] for rev in SAMPLE_REVIEWS]
    selected_review_text = st.selectbox("Choose a sample review:", review_options)
    
    if selected_review_text:
        # 1. Transform the input review
        review_scale = scaler.transform([selected_review_text]).toarray()
        probabilities = model.predict_proba(review_scale)[0]
        negative_proba = probabilities[0]
        positive_proba = probabilities[1]
        prediction_class = model.predict(review_scale)[0]

        st.markdown('<div class="review-box">', unsafe_allow_html=True)
        st.write(f"**Selected Review:** *{selected_review_text}*")
        st.markdown("</div>", unsafe_allow_html=True)
        
        col_status, col_chart = st.columns([1, 1])

        with col_status:
            st.subheader("Prediction:")
            if prediction_class == 0:
                st.error(f"Negative (Confidence: {negative_proba*100:.2f}%) 👎")
            else:
                st.success(f"Positive (Confidence: {positive_proba*100:.2f}%) 👍")
                
        with col_chart:
            st.subheader("Breakdown:")
            # Use the same Plotly logic as the Analyzer page
            labels = ['Negative (0)', 'Positive (1)']
            values = [negative_proba, positive_proba]
            colors = ['#ff4d4d', '#00A34A']

            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, marker_colors=colors, textinfo='percent')])
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=False, paper_bgcolor="#1e1e1e", font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)


def model_insights_page():
    """Visualizes the top features (words) that influence the model's predictions."""
    st.title("📈 Model Insights")
    st.markdown("---") 
    
    st.subheader("Understanding the Model's Logic (Feature Importance)")
    st.write("This visualization shows the most important words that the model uses to classify a review as Positive or Negative.")

    # Get feature names (words) and coefficients (importance)
    feature_names = scaler.get_feature_names_out()
    coefficients = model.coef_[0]

    # Create a DataFrame for easy sorting
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    # Top Positive Features (Highest Coefficients)
    top_positive_features = feature_df.sort_values(by='Coefficient', ascending=False).head(50)
    
    # Top Negative Features (Lowest Coefficients / Most negative influence)
    top_negative_features = feature_df.sort_values(by='Coefficient', ascending=True).head(50)

    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("### Top Positive Words")
        st.write("Words most likely to lead to a **Positive (1)** prediction:")
        
        # Generate Word Cloud for Positive words
        pos_word_dict = dict(zip(top_positive_features['Feature'], top_positive_features['Coefficient']))
        wc_pos = WordCloud(width=400, height=400, background_color="#2b2b2b", colormap='Greens', random_state=42)
        wc_pos.generate_from_frequencies(pos_word_dict)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(wc_pos, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig) # Display the word cloud

    with col_neg:
        st.markdown("### Top Negative Words")
        st.write("Words most likely to lead to a **Negative (0)** prediction:")
        
        # Generate Word Cloud for Negative words
        # Use absolute value of coefficient for word cloud size, but keep negative context
        neg_word_dict = dict(zip(top_negative_features['Feature'], np.abs(top_negative_features['Coefficient'])))
        wc_neg = WordCloud(width=400, height=400, background_color="#2b2b2b", colormap='Reds', random_state=42)
        wc_neg.generate_from_frequencies(neg_word_dict)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(wc_neg, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig) # Display the word cloud


# --- 5. Main Page Execution ---
if page == "Home":
    home_page()
elif page == "Sentiment Analyzer":
    analyzer_page()
elif page == "Review Lookup":
    review_lookup_page()
elif page == "Model Insights":
    model_insights_page()