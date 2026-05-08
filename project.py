import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(
    page_title="Career Compass",
    page_icon="🎯",
    layout="wide"
)


@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_model()



# ---------------------------------
# 1. Simulated dataset of career profiles
# Each career has title, required skills, education, interests, experience level, description
data = [
    {
        "job_title": "Data Scientist",
        "skills": "python, machine learning, statistics, data visualization",
        "education": "Bachelor",
        "interests": "data, analytics, research",
        "experience": "Mid",
        "description": "Analyzes data to extract insights and build predictive models."
    },
    {
        "job_title": "Software Engineer",
        "skills": "java, c++, algorithms, problem solving",
        "education": "Bachelor",
        "interests": "coding, development, problem solving",
        "experience": "Junior",
        "description": "Designs and develops software applications and systems."
    },
    {
        "job_title": "Graphic Designer",
        "skills": "photoshop, creativity, adobe illustrator, visual design",
        "education": "Associate",
        "interests": "art, creativity, media",
        "experience": "Entry",
        "description": "Creates visual concepts to communicate ideas."
    },
    {
        "job_title": "Project Manager",
        "skills": "leadership, communication, scheduling, budgeting",
        "education": "Bachelor",
        "interests": "management, organization, planning",
        "experience": "Senior",
        "description": "Oversees projects to ensure timely delivery within budget."
    },
    {
        "job_title": "Marketing Specialist",
        "skills": "seo, content creation, social media, communication",
        "education": "Bachelor",
        "interests": "marketing, branding, communication",
        "experience": "Mid",
        "description": "Develops strategies to promote products and brands."
    },
    {
        "job_title": "Cybersecurity Analyst",
        "skills": "network security, python, risk assessment, cryptography",
        "education": "Bachelor",
        "interests": "security, technology, risk management",
        "experience": "Mid",
        "description": "Protects an organization's computer systems and networks."
    },
    {
        "job_title": "Mechanical Engineer",
        "skills": "cad, thermodynamics, mechanics, problem solving",
        "education": "Bachelor",
        "interests": "engineering, mechanics, design",
        "experience": "Mid",
        "description": "Designs and tests mechanical devices and systems."
    },
    {
        "job_title": "Financial Analyst",
        "skills": "excel, finance, accounting, data analysis",
        "education": "Bachelor",
        "interests": "finance, economics, data",
        "experience": "Junior",
        "description": "Provides investment and financial recommendations."
    },
    {
        "job_title": "Teacher",
        "skills": "communication, patience, subject knowledge, mentoring",
        "education": "Bachelor",
        "interests": "teaching, education, helping others",
        "experience": "Mid",
        "description": "Educates and supports students in learning."
    },
    {
        "job_title": "UX Designer",
        "skills": "wireframing, user research, creativity, prototyping",
        "education": "Bachelor",
        "interests": "design, user experience, psychology",
        "experience": "Mid",
        "description": "Improves user satisfaction with products by enhancing usability."
    },
     {
        "job_title": "Content Creater",
        "skills": "video editing, social media, creativity, branding",
        "education": "Bachelor",
        "interests": "design, communication, media",
        "experience": "Junior",
        "description": "Creates engaging content while self promoting, creating a platform to build an audience."
    },

     {
        "job_title": "Business Analyst",
        "skills": "communication, problem solving, creativity, engaging",
        "education": "Bachelor",
        "interests": "strategy, business, analysis",
        "experience": "Mid",
        "description": "Analyzing business data and providing meaningful solutions to improve efficiency/value."
    },

     {
        "job_title": "Web Developer",
        "skills": "html, css, javascript, web developement",
        "education": "Bachelor",
        "interests": "coding, web design, development",
        "experience": "Mid",
        "description": "Builds and maintains web applications/websites."
    },

]

df = pd.DataFrame(data)

def get_career_embeddings():
    return embedding_model.encode(df['skills'].tolist())

career_embeddings = get_career_embeddings()
# ---------------------------------
# 2. Data preparation

# Combine all textual information into a single string feature
def combine_text_features(row):
    return f"{row['skills']} {row['education']} {row['interests']} {row['experience']}"

df['combined_features'] = df.apply(combine_text_features, axis=1)

# Vectorize the combined features using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['combined_features'])

# Encode job titles as labels for classification
le = LabelEncoder()
y = le.fit_transform(df['job_title'])

# ---------------------------------
# Model training
# Using a Random Forest Classifier to learn from combined textual features
model = MultinomialNB()
model.fit(X, y)

# ---------------------------------
# Streamlit UI
st.title("Career Compass")

st.markdown("""
Enter your profile information below to get the top 3 career path recommendations.
""")

# User inputs
skills_input = st.text_input("Enter your skills (comma separated):", "")
education_input = st.selectbox("Select your highest education level:",
                               ['Secondary School', 'Associate', 'Bachelor', 'Master', 'PhD'])
interests_input = st.text_input("Enter your interests (comma separated):", "")
experience_input = st.selectbox("Select your experience level:", ['Entry', 'Junior', 'Mid', 'Senior'])

# When user clicks the button, predict career recommendations
if st.button("Recommend Careers"):

    # Prepare user input: combine all fields same way as dataset
    user_features = f"{skills_input} {education_input} {interests_input} {experience_input}"
    user_vector = vectorizer.transform([user_features])

    # Predict probabilities for each class (career)
    pred_probs = model.predict_proba(user_vector)[0]

    # Add NLP keyword matching similarity to enhance recommendations
    # For each career, calculate cosine similarity between user skills and job skills
   
    user_embedding = embedding_model.encode(skills_input)


    similarities = cosine_similarity([user_embedding], career_embeddings)[0]

    # Combine model prediction probability and similarity score by weighted sum
    combined_scores = 0.7 * pred_probs + 0.3 * similarities

    # Get indices of top 3 recommended careers
    top3_idx = combined_scores.argsort()[::-1][:3]

    st.subheader("Top 3 Career Recommendations")
    for idx in top3_idx:
        with st.container():
            st.markdown(f"## {df.iloc[idx]['job_title']}")
            st.write(df.iloc[idx]['description'])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Skills:** {df.iloc[idx]['skills']}")
            st.markdown(f"**Education:** {df.iloc[idx]['education']}")
        with col2:
            st.markdown(f"**Interests:** {df.iloc[idx]['interests']}")
            st.markdown(f"**Experience:** {df.iloc[idx]['experience']}")

        st.markdown("---")
        if idx == top3_idx[0]:
            st.success("⭐ Best Match")

# ---------------------------------
# Explanation of Process in sidebar
st.sidebar.title("How it Works")
st.sidebar.info("""
This system uses a small sample dataset of career profiles with associated skills, education, interests, and experience.

- Text features from these categories are combined and vectorized.
- I used a Naive Bayes classification algorithm that predicts job categories based on the users input data.
- When given your profile, the model predicts career fit probabilities.
- Recommendations are enhanced using semantic mean-based matching  (cosine similarity) and Word Embeddings between your skills and job-required skills.
- The top 3 highest scored careers are displayed with descriptions for you to explore.
""")