import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# Set page config
st.set_page_config(
    page_title="FuturePath - Smart Career Prediction System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling for modern aesthetics
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }

    /* Gradient Background for Header Banner */
    .header-container {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #C084FC 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(124, 58, 237, 0.2);
    }
    
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Section Cards */
    .section-card {
        background-color: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Dark mode adjustments for cards if background is light */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(26, 20, 48, 1) 0%, rgba(15, 12, 30, 1) 90.2%);
        color: #E2E8F0;
    }
    
    /* Dropdowns and Inputs styling */
    div[data-baseweb="select"] > div {
        background-color: #1F2937 !important;
        color: white !important;
        border: 1px solid #4B5563 !important;
        border-radius: 8px !important;
    }
    
    input, [data-baseweb="slider"] {
        border-color: #4B5563 !important;
    }

    /* Prediction Card */
    .prediction-container {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.15) 100%);
        border: 2px solid #10B981;
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.15);
    }
    
    .prediction-role {
        font-size: 2.5rem;
        font-weight: 800;
        color: #10B981;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    
    .metric-badge {
        background-color: #2D3748;
        border: 1px solid #4A5568;
        border-radius: 50px;
        padding: 0.3rem 1rem;
        display: inline-block;
        font-weight: 600;
        color: #A0AEC0;
        margin-bottom: 1rem;
    }
    
    .top-recommendations {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    
    /* Sidebar */
    .sidebar-info {
        padding: 1rem;
        background-color: #1E1B4B;
        border-radius: 12px;
        border: 1px solid #312E81;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check if model file exists
@st.cache_resource
def load_model():
    model_file = 'career_model_pipeline.pkl'
    if not os.path.exists(model_file):
        return None
    return joblib.load(model_file)

# Load the model
model_data = load_model()

# Title banner
st.markdown("""
    <div class="header-container">
        <div class="header-title">🔮 FuturePath AI</div>
        <div class="header-subtitle">Advanced Machine Learning Career Recommendation Engine</div>
    </div>
""", unsafe_allow_html=True)

if model_data is None:
    st.error("⚠️ Model file `career_model_pipeline.pkl` not found! Please run `train_model.py` first to train and save the model.")
    st.info("ℹ️ To train the model, open your terminal and run:\n```bash\npython train_model.py\n```")
    st.stop()

# Extract models and metadata
pipeline = model_data['pipeline']
metadata = model_data['metadata']
target_classes = model_data['target_classes']

# Sidebar Info
with st.sidebar:
    st.title("About FuturePath")
    st.markdown("""
    FuturePath analyzes your cognitive abilities, technical skills, interests, and personality traits to predict the most suitable IT career path for you.
    
    Powered by a **Random Forest Classifier** trained on 6,900+ professional profiles.
    """)
    
    st.markdown("""
    <div class="sidebar-info">
        <h4 style="margin-top:0; color:#818CF8;">Engine Stats</h4>
        <p style="margin-bottom:6px;">📊 <b>Dataset Size:</b> 6,901 rows</p>
        <p style="margin-bottom:6px;">🧠 <b>Classifier:</b> Random Forest</p>
        <p style="margin-bottom:6px;">🎯 <b>Predictive Roles:</b> 12 Classes</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("---")
    st.markdown("Created by Antigravity AI 🚀")

# Form setup
st.markdown("### 📋 Fill in your profile details below:")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧠 Cognitive & Technical Skills")
        
        coding_rating = st.slider(
            "💻 Coding Skills Rating (1-9)",
            min_value=1, max_value=9, value=5,
            help="Rate your programming proficiency level."
        )
        
        logical_rating = st.slider(
            "🧩 Logical Quotient Rating (1-9)",
            min_value=1, max_value=9, value=5,
            help="Rate your logical reasoning and problem-solving capability."
        )
        
        hackathons = st.slider(
            "🏆 Hackathons Participated (0-6)",
            min_value=0, max_value=6, value=1,
            help="Number of hackathons or coding competitions you have joined."
        )
        
        sub = st.selectbox(
            "📚 Interested Subjects",
            options=metadata['Interested subjects'],
            help="Which academic subject interest area aligns most with you?"
        )
        
        cert = st.selectbox(
            "🎖️ Primary Certification",
            options=metadata['certifications'],
            help="Which area are you certified in or aiming to get certified in?"
        )
        
        workshop = st.selectbox(
            "🛠️ Main Workshop Attended",
            options=metadata['workshops'],
            help="Select the key technical workshop topic you have attended."
        )
        
        extra_courses = st.selectbox(
            "🎓 Completed Extra Courses?",
            options=metadata['Extra-courses did'],
            help="Have you completed additional professional courses outside university?"
        )
        
    with col2:
        st.subheader("🗣️ Communication & Personal Traits")
        
        public_speaking = st.slider(
            "🎙️ Public Speaking Points (1-9)",
            min_value=1, max_value=9, value=5,
            help="Assess your speaking, presentation, and communication confidence."
        )
        
        read_write = st.selectbox(
            "📝 Reading & Writing Skills",
            options=metadata['reading and writing skills'],
            help="Rate your textual communication and documentation skills."
        )
        
        memory = st.selectbox(
            "🧠 Memory Capability Score",
            options=metadata['memory capability score'],
            help="Rate your memory retention and recall ability."
        )
        
        self_learning = st.selectbox(
            "⚡ Self-learning Capability?",
            options=metadata['self-learning capability?'],
            help="Do you easily learn new things independently?"
        )
        
        introvert = st.selectbox(
            "👤 Introvert?",
            options=metadata['Introvert'],
            help="Do you identify as an introvert?"
        )
        
        teams = st.selectbox(
            "👥 Worked in Teams Ever?",
            options=metadata['worked in teams ever?'],
            help="Do you have experience collaborating in team settings?"
        )
        
        hard_smart = st.selectbox(
            "⚙️ Work Style",
            options=metadata['hard/smart worker'],
            help="Do you consider yourself a hard worker or a smart worker?"
        )

    st.write("---")
    st.subheader("🎯 Career Preferences & Context")
    col3, col4 = st.columns(2)
    
    with col3:
        career_area = st.selectbox(
            "💼 Interested Career Area",
            options=metadata['interested career area '],
            help="Which career track interests you the most?"
        )
        
        company_type = st.selectbox(
            "🏢 Target Company Type",
            options=metadata['Type of company want to settle in?'],
            help="What type of company do you prefer to settle in?"
        )
        
    with col4:
        mgmt_tech = st.selectbox(
            "⚖️ Management or Technical Track",
            options=metadata['Management or Technical'],
            help="Do you prefer management roles or pure engineering/technical roles?"
        )
        
        books = st.selectbox(
            "📖 Favorite Type of Books",
            options=metadata['Interested Type of Books'],
            help="What genre of books do you enjoy reading the most?"
        )
        
        senior_input = st.selectbox(
            "💬 Taken Inputs from Seniors/Elders?",
            options=metadata['Taken inputs from seniors or elders'],
            help="Have you sought career guidance or advice from seniors or mentors?"
        )
        
    submit_button = st.form_submit_button("🔮 Predict My Optimal Career Path")

# Perform prediction on submit
if submit_button:
    # 1. Structure the input data exactly matching the training feature columns
    # We should retrieve column names from preprocessor to be safe, but let's define them explicitly in order
    input_data = {
        'Logical quotient rating': coding_rating, # Wait! Let's check training columns:
        # Logical quotient rating, hackathons, coding skills rating, public speaking points, self-learning capability?,
        # Extra-courses did, certifications, workshops, reading and writing skills, memory capability score,
        # Interested subjects, interested career area , Type of company want to settle in?, Taken inputs from seniors or elders,
        # Interested Type of Books, Management or Technical, hard/smart worker, worked in teams ever?, Introvert
    }
    
    # Note: Column names in the original dataset had a space in 'interested career area '
    # Our train_model.py cleans names by df.columns.str.strip()!
    # So all column names are stripped of leading/trailing spaces.
    
    user_inputs = {
        'Logical quotient rating': logical_rating,
        'hackathons': hackathons,
        'coding skills rating': coding_rating,
        'public speaking points': public_speaking,
        'self-learning capability?': self_learning,
        'Extra-courses did': extra_courses,
        'certifications': cert,
        'workshops': workshop,
        'reading and writing skills': read_write,
        'memory capability score': memory,
        'Interested subjects': sub,
        'interested career area': career_area, # Note: stripped name
        'Type of company want to settle in?': company_type,
        'Taken inputs from seniors or elders': senior_input,
        'Interested Type of Books': books,
        'Management or Technical': mgmt_tech,
        'hard/smart worker': hard_smart,
        'worked in teams ever?': teams,
        'Introvert': introvert
    }
    
    # Create pandas DataFrame for the prediction
    input_df = pd.DataFrame([user_inputs])
    
    # Make prediction
    try:
        # Predict class probabilities
        probs = pipeline.predict_proba(input_df)[0]
        classes = pipeline.classes_
        
        # Sort in descending order of probabilities
        sorted_indices = np.argsort(probs)[::-1]
        sorted_classes = classes[sorted_indices]
        sorted_probs = probs[sorted_indices]
        
        top_role = sorted_classes[0]
        top_prob = sorted_probs[0] * 100
        
        # Display Results
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        
        # Best match card
        st.markdown(f"""
            <div class="prediction-container">
                <span class="metric-badge">🏆 PRIMARY RECOMMENDED PATH</span>
                <div class="prediction-role">{top_role}</div>
                <p style="font-size:1.15rem; margin-top:0.5rem; color:#A7F3D0;">
                    Our AI models predict a match probability of <b>{top_prob:.2f}%</b> for this role based on your profile!
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Plotly chart and details
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.markdown("### 📊 Match Probabilities by Role")
            # Build DataFrame for plot
            chart_df = pd.DataFrame({
                'Role': sorted_classes,
                'Probability (%)': sorted_probs * 100
            })
            # Take top 5 for better readability
            top_5_chart = chart_df.head(5)
            
            fig = px.bar(
                top_5_chart,
                y='Role',
                x='Probability (%)',
                orientation='h',
                color='Probability (%)',
                color_continuous_scale='purples',
                text='Probability (%)'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#E2E8F0',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=0, r=0, t=30, b=0),
                height=300
            )
            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside',
                marker_line_color='rgba(0,0,0,0)',
                marker_line_width=0
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col_res2:
            st.markdown("### 📋 Top Recommendations Breakdown")
            st.markdown(f"1. **{sorted_classes[0]}**: `{sorted_probs[0]*100:.1f}%` match probability")
            st.markdown(f"2. **{sorted_classes[1]}**: `{sorted_probs[1]*100:.1f}%` match probability")
            st.markdown(f"3. **{sorted_classes[2]}**: `{sorted_probs[2]*100:.1f}%` match probability")
            
            # Context info for top predicted job roles
            st.info(f"💡 **Career Tip**: Based on your choice of **{mgmt_tech}** alignment and **{sub}** interest, a role in **{top_role}** represents a strong synergy between your technical skills (coding rating: {coding_rating}/9) and career preferences.")
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)
