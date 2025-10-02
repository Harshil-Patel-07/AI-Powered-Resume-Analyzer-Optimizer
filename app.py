import streamlit as st
import os
import plotly.graph_objects as go
import plotly.express as px
from utils import (
    enhanced_pdf_extraction, 
    advanced_text_cleaning,
    extract_skills_by_category, 
    calculate_match_score,
    semantic_skill_matching,
    calculate_weighted_score,
    generate_optimization_suggestions,
    check_ats_compatibility,
    extract_experience_info,
    generate_visualization_data,
    ai_rewrite_bullet_points,
    generate_interview_questions,
    identify_weak_bullets,
    extract_experience_entries
)
from skills_data import skills_data

# Set page configuration
st.set_page_config(
    page_title="AI Resume Analyzer & Optimizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .score-card h3 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .score-card p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .skill-tag {
        display: inline-block;
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .skill-tag.matched {
        background: #e8f5e8;
        color: #2e7d32;
    }
    
    .skill-tag.missing {
        background: #ffebee;
        color: #c62828;
    }
    
    .semantic-match {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .suggestion-card {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .critical-suggestion {
        background: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .ats-score {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    .experience-badge {
        background: linear-gradient(135deg, #4caf50, #45a049);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        display: inline-block;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    
    .rewrite-box {
        background: #f5f5f5;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .original-text {
        background: #ffebee;
        padding: 0.8rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        font-style: italic;
    }
    
    .improved-text {
        background: #e8f5e9;
        padding: 0.8rem;
        border-radius: 5px;
        font-weight: 500;
    }
    
    .improvement-tag {
        display: inline-block;
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.2rem 0.6rem;
        margin: 0.2rem;
        border-radius: 12px;
        font-size: 0.75rem;
    }
    
    .interview-question {
        background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #5c6bc0;
    }
    
    .question-type {
        display: inline-block;
        background: #5c6bc0;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .star-section {
        background: white;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border-left: 3px solid #9c27b0;
    }
    
    .impact-score {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        margin-left: 0.5rem;
    }
    
    .score-low {
        background: #ffebee;
        color: #c62828;
    }
    
    .score-medium {
        background: #fff3e0;
        color: #ef6c00;
    }
    
    .score-high {
        background: #e8f5e9;
        color: #2e7d32;
    }
    
    .tab-content {
        padding: 1.5rem;
        background: white;
        border-radius: 10px;
        margin-top: 1rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def display_overview_tab(basic_score, weighted_score, experience_info, ats_results, enable_weighted, enable_ats):
    """Display Overview & Scores tab"""
    st.markdown("## 📊 Performance Overview")
    
    # Score cards in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="score-card">
            <h3>{basic_score}%</h3>
            <p>Basic Match Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if enable_weighted:
            st.markdown(f"""
            <div class="score-card">
                <h3>{weighted_score}%</h3>
                <p>Smart Score</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Enable Smart Scoring in sidebar")
    
    with col3:
        if enable_ats and ats_results:
            ats_score = ats_results.get('ats_score', 0)
            score_color = "#4caf50" if ats_score >= 80 else "#ff9800" if ats_score >= 60 else "#f44336"
            st.markdown(f"""
            <div class="score-card" style="background: {score_color};">
                <h3>{ats_score}%</h3>
                <p>ATS Score</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Enable ATS Check in sidebar")
    
    # Experience Badge
    if experience_info.get('total_years', 0) > 0:
        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0;">
            <div class="experience-badge">
                📅 Total Experience: {experience_info['total_years']} years
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Metrics Summary
    st.markdown("---")
    st.markdown("### 📈 Quick Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        improvement = weighted_score - basic_score if enable_weighted else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">+{improvement:.1f}%</div>
            <div class="metric-label">Smart Score Boost</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{experience_info.get('total_years', 0)}</div>
            <div class="metric-label">Years Experience</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        positions = len(experience_info.get('positions', []))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{positions}</div>
            <div class="metric-label">Positions Held</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if enable_ats and ats_results:
            issues_count = len(ats_results.get('issues', []))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{issues_count}</div>
                <div class="metric-label">ATS Issues Found</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ATS Detailed Results
    if enable_ats and ats_results:
        st.markdown("---")
        st.markdown("### 🤖 ATS Compatibility Details")
        
        if ats_results.get('issues'):
            st.markdown("**⚠️ Issues Found:**")
            for issue in ats_results['issues']:
                st.markdown(f"• {issue}")
        
        if ats_results.get('recommendations'):
            st.markdown("**✅ Recommendations:**")
            for rec in ats_results['recommendations']:
                st.markdown(f"• {rec}")

def display_skills_tab(viz_data, semantic_matches, details, enable_visualizations, enable_semantic):
    """Display Skills Analysis tab"""
    
    # Interactive Visualizations
    if enable_visualizations and viz_data:
        st.markdown("## 📊 Interactive Skill Visualizations")
        
        # Radar Chart
        if viz_data.get('radar_data'):
            st.markdown("### 🎯 Skills Comparison Radar")
            radar_fig = create_radar_chart(viz_data['radar_data'])
            st.plotly_chart(radar_fig, use_container_width=True)
        
        # Category Bar Chart
        if viz_data.get('category_scores'):
            st.markdown("### 📈 Category-wise Match Scores")
            bar_fig = create_category_bar_chart(viz_data['category_scores'])
            st.plotly_chart(bar_fig, use_container_width=True)
        
        # Bubble Chart
        if viz_data.get('bubble_data'):
            st.markdown("### 🔍 Skill Gap Analysis")
            bubble_fig = create_bubble_chart(viz_data['bubble_data'])
            st.plotly_chart(bubble_fig, use_container_width=True)
        
        st.markdown("---")
    
    # AI Semantic Matches
    if enable_semantic and semantic_matches:
        st.markdown("## 🧠 AI Semantic Skill Matches")
        st.markdown("*Skills matched using AI understanding (not just exact words)*")
        
        for category, matches in semantic_matches.items():
            if matches:
                st.markdown(f"**{category.title()}:**")
                for match in matches:
                    similarity_percent = int(match['similarity'] * 100)
                    st.markdown(f"""
                    <div class="semantic-match">
                        "{match['jd_skill']}" ↔ "{match['resume_skill']}"
                        <br><strong>{similarity_percent}% similar</strong>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Traditional Skill Matching
    st.markdown("## 🎯 Traditional Skill Matching")
    
    # Category breakdown
    for category, data in details.items():
        if data['required'] or data['matched'] or data['missing']:
            with st.expander(f"📂 {category.upper()}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Required:**")
                    if data['required']:
                        for skill in data['required']:
                            st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
                    else:
                        st.info("None")
                
                with col2:
                    st.markdown("**Matched:**")
                    if data['matched']:
                        for skill in data['matched']:
                            st.markdown(f'<span class="skill-tag matched">{skill}</span>', unsafe_allow_html=True)
                    else:
                        st.warning("None")
                
                with col3:
                    st.markdown("**Missing:**")
                    if data['missing']:
                        for skill in data['missing']:
                            st.markdown(f'<span class="skill-tag missing">{skill}</span>', unsafe_allow_html=True)
                    else:
                        st.success("None")
                
                # Category score
                match_percent = data.get('match_percentage', 0)
                st.progress(match_percent / 100)
                st.markdown(f"**Category Score: {match_percent}%**")

def display_optimizer_tab(rewritten_bullets, suggestions, enable_rewriter, enable_suggestions):
    """Display Resume Optimizer tab"""
    
    # AI Resume Rewriter
    if enable_rewriter and rewritten_bullets:
        st.markdown("## ✨ AI Resume Rewriter")
        st.markdown("*AI-enhanced bullet points with stronger impact*")
        
        for idx, bullet in enumerate(rewritten_bullets):
            with st.expander(f"📝 Bullet Point {idx + 1} - Impact Score: {bullet['impact_score']}/10", expanded=(idx < 2)):
                st.markdown('<div class="rewrite-box">', unsafe_allow_html=True)
                
                # Original
                st.markdown("**❌ Original:**")
                st.markdown(f'<div class="original-text">{bullet["original"]}</div>', unsafe_allow_html=True)
                
                # Improved
                st.markdown("**✅ AI-Enhanced:**")
                st.markdown(f'<div class="improved-text">{bullet["improved"]}</div>', unsafe_allow_html=True)
                
                # Improvements
                st.markdown("**🎯 Improvements:**")
                for improvement in bullet['improvements']:
                    st.markdown(f'<span class="improvement-tag">{improvement}</span>', unsafe_allow_html=True)
                
                # Copy button
                if st.button(f"📋 Copy Enhanced Version", key=f"copy_{idx}"):
                    st.success("✅ Copied to clipboard! (Use Ctrl+C to copy the text above)")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Optimization Suggestions
    if enable_suggestions and suggestions:
        st.markdown("## 💡 Optimization Suggestions")
        
        # Group suggestions by priority
        critical_suggestions = [s for s in suggestions if s['type'] == 'critical']
        other_suggestions = [s for s in suggestions if s['type'] != 'critical']
        
        if critical_suggestions:
            st.markdown("### 🚨 Critical Improvements")
            for suggestion in critical_suggestions:
                st.markdown(f"""
                <div class="critical-suggestion">
                    <strong>🚨 {suggestion['message']}</strong>
                    <br>Action: {suggestion['action']}
                </div>
                """, unsafe_allow_html=True)
        
        if other_suggestions:
            st.markdown("### 💡 Recommended Improvements")
            for suggestion in other_suggestions:
                st.markdown(f"""
                <div class="suggestion-card">
                    <strong>💡 {suggestion['message']}</strong>
                    <br>Action: {suggestion['action']}
                </div>
                """, unsafe_allow_html=True)
    
    # Show message if features are disabled
    if not enable_rewriter and not enable_suggestions:
        st.info("Enable 'AI Resume Rewriter' and 'Optimization Tips' in the sidebar to see recommendations.")

def display_interview_tab(interview_questions, enable_interview):
    """Display Interview Preparation tab"""
    
    if enable_interview and interview_questions:
        st.markdown("## 🎤 AI-Generated Interview Questions")
        st.markdown("*Personalized questions based on your resume and job requirements*")
        
        # Create tabs for different question types
        tab1, tab2, tab3 = st.tabs(["🎯 Technical Questions", "💼 Behavioral Questions", "🔍 Gap-Based Questions"])
        
        with tab1:
            if interview_questions.get('technical'):
                for idx, q in enumerate(interview_questions['technical'][:5]):
                    display_interview_question(q, idx, "technical")
            else:
                st.info("No technical questions generated")
        
        with tab2:
            if interview_questions.get('behavioral'):
                for idx, q in enumerate(interview_questions['behavioral'][:5]):
                    display_interview_question(q, idx, "behavioral")
            else:
                st.info("No behavioral questions generated")
        
        with tab3:
            if interview_questions.get('gap_based'):
                for idx, q in enumerate(interview_questions['gap_based'][:5]):
                    display_interview_question(q, idx, "gap")
            else:
                st.info("No gap-based questions generated")
    else:
        st.info("Enable 'Interview Question Generator' in the sidebar to see personalized interview questions.")

def display_interview_question(question_data, idx, q_type):
    """Display a single interview question with STAR template"""
    st.markdown(f"""
    <div class="interview-question">
        <span class="question-type">{q_type.upper()}</span>
        <h4 style="margin: 0.5rem 0;">Q{idx + 1}: {question_data['question']}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("💡 STAR Method Answer Template"):
        star = question_data.get('star_template', {})
        
        st.markdown(f"""
        <div class="star-section">
            <strong>📍 Situation:</strong><br>
            {star.get('situation', 'Describe the context and background...')}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="star-section">
            <strong>🎯 Task:</strong><br>
            {star.get('task', 'Explain what you needed to accomplish...')}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="star-section">
            <strong>⚡ Action:</strong><br>
            {star.get('action', 'Detail the specific steps you took...')}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="star-section">
            <strong>🏆 Result:</strong><br>
            {star.get('result', 'Share the quantifiable outcomes...')}
        </div>
        """, unsafe_allow_html=True)
        
        if question_data.get('tips'):
            st.markdown("**💡 Pro Tips:**")
            for tip in question_data['tips']:
                st.markdown(f"• {tip}")

def display_detailed_report_tab(details, resume_skills, jd_skills, basic_score, weighted_score):
    """Display Detailed Report tab"""
    st.markdown("## 📈 Comprehensive Analysis Report")
    
    # Summary Statistics
    st.markdown("### 📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_required = sum(len(data['required']) for data in details.values())
    total_matched = sum(len(data['matched']) for data in details.values())
    total_missing = sum(len(data['missing']) for data in details.values())
    
    with col1:
        st.metric("Total Required Skills", total_required)
    with col2:
        st.metric("Skills Matched", total_matched, delta=f"{(total_matched/total_required*100):.0f}%" if total_required > 0 else "0%")
    with col3:
        st.metric("Skills Missing", total_missing)
    with col4:
        st.metric("Basic Score", f"{basic_score}%")
    
    st.markdown("---")
    
    # Category Breakdown Table
    st.markdown("### 📋 Category-wise Breakdown")
    
    import pandas as pd
    
    category_data = []
    for category, data in details.items():
        category_data.append({
            'Category': category.upper(),
            'Required': len(data['required']),
            'Matched': len(data['matched']),
            'Missing': len(data['missing']),
            'Match %': f"{data.get('match_percentage', 0)}%"
        })
    
    df = pd.DataFrame(category_data)
    st.dataframe(df, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed Category Analysis
    st.markdown("### 🔍 Detailed Category Analysis")
    
    for category, data in details.items():
        if data['required']:
            with st.expander(f"📂 {category.upper()} - {data.get('match_percentage', 0)}% Match"):
                st.markdown(f"**Total Required:** {len(data['required'])}")
                st.markdown(f"**Matched:** {len(data['matched'])}")
                st.markdown(f"**Missing:** {len(data['missing'])}")
                
                st.markdown("---")
                
                if data['matched']:
                    st.markdown("**✅ Matched Skills:**")
                    st.write(", ".join(data['matched']))
                
                if data['missing']:
                    st.markdown("**❌ Missing Skills:**")
                    st.write(", ".join(data['missing']))

def create_radar_chart(radar_data):
    """Create radar chart for skill comparison"""
    categories = list(radar_data.keys())
    resume_values = [radar_data[cat]['resume'] for cat in categories]
    jd_values = [radar_data[cat]['required'] for cat in categories]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=resume_values,
        theta=categories,
        fill='toself',
        name='Your Resume',
        line_color='#667eea',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=jd_values,
        theta=categories,
        fill='toself',
        name='Job Requirements',
        line_color='#764ba2',
        fillcolor='rgba(118, 75, 162, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        height=500
    )
    
    return fig

def create_category_bar_chart(category_scores):
    """Create bar chart for category scores"""
    categories = list(category_scores.keys())
    scores = [category_scores[cat] for cat in categories]
    
    colors = ['#4caf50' if s >= 70 else '#ff9800' if s >= 40 else '#f44336' for s in scores]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=scores,
            marker_color=colors,
            text=scores,
            texttemplate='%{text}%',
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Match Percentage by Skill Category",
        xaxis_title="Skill Categories",
        yaxis_title="Match Percentage (%)",
        yaxis=dict(range=[0, 110]),
        height=400
    )
    
    return fig

def create_bubble_chart(bubble_data):
    """Create bubble chart for skill gap analysis"""
    import pandas as pd
    
    df = pd.DataFrame(bubble_data)
    
    fig = px.scatter(df, 
                     x='category', 
                     y='importance',
                     size='count',
                     color='priority',
                     hover_data=['skills'],
                     color_discrete_map={'High': '#f44336', 'Medium': '#ff9800', 'Low': '#4caf50'},
                     title="Skill Gaps by Priority")
    
    fig.update_layout(height=400)
    
    return fig

def main():
    # Main header
    st.markdown('<h1 class="main-header">🤖 AI-Powered Resume Analyzer & Optimizer</h1>', unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.markdown("## ⚙️ Analysis Settings")
        
        # Job title input for weighted scoring
        job_title = st.text_input(
            "Job Title (Optional)", 
            placeholder="e.g., Senior Data Scientist",
            help="Helps optimize scoring weights for specific roles"
        )
        
        # Semantic similarity threshold
        semantic_threshold = st.slider(
            "AI Similarity Threshold", 
            min_value=0.5, 
            max_value=0.9, 
            value=0.7, 
            step=0.05,
            help="How similar skills need to be for AI matching (higher = stricter)"
        )
        
        # Analysis options
        st.markdown("## 🔍 Analysis Options")
        enable_semantic = st.checkbox("Enable AI Semantic Matching", value=True)
        enable_weighted = st.checkbox("Enable Smart Scoring", value=True)
        enable_suggestions = st.checkbox("Enable Optimization Tips", value=True)
        enable_ats = st.checkbox("Enable ATS Check", value=True)
        
        # NEW OPTIONS
        st.markdown("## ✨ Advanced Features")
        enable_visualizations = st.checkbox("📊 Interactive Visualizations", value=True)
        enable_rewriter = st.checkbox("✨ AI Resume Rewriter", value=True)
        enable_interview = st.checkbox("🎤 Interview Question Generator", value=True)
        
        st.markdown("## 📋 How to Use")
        st.markdown("""
        1. **Upload Resume**: PDF format preferred
        2. **Job Description**: Paste full job posting
        3. **Configure Settings**: Adjust options above
        4. **Analyze**: Get comprehensive insights
        """)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Upload Your Resume")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type=['pdf'],
            help="Upload your resume in PDF format (Max 10MB)"
        )
        
        st.subheader("📋 Job Description")
        job_description = st.text_area(
            "Paste the job description here:",
            height=200,
            placeholder="Enter the complete job description including requirements, responsibilities, and qualifications..."
        )
        
        # Submit button
        submit_button = st.button("🚀 ANALYZE RESUME", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📊 Analysis Results")
        st.info("👈 Upload your resume and paste the job description, then click 'ANALYZE RESUME'")
    
    # Process when submit button is clicked
    if submit_button:
        if uploaded_file is not None and job_description.strip():
            with st.spinner("🔍 Analyzing your resume with AI..."):
                try:
                    # Create uploads directory if it doesn't exist
                    os.makedirs("uploads", exist_ok=True)
                    
                    # Save uploaded file
                    file_path = os.path.join("uploads", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Extract text from PDF using enhanced method
                    resume_text = enhanced_pdf_extraction(file_path)
                    
                    # Advanced text cleaning
                    resume_text_cleaned = advanced_text_cleaning(resume_text)
                    jd_text_cleaned = advanced_text_cleaning(job_description)
                    
                    # Extract skills
                    resume_skills = extract_skills_by_category(resume_text_cleaned, skills_data)
                    jd_skills = extract_skills_by_category(jd_text_cleaned, skills_data)
                    
                    # Calculate basic match score
                    basic_score, details = calculate_match_score(resume_skills, jd_skills)
                    
                    # Initialize variables for optional features
                    semantic_matches = {}
                    weighted_score = basic_score
                    suggestions = []
                    ats_results = {}
                    experience_info = {}
                    viz_data = {}
                    rewritten_bullets = []
                    interview_questions = {}
                    
                    # Semantic matching (if enabled)
                    if enable_semantic and resume_skills and jd_skills:
                        semantic_matches = semantic_skill_matching(
                            resume_skills, jd_skills, threshold=semantic_threshold
                        )
                    
                    # Weighted scoring (if enabled)
                    if enable_weighted:
                        weighted_score = calculate_weighted_score(details, job_title)
                    
                    # Generate suggestions (if enabled)
                    if enable_suggestions:
                        suggestions = generate_optimization_suggestions(
                            details, weighted_score, resume_text
                        )
                    
                    # ATS compatibility check (if enabled)
                    if enable_ats:
                        ats_results = check_ats_compatibility(resume_text, file_path)
                    
                    # Extract experience information
                    experience_info = extract_experience_info(resume_text)
                    
                    # NEW FEATURES
                    # Generate visualization data
                    if enable_visualizations:
                        viz_data = generate_visualization_data(resume_skills, jd_skills, details)
                    
                    # AI Resume Rewriter
                    if enable_rewriter:
                        missing_skills = []
                        for cat in details.values():
                            missing_skills.extend(cat['missing'][:3])  # Top 3 per category
                        rewritten_bullets = ai_rewrite_bullet_points(resume_text, job_description, missing_skills[:5])
                    
                    # Interview Questions Generator
                    if enable_interview:
                        interview_questions = generate_interview_questions(
                            resume_text, resume_skills, jd_skills, details, experience_info
                        )
                    
                    # Clean up uploaded file
                    os.remove(file_path)
                    
                    # Success message
                    improvement_percentage = weighted_score - basic_score if enable_weighted else 0
                    semantic_matches_count = sum(len(matches) for matches in semantic_matches.values()) if semantic_matches else 0
                    
                    success_message = f"✅ Analysis completed! "
                    if semantic_matches_count > 0:
                        success_message += f"Found {semantic_matches_count} AI matches. "
                    if improvement_percentage > 0:
                        success_message += f"Smart scoring improved by {improvement_percentage:.1f}%"
                    
                    st.success(success_message)
                    
                    # DISPLAY RESULTS IN TABS
                    st.markdown("---")
                    st.markdown("## 📑 Analysis Results")
                    
                    # Create main tabs
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "📊 Overview & Scores",
                        "🎯 Skills Analysis", 
                        "✨ Resume Optimizer",
                        "🎤 Interview Preparation",
                        "📈 Detailed Report"
                    ])
                    
                    # Tab 1: Overview & Scores
                    with tab1:
                        display_overview_tab(
                            basic_score, 
                            weighted_score, 
                            experience_info, 
                            ats_results, 
                            enable_weighted, 
                            enable_ats
                        )
                    
                    # Tab 2: Skills Analysis
                    with tab2:
                        display_skills_tab(
                            viz_data, 
                            semantic_matches, 
                            details, 
                            enable_visualizations, 
                            enable_semantic
                        )
                    
                    # Tab 3: Resume Optimizer
                    with tab3:
                        display_optimizer_tab(
                            rewritten_bullets, 
                            suggestions, 
                            enable_rewriter, 
                            enable_suggestions
                        )
                    
                    # Tab 4: Interview Preparation
                    with tab4:
                        display_interview_tab(
                            interview_questions, 
                            enable_interview
                        )
                    
                    # Tab 5: Detailed Report
                    with tab5:
                        display_detailed_report_tab(
                            details, 
                            resume_skills, 
                            jd_skills, 
                            basic_score, 
                            weighted_score
                        )
                
                except Exception as e:
                    st.error(f"❌ Error processing your resume: {str(e)}")
                    st.error("Please make sure your PDF is valid and try again.")
                    # Optional: Show more detailed error for debugging
                    if st.checkbox("Show detailed error (for debugging)"):
                        st.exception(e)
        
        elif uploaded_file is None:
            st.warning("⚠️ Please upload a PDF file.")
        elif not job_description.strip():
            st.warning("⚠️ Please enter a job description.")
    
    # Footer with features
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 🧠 AI Features")
        st.markdown("• Semantic matching\n• Smart scoring\n• Experience extraction")
    
    with col2:
        st.markdown("### 📊 Visualizations")
        st.markdown("• Radar charts\n• Skill gaps\n• Interactive plots")
    
    with col3:
        st.markdown("### ✨ AI Rewriter")
        st.markdown("• Enhanced bullets\n• Impact scoring\n• STAR method")
    
    with col4:
        st.markdown("### 🎤 Interview Prep")
        st.markdown("• Custom questions\n• STAR templates\n• Gap analysis")

if __name__ == "__main__":
    main()
