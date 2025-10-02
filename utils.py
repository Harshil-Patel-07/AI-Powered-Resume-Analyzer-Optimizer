import re
import fitz
import pdfplumber
import PyPDF2
import nltk
import spacy
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import dateparser
from datetime import datetime
import random

#================================================================================= 
# STEP 0: Load the models
#=================================================================================
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

#================================================================================= 
# STEP 1: Extract text from the resume PDF
#=================================================================================
def enhanced_pdf_extraction(pdf_path):
    """Extract text from PDF using multiple methods for better accuracy."""
    text_methods = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text1 = ""
            for page in pdf.pages:
                text1 += page.extract_text() or ""
        text_methods.append(text1)
    except:
        pass
    
    try:
        doc = fitz.open(pdf_path)
        text2 = ""
        for page_num in range(len(doc)):
            text2 += doc.load_page(page_num).get_text()
        text_methods.append(text2)
    except:
        pass
    
    return max(text_methods, key=len) if text_methods else ""

#================================================================================= 
# STEP 2: Clean the text
#=================================================================================
def advanced_text_cleaning(text):
    """Advanced text cleaning that preserves important punctuation."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'[^\w\s\+\#\.\-]', ' ', text)
    return text

#================================================================================= 
# STEP 4: Extract skills by category
#=================================================================================
def extract_skills_by_category(text, skills_data):
    """Extract skills from text by matching against predefined skill categories."""
    matched_skills = {}
    for category, skills in skills_data.items():
        found = []
        for skill in skills:
            skill_lower = skill.lower()
            if re.search(r'\b' + re.escape(skill_lower) + r'\b', text):
                found.append(skill)
        if found:
            matched_skills[category] = found
    return matched_skills

#================================================================================= 
# STEP 5: Calculate match score
#=================================================================================
def calculate_match_score(resume_skills, jd_skills):
    """Calculate percentage match between resume and job description skills."""
    total_required = 0
    total_matched = 0
    detailed_result = {}
    
    for category in jd_skills:
        jd_category_skills = set(jd_skills[category])
        resume_category_skills = set(resume_skills.get(category, []))
        matched = jd_category_skills & resume_category_skills
        
        total_required += len(jd_category_skills)
        total_matched += len(matched)
        
        detailed_result[category] = {
            "required": list(jd_category_skills),
            "matched": list(matched),
            "missing": list(jd_category_skills - resume_category_skills)
        }
    
    match_score = round((total_matched / total_required) * 100, 2) if total_required > 0 else 0
    return match_score, detailed_result

#================================================================================= 
# STEP 6: Calculate semantic match
#=================================================================================
def semantic_skill_matching(resume_skills, jd_skills, threshold=0.7):
    """Find semantically similar skills using AI."""
    semantic_matches = {}
    
    for category in jd_skills:
        jd_category_skills = jd_skills[category]
        resume_category_skills = resume_skills.get(category, [])
        semantic_matched = []
        
        for jd_skill in jd_category_skills:
            best_match = None
            best_score = 0
            
            for resume_skill in resume_category_skills:
                embeddings = semantic_model.encode([jd_skill, resume_skill])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                
                if similarity > threshold and similarity > best_score:
                    best_match = resume_skill
                    best_score = similarity
            
            if best_match:
                semantic_matched.append({
                    'jd_skill': jd_skill,
                    'resume_skill': best_match,
                    'similarity': round(best_score, 2)
                })
        
        if semantic_matched:
            semantic_matches[category] = semantic_matched
    
    return semantic_matches

#================================================================================= 
# STEP 7: Calculate weighted score
#=================================================================================
def get_skill_weights(job_title=""):
    """Get category weights based on job title."""
    base_weights = {
        'programming': 0.35,
        'technical': 0.30,
        'frameworks': 0.20,
        'soft_skills': 0.10,
        'tools': 0.05
    }
    
    job_title_lower = job_title.lower()
    
    if 'senior' in job_title_lower or 'lead' in job_title_lower:
        base_weights['soft_skills'] = 0.20
        base_weights['technical'] = 0.25
    
    if 'data' in job_title_lower:
        base_weights['technical'] = 0.40
        base_weights['tools'] = 0.15
    
    return base_weights

def calculate_weighted_score(detailed_result, job_title=""):
    """Calculate weighted match score based on job-specific importance."""
    weights = get_skill_weights(job_title)
    
    total_weighted_score = 0
    total_weight = 0
    
    for category, results in detailed_result.items():
        if category in weights:
            required_count = len(results['required'])
            matched_count = len(results['matched'])
            
            if required_count > 0:
                category_score = matched_count / required_count
                weighted_contribution = category_score * weights[category]
                total_weighted_score += weighted_contribution
                total_weight += weights[category]
    
    final_score = (total_weighted_score / total_weight * 100) if total_weight > 0 else 0
    return round(final_score, 2)

#================================================================================= 
# STEP 8: Generate optimization suggestions
#=================================================================================
def generate_optimization_suggestions(detailed_result, match_score, resume_text):
    """Generate actionable suggestions to improve resume."""
    suggestions = []
    priority_suggestions = []
    
    if match_score < 70:
        suggestions.append({
            'type': 'important',
            'message': 'Your resume could be more competitive.',
            'action': 'Add missing key skills and relevant experience'
        })
    
    for category, results in detailed_result.items():
        missing_skills = results['missing']
        if len(missing_skills) > 0:
            top_missing = missing_skills[:3]
            suggestions.append({
                'type': 'skill_gap',
                'category': category,
                'message': f'Missing important {category} skills',
                'action': f'Consider adding: {", ".join(top_missing)}',
                'skills': top_missing
            })
    
    if len(resume_text.split()) < 200:
        suggestions.append({
            'type': 'content',
            'message': 'Resume appears too brief',
            'action': 'Add more detailed descriptions of your experience and achievements'
        })
    
    return priority_suggestions + suggestions

#================================================================================= 
# STEP 9: Check ATS compatibility
#=================================================================================
def check_ats_compatibility(resume_text, pdf_path):
    """Check resume compatibility with Applicant Tracking Systems."""
    issues = []
    recommendations = []
    
    if not re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text):
        issues.append("Missing email address")
        recommendations.append("Add a professional email address")
    
    if not re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', resume_text):
        issues.append("Missing phone number")
        recommendations.append("Add your phone number")
    
    standard_sections = ['experience', 'education', 'skills']
    for section in standard_sections:
        if not re.search(f'\\b{section}\\b', resume_text.lower()):
            issues.append(f"Missing {section} section")
            recommendations.append(f"Add a clear {section} section")
    
    special_char_ratio = len(re.findall(r'[^\w\s]', resume_text)) / len(resume_text)
    if special_char_ratio > 0.1:
        issues.append("Too many special characters")
        recommendations.append("Simplify formatting and reduce special characters")
    
    word_count = len(resume_text.split())
    if word_count < 150:
        issues.append("Resume too short")
        recommendations.append("Add more detailed content (aim for 200-400 words)")
    elif word_count > 800:
        issues.append("Resume too long")
        recommendations.append("Condense content to essential information")
    
    ats_score = max(0, 100 - len(issues) * 15)
    
    return {
        'ats_score': ats_score,
        'issues': issues,
        'recommendations': recommendations
    }

#================================================================================= 
# STEP 10: Extract experience info
#=================================================================================
def extract_experience_info(resume_text):
    """Extract years of experience from resume."""
    experience_data = {
        'total_years': 0,
        'experience_entries': [],
        'current_role': None
    }
    
    experience_patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
        r'(\d+)\+?\s*yrs?\s*(?:of\s*)?(?:experience|exp)',
        r'(\d{4})\s*[-–]\s*(?:present|current|\d{4})',
        r'(\d{1,2}/\d{4})\s*[-–]\s*(?:present|current|\d{1,2}/\d{4})'
    ]
    
    years_mentioned = []
    for pattern in experience_patterns:
        matches = re.findall(pattern, resume_text.lower())
        years_mentioned.extend(matches)
    
    if years_mentioned:
        numeric_years = []
        for year in years_mentioned:
            try:
                numeric_years.append(int(year))
            except:
                continue
        
        if numeric_years:
            experience_data['total_years'] = max(numeric_years)
    
    return experience_data

#=================================================================================
# NEW FEATURE 1: INTERACTIVE VISUALIZATIONS
#=================================================================================
def generate_visualization_data(resume_skills, jd_skills, detailed_result):
    """Generate data for interactive visualizations."""
    viz_data = {}
    
    # Radar chart data
    radar_data = {}
    for category in detailed_result:
        required = len(detailed_result[category]['required'])
        matched = len(detailed_result[category]['matched'])
        
        radar_data[category] = {
            'required': 100,  # Normalized to 100
            'resume': round((matched / required * 100) if required > 0 else 0, 1)
        }
    
    viz_data['radar_data'] = radar_data
    
    # Category scores for bar chart
    category_scores = {}
    for category in detailed_result:
        required = len(detailed_result[category]['required'])
        matched = len(detailed_result[category]['matched'])
        category_scores[category] = round((matched / required * 100) if required > 0 else 0, 1)
    
    viz_data['category_scores'] = category_scores
    
    # Bubble chart data for skill gaps
    bubble_data = []
    for category in detailed_result:
        missing = detailed_result[category]['missing']
        if missing:
            priority = 'High' if len(missing) >= 3 else 'Medium' if len(missing) == 2 else 'Low'
            bubble_data.append({
                'category': category,
                'count': len(missing),
                'importance': 10 - len(missing),  # Inverse for positioning
                'priority': priority,
                'skills': ', '.join(missing[:3])
            })
    
    viz_data['bubble_data'] = bubble_data
    
    return viz_data

#=================================================================================
# NEW FEATURE 2: AI RESUME REWRITER
#=================================================================================
def identify_weak_bullets(resume_text):
    """Identify weak bullet points in resume."""
    # Split into lines and find bullet points
    lines = resume_text.split('\n')
    bullets = []
    
    bullet_patterns = [r'^\s*[-•*]\s+', r'^\s*\d+[\.\)]\s+']
    
    for line in lines:
        line = line.strip()
        if any(re.match(pattern, line) for pattern in bullet_patterns):
            # Remove bullet markers
            clean_line = re.sub(r'^\s*[-•*\d+\.\)]\s+', '', line)
            if len(clean_line.split()) >= 5:  # At least 5 words
                bullets.append(clean_line)
    
    # If no bullets found, extract sentences from experience section
    if not bullets:
        exp_section = extract_text_between_sections(resume_text, 'experience', 'education')
        if exp_section:
            sentences = re.split(r'[.!?]+', exp_section)
            bullets = [s.strip() for s in sentences if len(s.split()) >= 5][:5]
    
    return bullets[:5]  # Return top 5

def extract_text_between_sections(text, start_keyword, end_keyword):
    """Extract text between two section headers."""
    text_lower = text.lower()
    start_idx = text_lower.find(start_keyword)
    end_idx = text_lower.find(end_keyword, start_idx) if start_idx != -1 else -1
    
    if start_idx != -1:
        if end_idx != -1:
            return text[start_idx:end_idx]
        else:
            return text[start_idx:start_idx+500]  # Get next 500 chars
    return ""

def calculate_impact_score(bullet_text):
    """Calculate impact score for a bullet point."""
    score = 5  # Base score
    
    # Check for quantifiable metrics
    if re.search(r'\d+%', bullet_text):
        score += 2
    if re.search(r'\$\d+', bullet_text):
        score += 2
    if re.search(r'\d+[kmb]?\+?', bullet_text, re.IGNORECASE):
        score += 1
    
    # Check for action verbs
    strong_verbs = ['achieved', 'developed', 'implemented', 'led', 'managed', 'created', 
                    'improved', 'increased', 'reduced', 'optimized', 'designed', 'built']
    if any(verb in bullet_text.lower() for verb in strong_verbs):
        score += 1
    
    # Check length (too short or too long)
    word_count = len(bullet_text.split())
    if 10 <= word_count <= 25:
        score += 1
    
    return min(score, 10)  # Cap at 10

def rewrite_with_action_verbs(bullet_text, missing_skills):
    """Rewrite bullet point with stronger action verbs and missing skills."""
    strong_verbs = ['Developed', 'Implemented', 'Designed', 'Led', 'Architected', 
                    'Optimized', 'Engineered', 'Built', 'Created', 'Improved']
    
    # Select a random strong verb
    verb = random.choice(strong_verbs)
    
    # Try to inject missing skills naturally
    skill_to_add = missing_skills[0] if missing_skills else None
    
    # Simple rewriting logic (in production, use GPT/Claude API)
    improved = bullet_text
    
    # Add quantifiable metric if missing
    if not re.search(r'\d+%|\$\d+|\d+[kmb]', bullet_text, re.IGNORECASE):
        improved += ", resulting in 25% efficiency improvement"
    
    # Add skill if missing
    if skill_to_add and skill_to_add.lower() not in improved.lower():
        improved += f" using {skill_to_add}"
    
    # Make it start with strong verb
    if not any(improved.lower().startswith(v.lower()) for v in strong_verbs):
        improved = f"{verb} {improved.lower()}"
    
    return improved

def ai_rewrite_bullet_points(resume_text, jd_text, missing_skills):
    """Rewrite resume bullet points with AI enhancement."""
    weak_bullets = identify_weak_bullets(resume_text)
    
    rewritten_bullets = []
    
    for bullet in weak_bullets:
        impact_score = calculate_impact_score(bullet)
        improved = rewrite_with_action_verbs(bullet, missing_skills)
        
        improvements = []
        
        # Identify what was improved
        if impact_score < 7:
            improvements.append("Added quantifiable metrics")
        
        if not any(verb in bullet.lower() for verb in ['developed', 'led', 'implemented']):
            improvements.append("Stronger action verb")
        
        if missing_skills and missing_skills[0].lower() not in bullet.lower():
            improvements.append(f"Included skill: {missing_skills[0]}")
        
        if len(improvements) == 0:
            improvements.append("Enhanced clarity and impact")
        
        rewritten_bullets.append({
            'original': bullet,
            'improved': improved,
            'impact_score': calculate_impact_score(improved),
            'improvements': improvements
        })
    
    return rewritten_bullets

def extract_experience_entries(resume_text):
    """Extract individual experience entries from resume."""
    entries = []
    
    # Look for date patterns indicating job entries
    date_pattern = r'(\d{4})\s*[-–]\s*(?:present|current|\d{4})'
    matches = re.finditer(date_pattern, resume_text.lower())
    
    positions = [m.start() for m in matches]
    
    for i, pos in enumerate(positions):
        start = pos
        end = positions[i+1] if i+1 < len(positions) else pos + 300
        entry = resume_text[start:end]
        entries.append(entry)
    
    return entries

#=================================================================================
# NEW FEATURE 3: INTERVIEW QUESTION GENERATOR
#=================================================================================
def generate_interview_questions(resume_text, resume_skills, jd_skills, detailed_result, experience_info):
    """Generate personalized interview questions based on resume and JD."""
    questions = {
        'technical': [],
        'behavioral': [],
        'gap_based': []
    }
    
    # Technical questions based on JD skills
    all_jd_skills = []
    for category in jd_skills:
        all_jd_skills.extend(jd_skills[category][:3])  # Top 3 per category
    
    for skill in all_jd_skills[:5]:
        questions['technical'].append({
            'question': f"Can you describe your experience with {skill}?",
            'star_template': {
                'situation': f"In my previous role working with {skill}...",
                'task': f"I was tasked with implementing/optimizing...",
                'action': f"I used {skill} to develop/create...",
                'result': "Which resulted in [quantifiable outcome like 30% improvement]"
            },
            'tips': [
                f"Mention specific projects where you used {skill}",
                "Quantify your impact with numbers and percentages",
                "Discuss challenges you overcame"
            ]
        })
    
    # Behavioral questions
    behavioral_templates = [
        "Tell me about a time when you had to work with a difficult team member.",
        "Describe a situation where you had to meet a tight deadline.",
        "Give an example of a project where you demonstrated leadership.",
        "Tell me about a time you failed and what you learned from it.",
        "Describe how you handle conflicting priorities."
    ]
    
    exp_entries = extract_experience_entries(resume_text)
    sample_experience = exp_entries[0][:100] if exp_entries else "In my recent role"
    
    for template in behavioral_templates[:5]:
        questions['behavioral'].append({
            'question': template,
            'star_template': {
                'situation': f"{sample_experience}...",
                'task': "I needed to [describe the challenge or goal]...",
                'action': "I took the following steps: [list specific actions]...",
                'result': "This led to [positive outcome with metrics]"
            },
            'tips': [
                "Use the STAR method to structure your answer",
                "Be specific with examples from your experience",
                "Focus on your individual contribution"
            ]
        })
    
    # Gap-based questions (about missing skills)
    for category in detailed_result:
        missing = detailed_result[category]['missing']
        if missing:
            for skill in missing[:2]:  # Top 2 missing per category
                questions['gap_based'].append({
                    'question': f"I notice you don't mention {skill} on your resume. Do you have any experience with it?",
                    'star_template': {
                        'situation': f"While I haven't used {skill} extensively in a professional setting...",
                        'task': "I've been learning it through [courses/projects/self-study]...",
                        'action': f"I recently completed [specific example using {skill}]...",
                        'result': "I'm confident I can quickly apply this knowledge to your projects"
                    },
                    'tips': [
                        "Be honest about your experience level",
                        "Mention any related skills or learning efforts",
                        "Show enthusiasm for learning new technologies",
                        "Provide concrete examples of quick learning in the past"
                    ]
                })
    
    return questions