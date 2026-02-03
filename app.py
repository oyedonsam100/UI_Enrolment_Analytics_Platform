import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import joblib
import pickle
from pathlib import Path
import requests
from io import BytesIO
from scipy.optimize import differential_evolution

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="UI Analytics Platform",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a5490;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card-secondary {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        height: 100%;
    }
    .info-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-left: 4px solid #4caf50;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .insight-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .hierarchy-card {
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'eda_data' not in st.session_state:
    st.session_state.eda_data = None

# --------------------------------------------------
# HIERARCHICAL STRUCTURE DATA
# --------------------------------------------------
UNIVERSITY_STRUCTURE = {
    'AGRICULTURE': ['Agricultural Economics', 'Agricultural Extension & Rural Development', 'Agronomy', 'Animal Science', 'Crop Protection & Environmental Biology', 'Soil Resources Management'],
    'ARTS': ['Arabic & Islamic Studies', 'Classics', 'Communication & Language Arts', 'English', 'European Studies', 'History', 'Linguistics & African Languages', 'Music', 'Philosophy', 'Theatre Arts'],
    'BASIC MEDICAL SCIENCES': ['Anatomy', 'Biochemistry', 'Physiology'],
    'CLINICAL SCIENCES': ['Anaesthesia', 'Chemical Pathology', 'Community Medicine', 'Haematology', 'Medical Microbiology & Parasitology', 'Medicine', 'Morbid Anatomy', 'Obstetrics & Gynaecology', 'Ophthalmology', 'Otorhinolaryngology', 'Paediatrics', 'Psychiatry', 'Radiation Medicine', 'Surgery'],
    'DENTISTRY': ['Child Dental Health', 'Oral & Maxillofacial Surgery', 'Oral Pathology & Biology', 'Periodontology & Community Dentistry', 'Preventive Dentistry', 'Restorative Dentistry'],
    'EDUCATION': ['Adult Education', 'Arts & Social Sciences Education', 'Educational Management', 'Guidance & Counselling', 'Library, Archival & Information Studies', 'Science & Technology Education', 'Teacher Education'],
    'ENVIRONMENTAL DESIGN & MANAGEMENT': ['Architecture', 'Estate Management', 'Quantity Surveying', 'Urban & Regional Planning'],
    'LAW': ['Commercial & Industrial Law', 'International Law', 'Private & Property Law', 'Public Law'],
    'PHARMACY': ['Clinical Pharmacy & Pharmacy Administration', 'Pharmaceutical Chemistry', 'Pharmaceutics', 'Pharmacognosy', 'Pharmacology & Therapeutics'],
    'PUBLIC HEALTH': ['Epidemiology & Medical Statistics', 'Environmental Health Sciences', 'Health Policy & Management', 'Health Promotion & Education', 'Human Nutrition'],
    'RENEWABLE NATURAL RESOURCES': ['Aquaculture & Fisheries Management', 'Forest Resources Management', 'Social & Environmental Forestry', 'Wildlife & Ecotourism Management'],
    'SCIENCE': ['Botany', 'Chemistry', 'Computer Science', 'Geography', 'Geology', 'Mathematics', 'Microbiology', 'Physics', 'Statistics', 'Zoology'],
    'SOCIAL SCIENCES': ['Anthropology', 'Economics', 'Geography', 'Political Science', 'Psychology', 'Sociology'],
    'TECHNOLOGY': ['Agricultural & Environmental Engineering', 'Civil Engineering', 'Electrical & Electronics Engineering', 'Food Technology', 'Industrial & Production Engineering', 'Mechanical Engineering', 'Petroleum Engineering', 'Wood Products Engineering'],
    'VETERINARY MEDICINE': ['Veterinary Anatomy', 'Veterinary Medicine', 'Veterinary Microbiology & Parasitology', 'Veterinary Pathology', 'Veterinary Pharmacology & Toxicology', 'Veterinary Physiology & Biochemistry', 'Veterinary Public Health & Preventive Medicine', 'Veterinary Surgery & Reproduction']
}

def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("Incorrect password")
        return False
    else:
        return True

if not check_password():
    st.stop()

# --------------------------------------------------
# OPTIMIZATION FUNCTIONS (ADD BEFORE load_models)
# --------------------------------------------------

def prepare_model_input(year, enrollment, budget, male_staff, female_staff, resource_features):
    """Prepare input features for the resource allocation model."""
    total_staff = male_staff + female_staff
    budget_per_student = budget / enrollment if enrollment > 0 else 0
    student_staff_ratio = enrollment / total_staff if total_staff > 0 else 0
    log_budget = np.log1p(budget)
    
    input_data = {
        'year': year,
        'total_enrollment': enrollment,
        'annual_budget_dept(₦)': budget,
        'fac_staff_count_male': male_staff,
        'fac_staff_count_female': female_staff,
        'total_staff': total_staff,
        'budget_per_student': budget_per_student,
        'student_staff_ratio': student_staff_ratio,
        'log_budget': log_budget,
        'hostel_allocation_probability': 40,
        'strike_duration_months': 1,
        'gdp_growth_percentage': 2.5,
        'unemployment_rate_percentage': 20.0
    }
    
    df_input = pd.DataFrame([input_data])
    
    for feature in resource_features:
        if feature not in df_input.columns:
            df_input[feature] = 0
    
    return df_input[resource_features]


def optimize_resource_allocation(current_enrollment, projected_enrollment, current_staff,
                                current_male_staff, current_female_staff, current_budget,
                                budget_limit, target_grad_rate, max_ratio, current_grad_rate,
                                faculty, resource_model, resource_features):
    """Optimise resource allocation using multi-objective optimisation - 1 Year Plan."""
    
    def objective_function(x):
        male_hire = x[0]
        female_hire = x[1]
    
        total_new_staff = male_hire + female_hire
        final_total_staff = current_staff + total_new_staff
        final_male_staff = current_male_staff + male_hire
        final_female_staff = current_female_staff + female_hire

        if final_total_staff <= 0:
            return 1e8

        final_ratio = projected_enrollment / final_total_staff

        # Calculate total cost - more realistic with onboarding costs
        avg_salary_per_staff = 1.8e6
        total_salary_cost = total_new_staff * avg_salary_per_staff
    
        # Higher overhead for new staff
        if total_new_staff <= 10:
            overhead_multiplier = 1.40
        elif total_new_staff <= 20:
            overhead_multiplier = 1.50
        else:
            overhead_multiplier = 1.65
    
        total_cost = total_salary_cost * overhead_multiplier

        # GRADUATION RATE PREDICTION
        current_ratio = current_enrollment / current_staff
    
        ratio_improvement = current_ratio - final_ratio
        ratio_effect = max(0, ratio_improvement) * 1.8
    
        budget_per_student = (current_budget + total_cost) / projected_enrollment
        current_budget_per_student = current_budget / current_enrollment
        budget_improvement = budget_per_student - current_budget_per_student
        budget_effect = max(0, budget_improvement / 30000) * 1.5
    
        staff_growth = (total_new_staff / current_staff) * 100 if current_staff > 0 else 0
        staff_effect = min(staff_growth * 0.08, 4)

        # Diminishing returns - harder to improve when already good
        if current_grad_rate > 75:
            improvement_factor = 0.7  # 30% reduction if already good
        else:
            improvement_factor = 1.0
    
        predicted_grad_rate = current_grad_rate + ratio_effect + budget_effect + staff_effect
        predicted_grad_rate = min(predicted_grad_rate, 95.0)

        # FITNESS CALCULATION
        cost_normalized = total_cost / max(budget_limit, 1e7)
    
        if predicted_grad_rate < target_grad_rate:
            grad_penalty = ((target_grad_rate - predicted_grad_rate) ** 2) * 15
        else:
            grad_penalty = (predicted_grad_rate - target_grad_rate) * 0.3
    
        if final_ratio > max_ratio:
            ratio_penalty = ((final_ratio - max_ratio) ** 3) * 150
        else:
            ratio_penalty = max(0, final_ratio - max_ratio + 2) * 5
    
        female_pct = (final_female_staff / final_total_staff * 100) if final_total_staff > 0 else 0
        gender_penalty = abs(female_pct - 40) * 0.5
    
        if total_cost > budget_limit:
            budget_penalty = ((total_cost - budget_limit) / budget_limit) * 200
        else:
            budget_penalty = 0
    
        # NEW: Strong hiring size penalty (prefer smaller hiring)
        hiring_size_penalty = 0
        if total_new_staff > 15:
            hiring_size_penalty = ((total_new_staff - 15) ** 3) * 20  # Cubic penalty
        elif total_new_staff > 10:
            hiring_size_penalty = ((total_new_staff - 10) ** 2) * 10  # Quadratic penalty
        elif total_new_staff > 5:
            hiring_size_penalty = (total_new_staff - 5) * 3  # Linear penalty

        # Also add practical constraint penalty
        if total_new_staff > 12:
            hiring_size_penalty += 100  # Very large penalty for >12

        fitness = (
            cost_normalized * 0.35 +
            grad_penalty * 0.30 +
            ratio_penalty * 0.20 +
            gender_penalty * 0.05 +
            budget_penalty * 0.05 +
            hiring_size_penalty * 0.05
        )

        return fitness
    
    # SIMPLIFIED BOUNDS - Only 2 parameters (male and female hires)
    bounds = [
        (0, 10),  # Male hires
        (0, 10)   # Female hires
    ]
    
    # Run optimization
    result = differential_evolution(
        objective_function,
        bounds,
        strategy='best1bin',
        maxiter=30,
        popsize=8,
        tol=0.05,
        seed=42
    )
    
    # Extract results
    optimal_x = result.x
    male_hire = int(optimal_x[0])
    female_hire = int(optimal_x[1])
    total_new_staff = male_hire + female_hire
    
    final_total_staff = current_staff + total_new_staff
    final_male_staff = current_male_staff + male_hire
    final_female_staff = current_female_staff + female_hire
    final_ratio = projected_enrollment / final_total_staff
    
    # Calculate costs
    avg_salary = 1.8e6
    total_salary_cost = total_new_staff * avg_salary
    
    # Fixed budget allocation (not optimized)
    salary_budget = total_salary_cost
    infra_budget = total_salary_cost * 0.15      # 15% of salaries for infrastructure
    teaching_budget = total_salary_cost * 0.10   # 10% for teaching materials
    research_budget = total_salary_cost * 0.05   # 5% for research support
    
    total_additional_budget = salary_budget + infra_budget + teaching_budget + research_budget
    total_investment = total_additional_budget
    
    optimal_annual_budget = current_budget + total_additional_budget
    optimal_budget_per_student = optimal_annual_budget / projected_enrollment
    
    # Calculate achieved graduation rate
    current_ratio = current_enrollment / current_staff
    final_ratio_calc = projected_enrollment / final_total_staff

    ratio_improvement = max(0, current_ratio - final_ratio_calc)
    ratio_effect = ratio_improvement * 1.8

    current_budget_per_student = current_budget / current_enrollment
    budget_improvement = optimal_budget_per_student - current_budget_per_student
    budget_effect = max(0, budget_improvement / 30000) * 1.5

    staff_growth = (total_new_staff / current_staff) * 100 if current_staff > 0 else 0
    staff_effect = min(staff_growth * 0.08, 4)

    # Diminishing returns
    if current_grad_rate > 75:
        improvement_factor = 0.7
    else:
        improvement_factor = 1.0

    achieved_grad_rate = current_grad_rate + ratio_effect + budget_effect + staff_effect
    achieved_grad_rate = min(achieved_grad_rate, 92.0)
    
    grad_rate_improvement = achieved_grad_rate - current_grad_rate
    
    # Create yearly plan (1 year only)
    yearly_plan = [{
        'Year': 2026,
        'New Staff': total_new_staff,
        'Male': male_hire,
        'Female': female_hire,
        'Cumulative Staff': final_total_staff,
        'Annual Cost (₦M)': round(total_salary_cost / 1e6, 1),
        'Student-Staff Ratio': round(final_ratio, 1),
        'Grad Rate (%)': round(achieved_grad_rate, 1)
    }]
    
    # Calculate graduate impact
    baseline_graduates = projected_enrollment * (current_grad_rate / 100)
    optimized_graduates = projected_enrollment * (achieved_grad_rate / 100)
    additional_graduates = max(0, optimized_graduates - baseline_graduates)
    
    # Immediate actions
    immediate_actions = [
        f"Recruit {male_hire} male and {female_hire} female lecturers",
        f"Allocate ₦{salary_budget/1e6:.1f}M for new staff salaries",
        "Establish recruitment committee and begin hiring process",
        "Complete faculty onboarding and orientation programs"
    ]
    
    # Long-term strategies
    long_term_strategies = [
        f"Complete hiring of all {total_new_staff} staff in Year 1",
        f"Target {round(final_female_staff/final_total_staff*100, 1)}% female faculty representation",
        "Implement faculty development and orientation programs",
        "Establish quality assurance monitoring system",
        "Plan for future growth based on enrollment trends"
    ]
    
    # Constraint checking
    constraints_met = []
    constraints_violated = []
    
    if final_ratio <= max_ratio:
        constraints_met.append(f"✅ Student-staff ratio ({final_ratio:.1f}:1) within limit ({max_ratio}:1)")
    else:
        constraints_violated.append(f"⚠️ Student-staff ratio ({final_ratio:.1f}:1) exceeds limit ({max_ratio}:1)")
    
    female_pct = final_female_staff / final_total_staff * 100
    if 30 <= female_pct <= 50:
        constraints_met.append(f"✅ Gender balance ({female_pct:.1f}% female) within target (30-50%)")
    else:
        constraints_violated.append(f"⚠️ Gender balance ({female_pct:.1f}% female) outside target range")
    
    if total_investment <= budget_limit * 1.1:
        constraints_met.append(f"✅ Total investment (₦{total_investment/1e6:.1f}M) within budget")
    else:
        constraints_violated.append(f"⚠️ Investment (₦{total_investment/1e6:.1f}M) exceeds budget limit")
    
    if achieved_grad_rate >= target_grad_rate - 2:
        constraints_met.append(f"✅ Graduation rate ({achieved_grad_rate:.1f}%) near target ({target_grad_rate}%)")
    
    return {
        'total_new_staff': total_new_staff,
        'male_hires': male_hire,
        'female_hires': female_hire,
        'optimal_budget': optimal_annual_budget,
        'budget_increase': total_additional_budget,
        'achieved_grad_rate': achieved_grad_rate,
        'grad_rate_improvement': grad_rate_improvement,
        'final_ratio': final_ratio,
        'ratio_change': final_ratio - (current_enrollment / current_staff),
        'optimal_budget_per_student': optimal_budget_per_student,
        'yearly_plan': yearly_plan,
        'total_investment': total_investment,
        'additional_graduates': additional_graduates,
        'immediate_actions': immediate_actions,
        'long_term_strategies': long_term_strategies,
        'constraints_met': constraints_met,
        'constraints_violated': constraints_violated,
        'optimisation_success': result.success,
        'optimisation_message': result.message
    }

def generate_optimization_report(optimal_plan, faculty, year):
    """Generate a downloadable CSV report."""
    import io
    
    output = io.StringIO()
    
    output.write(f"OPTIMAL RESOURCE ALLOCATION PLAN\n")
    output.write(f"Faculty: {faculty}\n")
    output.write(f"Target Year: {year}\n")
    output.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output.write("\n")
    
    output.write("SUMMARY METRICS\n")
    output.write(f"Total New Staff (Year 1),{optimal_plan['total_new_staff']}\n")
    output.write(f"Male Hires,{optimal_plan['male_hires']}\n")
    output.write(f"Female Hires,{optimal_plan['female_hires']}\n")
    output.write(f"Optimal Annual Budget (₦),{optimal_plan['optimal_budget']:,.0f}\n")
    output.write(f"Additional Budget Needed (₦),{optimal_plan['budget_increase']:,.0f}\n")
    output.write(f"Achieved Graduation Rate (%),{optimal_plan['achieved_grad_rate']:.1f}\n")
    output.write(f"Final Student-Staff Ratio,{optimal_plan['final_ratio']:.1f}\n")
    output.write(f"Total Investment (₦),{optimal_plan['total_investment']:,.0f}\n")
    output.write(f"Additional Graduates,{optimal_plan['additional_graduates']:.0f}\n")
    output.write("\n")
    
    output.write("YEAR-BY-YEAR IMPLEMENTATION PLAN\n")
    yearly_df = pd.DataFrame(optimal_plan['yearly_plan'])
    yearly_df.to_csv(output, index=False)
    output.write("\n")
    
    output.write("ADDITIONAL BUDGET BREAKDOWN (₦ Millions)\n")
    output.write("Salaries,Other Costs (Buffer)\n")
    output.write("\n")
    
    output.write("IMMEDIATE ACTIONS (YEAR 1)\n")
    for action in optimal_plan['immediate_actions']:
        output.write(f"{action}\n")
    output.write("\n")
    
    output.write("IMPLEMENTATION STRATEGY\n")
    for strategy in optimal_plan['long_term_strategies']:
        output.write(f"{strategy}\n")
    
    return output.getvalue()

# --------------------------------------------------
# LOAD MODELS FROM LOCAL FILES
# --------------------------------------------------
@st.cache_resource
def load_models():
    """Load models from local 'models' directory."""
    
    models_dir = Path(__file__).parent / "models"
    
    if not models_dir.exists():
        st.error(f"❌ Models directory not found at: {models_dir}")
        st.info("""
        **Setup Instructions:**
        1. Create a 'models' folder in the same directory as your app.py file
        2. Download your model files from HuggingFace
        3. Place them in the 'models' folder with these exact names:
           - ui_enrollment_features.pkl
           - ui_enrollment_prediction_model.pkl
           - ui_resource_allocation_model.pkl
           - ui_resource_features.pkl
           - ui_system_metadata.pkl
        """)
        st.stop()
    
    model_files = {
        "enroll_features": models_dir / "ui_enrollment_features.pkl",
        "enroll_model": models_dir / "ui_enrollment_prediction_model.pkl",
        "resource_model": models_dir / "ui_resource_allocation_model.pkl",
        "resource_features": models_dir / "ui_resource_features.pkl",
        "metadata": models_dir / "ui_system_metadata.pkl",
    }
    
    missing_files = [name for name, path in model_files.items() if not path.exists()]
    if missing_files:
        st.error(f"❌ Missing model files: {', '.join(missing_files)}")
        st.info(f"""
        **Required files in {models_dir}:**
        - ui_enrollment_features.pkl
        - ui_enrollment_prediction_model.pkl
        - ui_resource_allocation_model.pkl
        - ui_resource_features.pkl
        - ui_system_metadata.pkl
        
        **Missing:** {', '.join(missing_files)}
        """)
        st.stop()
    
    try:
        with st.spinner("Loading models..."):
            enroll_model = joblib.load(model_files["enroll_model"])
            resource_model = joblib.load(model_files["resource_model"])
            
            with open(model_files["enroll_features"], 'rb') as f:
                enroll_features = pickle.load(f)
            
            with open(model_files["resource_features"], 'rb') as f:
                resource_features = pickle.load(f)
            
            with open(model_files["metadata"], 'rb') as f:
                metadata = pickle.load(f)
        
        return enroll_model, enroll_features, resource_model, resource_features, metadata
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("""
        **Troubleshooting:**
        1. Ensure all .pkl files are properly downloaded
        2. Check that files are not corrupted
        3. Verify you have the correct versions of joblib and pickle
        4. Make sure the files were saved with compatible Python versions
        """)
        st.stop()

# --------------------------------------------------
# NAVIGATION SIDEBAR
# --------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        if st.button("🏠 Home", use_container_width=True, type="primary" if st.session_state.page == 'landing' else "secondary"):
            st.session_state.page = 'landing'
            st.rerun()
        
        if st.button("📊 EDA Dashboard", use_container_width=True, type="primary" if st.session_state.page == 'eda' else "secondary"):
            st.session_state.page = 'eda'
            st.rerun()
        
        if st.button("🎯 Prediction Tool", use_container_width=True, type="primary" if st.session_state.page == 'prediction' else "secondary"):
            st.session_state.page = 'prediction'
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("**University of Ibadan**\n\nEnrolment Prediction and Resource Optimisation Platform")
        
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        st.metric("Faculties", len(UNIVERSITY_STRUCTURE))
        total_depts = sum(len(depts) for depts in UNIVERSITY_STRUCTURE.values())
        st.metric("Departments", total_depts)

# --------------------------------------------------
# LOAD MODELS FROM LOCAL FILES
# --------------------------------------------------
@st.cache_resource
def load_models():
    """
    Load models from local 'models' directory.
    
    Directory structure should be:
    your_app_folder/
    ├── app.py (this file)
    └── models/
        ├── ui_enrollment_features.pkl
        ├── ui_enrollment_prediction_model.pkl
        ├── ui_resource_allocation_model.pkl
        ├── ui_resource_features.pkl
        └── ui_system_metadata.pkl
    """
    
    # Define the models directory path
    models_dir = Path(__file__).parent / "models"
    
    # Check if models directory exists
    if not models_dir.exists():
        st.error(f"❌ Models directory not found at: {models_dir}")
        st.info("""
        **Setup Instructions:**
        1. Create a 'models' folder in the same directory as your app.py file
        2. Download your model files from HuggingFace
        3. Place them in the 'models' folder with these exact names:
           - ui_enrollment_features.pkl
           - ui_enrollment_prediction_model.pkl
           - ui_resource_allocation_model.pkl
           - ui_resource_features.pkl
           - ui_system_metadata.pkl
        """)
        st.stop()
    
    # Define file paths
    model_files = {
        "enroll_features": models_dir / "ui_enrollment_features.pkl",
        "enroll_model": models_dir / "ui_enrollment_prediction_model.pkl",
        "resource_model": models_dir / "ui_resource_allocation_model.pkl",
        "resource_features": models_dir / "ui_resource_features.pkl",
        "metadata": models_dir / "ui_system_metadata.pkl",
    }
    
    # Check if all required files exist
    missing_files = [name for name, path in model_files.items() if not path.exists()]
    if missing_files:
        st.error(f"❌ Missing model files: {', '.join(missing_files)}")
        st.info(f"""
        **Required files in {models_dir}:**
        - ui_enrollment_features.pkl
        - ui_enrollment_prediction_model.pkl
        - ui_resource_allocation_model.pkl
        - ui_resource_features.pkl
        - ui_system_metadata.pkl
        
        **Missing:** {', '.join(missing_files)}
        """)
        st.stop()
    
    try:
        # Load models using joblib for .pkl files
        with st.spinner("Loading models..."):
            enroll_model = joblib.load(model_files["enroll_model"])
            resource_model = joblib.load(model_files["resource_model"])
            
            # Load features and metadata using pickle
            with open(model_files["enroll_features"], 'rb') as f:
                enroll_features = pickle.load(f)
            
            with open(model_files["resource_features"], 'rb') as f:
                resource_features = pickle.load(f)
            
            with open(model_files["metadata"], 'rb') as f:
                metadata = pickle.load(f)
        
        st.success("✅ All models loaded successfully!")
        return enroll_model, enroll_features, resource_model, resource_features, metadata
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("""
        **Troubleshooting:**
        1. Ensure all .pkl files are properly downloaded
        2. Check that files are not corrupted
        3. Verify you have the correct versions of joblib and pickle
        4. Make sure the files were saved with compatible Python versions
        """)
        st.stop()

# --------------------------------------------------
# NAVIGATION SIDEBAR
# --------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        if st.button("🏠 Home", use_container_width=True, type="primary" if st.session_state.page == 'landing' else "secondary"):
            st.session_state.page = 'landing'
            st.rerun()
        
        if st.button("📊 EDA Dashboard", use_container_width=True, type="primary" if st.session_state.page == 'eda' else "secondary"):
            st.session_state.page = 'eda'
            st.rerun()
        
        if st.button("🎯 Prediction Tool", use_container_width=True, type="primary" if st.session_state.page == 'prediction' else "secondary"):
            st.session_state.page = 'prediction'
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("**University of Ibadan**\n\nEnrolment Prediction and Resource Optimisation Platform")
        
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        st.metric("Faculties", len(UNIVERSITY_STRUCTURE))
        total_depts = sum(len(depts) for depts in UNIVERSITY_STRUCTURE.values())
        st.metric("Departments", total_depts)

# --------------------------------------------------
# LANDING PAGE
# --------------------------------------------------
def landing_page():
    st.markdown('<div class="main-header">🎓 University of Ibadan</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enrolment Trend Prediction and Resource Optimisation Analytics Platform</div>', unsafe_allow_html=True)
    
    st.warning("⚠️ **Disclaimer:** This system provides indicative predictions for planning and decision support. Final outcomes depend on policy decisions, admissions quotas, and external factors.")
    
    st.markdown("## 🚀 Platform Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 EDA Dashboard</h3>
            <p>Explore enrolment data with interactive visualisations, statistical analysis, and trend identification across faculties and departments.</p>
            <br>
            <p><strong>Features:</strong></p>
            <ul>
                <li>Data overview and statistics</li>
                <li>Distribution analysis</li>
                <li>Correlation matrices</li>
                <li>Trend visualisation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Enrolment Prediction</h3>
            <p>AI-powered enrolment growth forecasting using machine learning models trained on historical data and economic indicators.</p>
            <br>
            <p><strong>Predictions:</strong></p>
            <ul>
                <li>1-year enrolment projections</li>
                <li>Growth rate analysis</li>
                <li>Uncertainty ranges</li>
                <li>Faculty-level forecasts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>💰 Resource Planning</h3>
            <p>Optimise resource allocation with graduation rate predictions and scenario simulations for evidence-based planning.</p>
            <br>
            <p><strong>Analysis:</strong></p>
            <ul>
                <li>Graduation rate forecasts</li>
                <li>Resource impact assessment</li>
                <li>Scenario simulation</li>
                <li>Budget optimisation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("📖 Getting Started")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### How to Use This Platform
        
        **1. Exploratory Data Analysis (EDA)**
        - Upload enrolment datasets in CSV format
        - Explore historical trends and patterns
        - Analyse distributions and correlations
        - Identify key factors affecting enrolment
        
        **2. Enrolment Prediction**
        - Select faculty from hierarchical structure
        - Input current parameters and economic indicators
        - Get AI-powered enrolment growth predictions
        - View 1-year forecasts with uncertainty ranges
        
        **3. Resource Planning**
        - Analyse resource allocation efficiency
        - Predict graduation rates based on resources
        - Simulate different resource scenarios
        - Optimise budget and staffing decisions
        """)
    
    with col2:
        st.markdown("""
        <div class="info-box" style="background-color: #000000; color: #ffffff; padding: 1.5rem; border-radius: 10px; border: 1px solid #333333;">
        <h4>📁 Data Requirements</h4>
            <p><strong>Format:</strong> CSV files</p>
            <p><strong>Key Fields:</strong></p>
            <ul style='margin-left: 1rem;'>
                <li>Year</li>
                <li>Enrolment numbers</li>
                <li>Budget data</li>
                <li>Staff counts</li>
                <li>E.T.C</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 🎯 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Explore Data →", use_container_width=True, type="primary"):
            st.session_state.page = 'eda'
            st.rerun()
    
    with col2:
        if st.button("🎯 Make Predictions →", use_container_width=True, type="primary"):
            st.session_state.page = 'prediction'
            st.rerun()
    
    with col3:
        sample_data = "year,enrollment,budget,staff\n2024,4000,50000000,1200\n2023,3800,48000000,1150"
        st.download_button(
            label="📥 Sample Data",
            data=sample_data,
            file_name="sample_data.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; padding: 2rem 0;'>
        <p><strong>© University of Ibadan 2026</strong></p>
        <p>Developed as part of M.Info.Sci research on Machine Learning for Enrolment Prediction and Resource Optimisation</p>
        <p style='margin-top: 1rem; font-size: 0.8rem; color: #999;'>All Rights Reserved © 2026</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# EDA DASHBOARD
# --------------------------------------------------
def eda_dashboard():
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyse Enrolment Trends and Patterns</div>', unsafe_allow_html=True)
   
    # Initialize session state for upload method if not exists
    if 'upload_method' not in st.session_state:
        st.session_state.upload_method = "Upload CSV File"
   
    # Multiple upload options
    upload_method = st.radio(
        "Choose data input method:",
        ["Upload CSV File", "Use Sample Data", "Paste CSV Data"],
        horizontal=True,
        key='upload_method_radio'
    )
   
    df = None
   
    if upload_method == "Upload CSV File":
        # Clear instructions
        st.markdown("**📤 Upload your CSV file below:**")
       
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            accept_multiple_files=False,
            help="Upload a CSV file containing enrollment data"
        )
       
        if uploaded_file is not None:
            try:
                # Show file details
                st.write(f"**File name:** {uploaded_file.name}")
                file_size_mb = uploaded_file.size / (1024 * 1024)
                st.write(f"**File size:** {file_size_mb:.2f} MB")
               
                if file_size_mb > 200:
                    st.error("❌ File too large. Maximum size is 200MB")
                    st.stop()
               
                # Progress indicator
                with st.spinner('Loading file...'):
                    file_bytes = uploaded_file.read()
                    uploaded_file.seek(0) # Reset pointer
                   
                    # Try multiple encoding options
                    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                   
                    for encoding in encodings_to_try:
                        try:
                            df = pd.read_csv(uploaded_file, encoding=encoding, on_bad_lines='skip')
                            st.session_state.eda_data = df
                            st.success(f"✅ File loaded successfully!")
                            break
                        except Exception as enc_error:
                            uploaded_file.seek(0) # Reset pointer for next attempt
                            if encoding == encodings_to_try[-1]: # Last encoding failed
                                raise Exception(f"Could not read file with any encoding. Last error: {str(enc_error)}")
                            continue
                   
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.markdown("### 💡 Troubleshooting Tips:")
                st.markdown("""
                1. **Check file format**: Make sure it's a valid CSV file
                2. **Re-save your file**:
                   - Open in Excel
                   - Go to File → Save As
                   - Choose "CSV UTF-8 (Comma delimited) (*.csv)"
                   - Save and try uploading again
                3. **Try alternative methods**: Use "Use Sample Data" or "Paste CSV Data" options above
                4. **File size**: Ensure file is under 200MB
                5. **Remove special characters**: Check if your data has unusual characters
                """)
               
                # Show option to try paste method
                st.warning("⚠️ **Alternative:** Try the 'Paste CSV Data' method instead - it often works when upload fails!")
   
    elif upload_method == "Use Sample Data":
        st.info("📊 Loading sample enrollment data...")
       
        # Create comprehensive sample data with your real columns
        sample_data = {
            'YEAR': [2019, 2020, 2021, 2022, 2023, 2024],
            'GENDER': ['M', 'F', 'M', 'F', 'M', 'F'],
            'FACULTY': ['SCIENCE', 'SCIENCE', 'ARTS', 'SCIENCE', 'LAW', 'MEDICINE'],
            'DEPARTMENT': ['Computer Science', 'Physics', 'English', 'Mathematics', 'Law', 'Medicine'],
            'MODE_OF_ENTRY': ['UTME', 'Direct Entry', 'UTME', 'UTME', 'Direct Entry', 'UTME'],
            'ENROLED': ['YES', 'YES', 'YES', 'YES', 'YES', 'YES'],
            'ANNUAL_BUDGET_DEPT(₦)': [42000000, 45000000, 46000000, 48000000, 49000000, 50000000],
            'FAC_STAFF_COUNT_MALE': [600, 650, 670, 680, 690, 700],
            'FAC_STAFF_COUNT_FEMALE': [400, 450, 470, 480, 490, 500],
            'HOSTEL_ALLOCATION_PROBABILITY': [35, 37, 38, 40, 40, 40],
            'STRIKE_DURATION_MONTHS': [2, 0, 1, 0, 1, 1],
            'GDP_GROWTH_PERCENTAGE': [2.2, -1.8, 3.4, 3.2, 2.5, 2.8],
            'UNEMPLOYMENT_RATE_PERCENTAGE': [23.1, 27.1, 22.5, 21.0, 20.5, 20.0],
            'DEPT_POST_UTME_CUT_OFF': [55, 56, 57, 58, 59, 60]
        }
        df = pd.DataFrame(sample_data)
        # Add derived columns for sample
        df['TOTAL_STAFF'] = df['FAC_STAFF_COUNT_MALE'] + df['FAC_STAFF_COUNT_FEMALE']
        df['TOTAL_ENROLLED'] = 4000  # Fixed sample total
        df['STUDENTS_PER_STAFF'] = df['TOTAL_ENROLLED'] / df['TOTAL_STAFF']
        df['BUDGET_PER_STUDENT'] = df['ANNUAL_BUDGET_DEPT(₦)'] / df['TOTAL_ENROLLED']
        st.session_state.eda_data = df
        st.success("✅ Sample data loaded successfully!")
   
    elif upload_method == "Paste CSV Data":
        st.markdown("### 📋 Paste Your CSV Data")
        st.info("""
        **Instructions:**
        1. Open your CSV file in Excel, Notepad, or any text editor
        2. Select all the content (Ctrl+A or Cmd+A)
        3. Copy it (Ctrl+C or Cmd+C)
        4. Paste it in the box below
        5. Click 'Load Data'
        """)
       
        csv_text = st.text_area(
            "Paste your CSV data here (include the header row):",
            height=250,
            placeholder="YEAR,GENDER,FACULTY,...,DEPT_POST_UTME_CUT_OFF\n2024,M,SCIENCE,...,55",
            key='csv_text_input'
        )
       
        if st.button("📊 Load Data", type="primary", use_container_width=True):
            if csv_text.strip():
                try:
                    from io import StringIO
                    df = pd.read_csv(StringIO(csv_text))
                    st.session_state.eda_data = df
                    st.success(f"✅ Data loaded successfully! {df.shape[0]} rows × {df.shape[1]} columns")
                except Exception as e:
                    st.error(f"❌ Error parsing CSV: {str(e)}")
                    st.info("""
                    **Common issues:**
                    - Make sure the first line contains column headers
                    - Ensure values are separated by commas
                    - Check that there are no extra spaces or special characters
                    - Each row should have the same number of columns
                    """)
            else:
                st.warning("⚠️ Please paste your CSV data in the text area above")
   
    # Display data if available
    if st.session_state.eda_data is not None:
        df = st.session_state.eda_data
       
        st.markdown("---")
        st.success(f"✅ **Dataset loaded:** {df.shape[0]} rows × {df.shape[1]} columns")
       
        # Add a clear data button
        if st.button("🗑️ Clear Data", help="Remove loaded data and start over"):
            st.session_state.eda_data = None
            st.rerun()
       
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Overview", "📊 Distributions", "🔗 Correlations", "📈 Trends", "📉 Summary"])
       
        with tab1:
            st.subheader("Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Records", f"{df.shape[0]:,}")
            with col2:
                st.metric("Features", df.shape[1])
            with col3:
                st.metric("Numeric", df.select_dtypes(include=[np.number]).shape[1])
            with col4:
                st.metric("Missing", df.isnull().sum().sum())
           
            st.markdown("### 📄 Data Preview (First 20 rows)")
            st.dataframe(df.head(20), use_container_width=True)
           
            st.markdown("### 📊 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values,
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
       
        with tab2:
            st.subheader("Distribution Analysis")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
           
            if numeric_cols:
                selected_col = st.selectbox("Select variable for distribution analysis", numeric_cols)
               
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(
                        df,
                        x=selected_col,
                        nbins=30,
                        title=f'Distribution of {selected_col}',
                        color_discrete_sequence=['#667eea']
                    )
                    st.plotly_chart(fig, use_container_width=True)
               
                with col2:
                    fig = px.box(
                        df,
                        y=selected_col,
                        title=f'Box Plot of {selected_col}',
                        color_discrete_sequence=['#764ba2']
                    )
                    st.plotly_chart(fig, use_container_width=True)
               
                # Statistics
                st.markdown(f"### 📊 Statistics for {selected_col}")
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
               
                with stat_col1:
                    st.metric("Mean", f"{df[selected_col].mean():.2f}")
                with stat_col2:
                    st.metric("Median", f"{df[selected_col].median():.2f}")
                with stat_col3:
                    st.metric("Std Dev", f"{df[selected_col].std():.2f}")
                with stat_col4:
                    st.metric("Range", f"{df[selected_col].max() - df[selected_col].min():.2f}")
            else:
                st.info("No numeric columns found in the dataset")
       
        with tab3:
            st.subheader("Correlation Analysis")
            numeric_df = df.select_dtypes(include=[np.number])
           
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
               
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect='auto',
                    color_continuous_scale='RdBu_r',
                    title='Correlation Heatmap',
                    labels=dict(color="Correlation")
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
               
                # Top correlations
                st.markdown("### 🔝 Strongest Correlations")
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
               
                corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False).head(10)
                st.dataframe(corr_df, use_container_width=True, hide_index=True)
            else:
                st.info("Need at least 2 numeric columns for correlation analysis")
       
        with tab4:
            st.subheader("Trend Analysis")
            
            # Clean percentage columns
            percentage_columns = [
                'GDP_GROWTH_PERCENTAGE', 'UNEMPLOYMENT_RATE_PERCENTAGE',
                'HOSTEL_ALLOCATION_PROBABILITY'
            ]
           
            for col in percentage_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace('%', '', regex=False)
                    df[col] = df[col].str.replace(',', '.', regex=False)
                    df[col] = df[col].str.strip()
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].median())
           
            st.info("Percentage columns cleaned and converted to numeric")
            
            # Derived columns
            if 'FAC_STAFF_COUNT_MALE' in df.columns and 'FAC_STAFF_COUNT_FEMALE' in df.columns:
                df['TOTAL_STAFF'] = df['FAC_STAFF_COUNT_MALE'] + df['FAC_STAFF_COUNT_FEMALE']
            if 'TOTAL_ENROLLED' not in df.columns and 'ENROLED' in df.columns:
                df['TOTAL_ENROLLED'] = df.groupby('YEAR')['ENROLED'].transform(lambda x: (x == 'YES').sum())
            if 'ANNUAL_BUDGET_DEPT(₦)' in df.columns and 'TOTAL_ENROLLED' in df.columns:
                df['BUDGET_PER_STUDENT'] = df['ANNUAL_BUDGET_DEPT(₦)'] / df['TOTAL_ENROLLED'].replace(0, 1)
            if 'TOTAL_STAFF' in df.columns and 'TOTAL_ENROLLED' in df.columns:
                df['STUDENTS_PER_STAFF'] = df['TOTAL_ENROLLED'] / df['TOTAL_STAFF'].replace(0, 1)
            
            if 'YEAR' not in df.columns:
                st.error("No 'YEAR' column found.")
                st.stop()
            
            df_trend = df.sort_values('YEAR')
            
            # Helper for line charts
            def plot_line(y_col, title, y_label=""):
                if y_col in df_trend.columns:
                    fig = px.line(df_trend, x='YEAR', y=y_col, title=title, markers=True)
                    fig.update_layout(yaxis_title=y_label, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"Missing '{y_col}' for '{title}'")
            
            # Helper for scatter + trend line
            def plot_scatter(x_col, title, x_label=""):
                if x_col in df.columns and 'TOTAL_ENROLLED' in df.columns:
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y='TOTAL_ENROLLED',
                        title=title,
                        labels={x_col: x_label, 'TOTAL_ENROLLED': 'Total Enrolment'},
                        trendline="ols",
                        trendline_color_override="red",
                        color_discrete_sequence=['#636EFA']
                    )
                    # Completely disable trendline labels and hover
                    for trace in fig.data:
                        if trace.mode == 'lines':  # This is the trendline
                            trace.showlegend = False
                            trace.hoverinfo = 'skip'
                            trace.hovertemplate = ''
                            
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"Cannot plot '{title}': missing column")
            
            # 1. Distribution of Students by Mode of Entry
            if 'MODE_OF_ENTRY' in df.columns:
                st.markdown("### Enrolment Distribution by Mode of Entry")
                mode_counts = df['MODE_OF_ENTRY'].value_counts()
                fig = px.pie(values=mode_counts.values, names=mode_counts.index, title="Students by Mode of Entry")
                st.plotly_chart(fig, use_container_width=True)
            
            # 2. Yearly Enrolment by Gender
            if 'GENDER' in df.columns:
                st.markdown("### Yearly Enrolment by Gender")
                gender_year = df[df['ENROLED'] == 'YES'].groupby(['YEAR', 'GENDER']).size().unstack().fillna(0)
                fig = px.bar(gender_year, barmode='group', title="Yearly Enrolment by Gender")
                st.plotly_chart(fig, use_container_width=True)
            
            # 3. Enrolment by Year (Faculties)
            if 'FACULTY' in df.columns:
                st.markdown("### Enrolment by Year (Faculties)")
                faculty_year = df[df['ENROLED'] == 'YES'].groupby(['YEAR', 'FACULTY']).size().reset_index(name='enrollment')
                fig = px.line(faculty_year, x='YEAR', y='enrollment', color='FACULTY', title="Enrolment by Year (Faculties)", markers=True)
                st.plotly_chart(fig, use_container_width=True)
            
            # 4. Total Enrolment by Year Chart
            st.markdown("### Total Enrolment by Year Chart")
            total_year = df[df['ENROLED'] == 'YES'].groupby('YEAR').size().reset_index(name='TOTAL_ENROLLED')
            fig = px.line(total_year, x='YEAR', y='TOTAL_ENROLLED', title="Total Enrolment by Year", markers=True)
            st.plotly_chart(fig, use_container_width=True)

            # 5. Enrolment distribution by State of Origin (very important in Nigeria)
            if 'STATE_OF_ORIGIN' in df.columns:
                st.markdown("### Enrolment Distribution by State of Origin")
                state_enrol = df[df['ENROLED'] == 'YES']['STATE_OF_ORIGIN'].value_counts().head(15)  # top 15
                fig = px.bar(
                    state_enrol, 
                    x=state_enrol.index, 
                    y=state_enrol.values,
                    title="Top States of Origin - Enrolled Students",
                    labels={'x': 'State of Origin', 'y': 'Number of Enrolled Students'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Enrolment versus External and Policy Variables")
            # 6. Enrolment versus Gross Domestic Product Growth Percentage Chart
            plot_scatter('GDP_GROWTH_PERCENTAGE', title="Enrolment versus Gross Domestic Product Growth Percentage", x_label="GDP Growth Percentage (%)")
            
            # 7. Enrolment versus Unemployment Rate Percentage Chart
            plot_scatter('UNEMPLOYMENT_RATE_PERCENTAGE', title="Enrolment versus Unemployment Rate Percentage", x_label="Unemployment Rate Percentage (%)")
                      
            # 8. Enrolment versus Students per staff chart
            plot_scatter('STUDENTS_PER_STAFF', title="Enrolment versus Students per Staff", x_label="Students per Staff")
            
            # 9. Enrolment versus Budget per Student Chart
            plot_scatter('BUDGET_PER_STUDENT', title="Enrolment versus Budget per Student", x_label="Budget per Student (₦)")
            
            # 10. Enrolment versus Hostel Allocation Probability Chart
            plot_scatter('HOSTEL_ALLOCATION_PROBABILITY', title="Enrolment versus Hostel Allocation Probability", x_label="Hostel Allocation Probability (%)")
            
            # 11. Enrolment versus Departmental Post UTME Cut Off Chart
            plot_scatter('DEPT_POST_UTME_CUT_OFF', title="Enrolment versus Departmental Post UTME Cut Off", x_label="Departmental Post UTME Cut Off")

        with tab5:
            st.subheader("Statistical Summary")
            st.markdown("### Descriptive Statistics (Numeric Columns)")
            st.dataframe(df.describe(), use_container_width=True)
           
            if df.select_dtypes(include=['object']).shape[1] > 0:
                st.markdown("### Categorical Variables Summary")
                cat_summary = df.select_dtypes(include=['object']).describe()
                st.dataframe(cat_summary, use_container_width=True)
   
    else:
        # Show helpful message when no data is loaded
        st.info("👆 **No data loaded.** Choose a method above to get started:")
        st.markdown("""
        - **Upload CSV File**: Browse and upload your file
        - **Use Sample Data**: Quick start with pre-loaded data
        - **Paste CSV Data**: Copy-paste your data directly
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; padding: 2rem 0;'>
        <p><strong>© University of Ibadan 2026</strong></p>
        <p>Developed as part of M.Info.Sci research on Machine Learning for Enrolment Prediction and Resource Optimisation</p>
        <p style='margin-top: 1rem; font-size: 0.8rem; color: #999;'>All Rights Reserved © 2026</p>
    </div>
    """, unsafe_allow_html=True)
    
def prediction_tool():
    """Prediction tool with enrollment forecasting and resource allocation"""
    
    st.markdown('<div class="main-header">🎓 University of Ibadan</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enrolment Trend Prediction and Resource Optimisation Planning Tool</div>', unsafe_allow_html=True)
   
    try:
        enroll_model, enroll_features, resource_model, resource_features, metadata = load_models()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()
   
    # Sidebar inputs
    st.sidebar.header("🏛️ Organizational Structure")
    st.sidebar.markdown('<div class="hierarchy-card"><strong>📍 Selection Path:</strong><br>University of Ibadan → Faculty</div>', unsafe_allow_html=True)
   
    faculty_options = sorted(list(UNIVERSITY_STRUCTURE.keys()))
    faculty = st.sidebar.selectbox("🎓 Select Faculty", faculty_options, index=faculty_options.index('SCIENCE') if 'SCIENCE' in faculty_options else 0)
   
    st.sidebar.markdown(f"""
    <div style='background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%); padding: 0.8rem; border-radius: 5px; margin-top: 0.5rem;'>
        <strong style='color: #ffffff;'>Current Selection:</strong><br>
        <small style='color: #e3f2fd;'>🏛️ University of Ibadan<br>📚 {faculty}</small>
    </div>
    """, unsafe_allow_html=True)
   
    st.sidebar.markdown("---")
    st.sidebar.header("📝 Input Parameters")
   
    year = st.sidebar.slider("Target Year", 2025, 2035, 2026)
    total_enrollment = st.sidebar.number_input("Current Total Enrollment", min_value=50, max_value=50000, value=4000, step=50)
    annual_budget = st.sidebar.number_input("Annual Department Budget (₦)", min_value=1_000_000, max_value=1_000_000_000, value=50_000_000, step=5_000_000, format="%d")
   
    col_staff1, col_staff2 = st.sidebar.columns(2)
    with col_staff1:
        male_staff = st.number_input("Male Staff", min_value=0, max_value=1000, value=700, step=10)
    with col_staff2:
        female_staff = st.number_input("Female Staff", min_value=0, max_value=1000, value=500, step=10)
   
    hostel_prob = st.sidebar.slider("Hostel Allocation Probability (%)", 0, 100, 40)
    strike_months = st.sidebar.slider("Strike Duration (months)", 0, 12, 1)
   
    st.sidebar.markdown("### 📊 Economic Indicators")
    gdp_growth = st.sidebar.slider("GDP Growth Rate (%)", -5.0, 10.0, 2.5, 0.1)
    unemployment = st.sidebar.slider("Unemployment Rate (%)", 0.0, 40.0, 20.0, 0.5)
   
    # Calculate metrics
    total_staff = male_staff + female_staff
    budget_per_student = annual_budget / (total_enrollment if total_enrollment > 0 else 1)
    student_staff_ratio = total_enrollment / (total_staff if total_staff > 0 else 1)
    log_budget = np.log1p(annual_budget)
   
    # Context display
    st.markdown("### 📋 Analysis Context")
    context_col1, context_col2 = st.columns(2)
   
    with context_col1:
        st.markdown(f"""
        <div style='background: #667eea20; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #667eea;'>
            <strong>🏛️ University</strong><br>
            <span style='font-size: 1.3rem;'>University of Ibadan</span>
        </div>
        """, unsafe_allow_html=True)
   
    with context_col2:
        st.markdown(f"""
        <div style='background: #764ba220; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #764ba2;'>
            <strong>📚 Faculty</strong><br>
            <span style='font-size: 1.3rem;'>{faculty}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # ENROLLMENT PREDICTION
    st.markdown("---")
    st.header("📈 Enrolment Trend Prediction")
    
    enroll_input = pd.DataFrame([{
        'year': year, 'total_enrollment': total_enrollment, 'annual_budget_dept(₦)': annual_budget,
        'fac_staff_count_male': male_staff, 'fac_staff_count_female': female_staff,
        'hostel_allocation_probability': hostel_prob, 'strike_duration_months': strike_months,
        'gdp_growth_percentage': gdp_growth, 'unemployment_rate_percentage': unemployment,
        'total_staff': total_staff, 'log_budget': log_budget
    }])
    
    for col in enroll_features:
        if col not in enroll_input.columns:
            enroll_input[col] = 0
    enroll_input = enroll_input[enroll_features]
    
    enroll_growth_pred = enroll_model.predict(enroll_input)[0]
    
    def calculate_uncertainty(prediction_value, base_uncertainty=0.15):
        uncertainty = abs(prediction_value) * base_uncertainty + 5.0
        return round(uncertainty, 1)
    
    enroll_uncertainty = calculate_uncertainty(enroll_growth_pred)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style='font-size: 0.9rem; opacity: 0.9;'>Predicted Enrolment Growth Rate</div>
            <div style='font-size: 3.5rem; font-weight: bold; margin: 1rem 0;'>{enroll_growth_pred:.1f}%</div>
            <div style='font-size: 1.2rem;'>
                {'⬆️ Strong Growth Expected' if enroll_growth_pred > 10 else '📈 Moderate Growth' if enroll_growth_pred > 5 else '➡️ Stable Enrolment' if enroll_growth_pred > 0 else '⬇️ Decline Expected'}
            </div>
            <div style='font-size: 0.95rem; opacity: 0.85; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.3);'>
                <strong>Uncertainty Range:</strong> ±{enroll_uncertainty} pp<br>
                <small style='font-size: 0.8rem;'>Range: {enroll_growth_pred - enroll_uncertainty:.1f}% to {enroll_growth_pred + enroll_uncertainty:.1f}%</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Key factors
    st.markdown("### 🔍 Key Predictive Factors")
    fact_col1, fact_col2, fact_col3 = st.columns(3)
    
    with fact_col1:
        econ_health = gdp_growth - (unemployment / 4)
        econ_status = "Strong" if econ_health > 2 else "Moderate" if econ_health > 0 else "Weak"
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%); padding: 1rem; border-left: 4px solid #66bb6a; border-radius: 5px;'>
            <strong style='color: #ffffff;'>📊 Economic Conditions</strong><br>
            <span style='font-size: 1.5rem; font-weight: bold; color: #a5d6a7;'>{econ_status}</span><br>
            <small style='color: #c8e6c9;'>GDP: {gdp_growth:.1f}% | Unemployment: {unemployment:.1f}%</small>
        </div>
        """, unsafe_allow_html=True)
    
    with fact_col2:
        fac_demand = "High" if faculty in ['SCIENCE', 'CLINICAL SCIENCES', 'TECHNOLOGY'] else "Moderate" if faculty in ['LAW', 'ARTS', 'SOCIAL SCIENCES'] else "Average"
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%); padding: 1rem; border-left: 4px solid #66bb6a; border-radius: 5px;'>
            <strong style='color: #ffffff;'>🎓 Faculty Demand</strong><br>
            <span style='font-size: 1.5rem; font-weight: bold; color: #a5d6a7;'>{fac_demand}</span><br>
            <small style='color: #c8e6c9;'>{faculty}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with fact_col3:
        disruption = "None" if strike_months == 0 else "Minor" if strike_months <= 3 else "Significant"
        disruption_impact = strike_months * -1.5
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%); padding: 1rem; border-left: 4px solid #66bb6a; border-radius: 5px;'>
            <strong style='color: #ffffff;'>⚠️ Disruption Level</strong><br>
            <span style='font-size: 1.5rem; font-weight: bold; color: #a5d6a7;'>{disruption}</span><br>
            <small style='color: #c8e6c9;'>{strike_months} month(s) | Impact: {disruption_impact:+.1f} pp</small>
        </div>
        """, unsafe_allow_html=True)
    
    
    # SINGLE-YEAR PROJECTION (2026)
    st.markdown("---")
    st.header("📊 1-Year Enrolment Projection")

    # Use the predicted growth rate for next year only
    enroll_growth_pred = enroll_model.predict(enroll_input)[0] / 100  # convert % to decimal

    projected_enrollment = round(total_enrollment * (1 + enroll_growth_pred))  # e.g., 2800 → ~2904
    annual_growth_students = projected_enrollment - total_enrollment  # e.g., +104

    projection_years = [year, year + 1]  # current year + next year
    projected_enrollment_list = [total_enrollment, projected_enrollment]

    # Display
    st.metric("Projected Enrolment Next Year", f"{projected_enrollment:,}", 
          delta=f"+{annual_growth_students:,} students ({enroll_growth_pred*100:.1f}%)")

    fig_projection = go.Figure()
    fig_projection.add_trace(go.Scatter(
        x=projection_years, 
        y=projected_enrollment_list, 
        mode='lines+markers+text',
        name='Projected Enrolment',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10),
        text=[f'{int(e):,}' for e in projected_enrollment_list],
        textposition='top center'
    ))
    fig_projection.update_layout(
        title=f"1-Year Enrolment Projection for {faculty}",
        xaxis_title="Year", yaxis_title="Total Enrolment", height=400,
        template="plotly_white"
    )
    st.plotly_chart(fig_projection, use_container_width=True)

    # Update the summary table for 1 year only
    summary_data = [
        {'Year': year, 'Projected Enrolment': f"{int(total_enrollment):,}", 'Growth Rate': "Current", 'Change': "—"},
        {'Year': year + 1, 'Projected Enrolment': f"{projected_enrollment:,}", 
         'Growth Rate': f"{enroll_growth_pred*100:.1f}%", 'Change': f"+{annual_growth_students:,}"}
    ]
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    
    # Key Statistics
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        annual_growth_students = projected_enrollment - total_enrollment  # e.g. 2904 - 2800 = 104
        annual_growth_pct = ((projected_enrollment - total_enrollment) / total_enrollment) * 100
        st.metric("1-Year Growth", f"{annual_growth_pct:+.1f}%", 
              delta=f"+{annual_growth_students:,} students")
    
    with stat_col2:
        st.metric("Expected Annual Growth", f"{enroll_growth_pred*100:.1f}%", delta="Per year")

    with stat_col3:
        st.metric(f"Projected {year + 1} Enrolment", f"{projected_enrollment:,}", 
              delta=f"From {int(total_enrollment):,}")

    with stat_col4:
        st.metric("Peak Growth Year", year + 1, delta=f"{enroll_growth_pred*100:.1f}%")
    
    # Planning Implications
    st.markdown("### 💡 Planning Implications")
    impl_col1, impl_col2 = st.columns(2)
    
    with impl_col1:
        st.markdown('<div class="insight-card" style="background-color: #121212; color: #e0e0e0; padding: 1.5rem; border-radius: 12px; border: 1px solid #444; box-shadow: 0 4px 15px rgba(0,0,0,0.7);"><strong>📊 Capacity Planning:</strong></div>', unsafe_allow_html=True)
    
        annual_growth_pct = ((projected_enrollment - total_enrollment) / total_enrollment) * 100
    
        if annual_growth_pct > 10:
            st.error("🚨 **High Growth Alert:** Significant infrastructure expansion needed")
        elif annual_growth_pct > 5:
            st.warning("📈 **Moderate Growth:** Gradual capacity expansion recommended")
        else:
            st.success("➡️ **Stable Growth:** Maintain current capacity")
    
    with impl_col2:
        st.markdown('<div class="insight-card" style="background-color: #121212; color: #e0e0e0; padding: 1.5rem; border-radius: 12px; border: 1px solid #444; box-shadow: 0 4px 15px rgba(0,0,0,0.7);"><strong>🎯 Resource Requirements:</strong></div>', unsafe_allow_html=True)
    
        final_student_staff = projected_enrollment / total_staff if total_staff > 0 else 0
        additional_staff_needed = max(0, int((projected_enrollment / 20) - total_staff))
    
        st.info(f"📌 Projected ratio next year: **{final_student_staff:.1f}:1**")
        if additional_staff_needed > 0:
            st.warning(f"👥 Recommended staff increase: **{additional_staff_needed:,}** lecturers")
    
    # RESOURCE ALLOCATION
    st.markdown("---")
    st.header("💰 Resource Allocation Impact")
    
    resource_input = pd.DataFrame([{
        'year': year, 'total_enrollment': total_enrollment, 'annual_budget_dept(₦)': annual_budget,
        'fac_staff_count_male': male_staff, 'fac_staff_count_female': female_staff,
        'hostel_allocation_probability': hostel_prob, 'strike_duration_months': strike_months,
        'gdp_growth_percentage': gdp_growth, 'unemployment_rate_percentage': unemployment,
        'total_staff': total_staff, 'budget_per_student': budget_per_student,
        'student_staff_ratio': student_staff_ratio, 'log_budget': log_budget
    }])
    
    for col in resource_features:
        if col not in resource_input.columns:
            resource_input[col] = 0
    resource_input = resource_input[resource_features]
    
    grad_rate_pred = resource_model.predict(resource_input)[0]
    
    # Size adjustment for small faculties
    def adjust_for_faculty_size(predicted_rate, enrollment, staff_ratio, budget_per_student):
        if enrollment < 1500 and staff_ratio < 20:
            ratio_bonus = (20 - staff_ratio) * 0.5
            predicted_rate += ratio_bonus
        if enrollment < 1500 and budget_per_student > 60000:
            funding_bonus = min((budget_per_student - 60000) / 20000 * 2, 5)
            predicted_rate += funding_bonus
        if enrollment < 1000:
            predicted_rate += 3.0
        elif enrollment < 1500:
            predicted_rate += 1.5
        return min(predicted_rate, 95.0)
    
    grad_rate_pred = adjust_for_faculty_size(grad_rate_pred, total_enrollment, student_staff_ratio, budget_per_student)
    
    def calculate_grad_uncertainty(grad_rate, base_uncertainty=0.10):
        distance_from_mean = abs(grad_rate - 65) / 65
        uncertainty = grad_rate * base_uncertainty + (distance_from_mean * 3.0)
        return round(uncertainty, 1)
    
    grad_uncertainty = calculate_grad_uncertainty(grad_rate_pred)
    
    # Display graduation rate
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="metric-card-secondary">
            <div style='font-size: 0.9rem; opacity: 0.9;'>Expected Graduation Rate</div>
            <div style='font-size: 3.5rem; font-weight: bold; margin: 1rem 0;'>{grad_rate_pred:.1f}%</div>
            <div style='font-size: 0.9rem; opacity: 0.85;'>National Average: 65% | NUC Target: 85%</div>
            <div style='font-size: 0.95rem; opacity: 0.85; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.3);'>
                <strong>Uncertainty Range:</strong> ±{grad_uncertainty} pp<br>
                <small style='font-size: 0.8rem;'>Range: {max(0, grad_rate_pred - grad_uncertainty):.1f}% to {min(100, grad_rate_pred + grad_uncertainty):.1f}%</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Resource metrics
    st.markdown("### 📊 Resource Metrics")
    met_col1, met_col2, met_col3 = st.columns(3)
    
    with met_col1:
        ratio_color = "#28a745" if student_staff_ratio < 15 else "#ffc107" if student_staff_ratio < 25 else "#dc3545"
        ratio_status = "Excellent" if student_staff_ratio < 15 else "Good" if student_staff_ratio < 25 else "Strained"
        st.markdown(f"""
        <div style='background: {ratio_color}20; padding: 1.5rem; border-radius: 10px; border-left: 4px solid {ratio_color};'>
            <div style='color: #666; font-size: 0.85rem; font-weight: bold;'>STUDENT-STAFF RATIO</div>
            <div style='font-size: 2.5rem; font-weight: bold; color: {ratio_color}; margin: 0.5rem 0;'>{student_staff_ratio:.2f}:1</div>
            <div style='font-size: 0.85rem;'>{ratio_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with met_col2:
        budget_status = "Well-funded" if budget_per_student > 100000 else "Adequate" if budget_per_student > 50000 else "Limited"
        st.markdown(f"""
        <div style='background: #2196f320; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #2196f3;'>
            <div style='color: #666; font-size: 0.85rem; font-weight: bold;'>BUDGET PER STUDENT</div>
            <div style='font-size: 2rem; font-weight: bold; color: #2196f3; margin: 0.5rem 0;'>₦{budget_per_student:,.0f}</div>
            <div style='font-size: 0.85rem;'>{budget_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with met_col3:
        st.markdown(f"""
        <div style='background: #9c27b020; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #9c27b0;'>
            <div style='color: #666; font-size: 0.85rem; font-weight: bold;'>TOTAL ACADEMIC STAFF</div>
            <div style='font-size: 2.5rem; font-weight: bold; color: #9c27b0; margin: 0.5rem 0;'>{total_staff}</div>
            <div style='font-size: 0.85rem;'>{male_staff}M / {female_staff}F</div>
        </div>
        """, unsafe_allow_html=True)
    
    # SCENARIO SIMULATION
    st.markdown("---")
    st.header("📊 Scenario Simulation: Resource Impact")
    
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        additional_lecturers = st.slider("Additional Lecturers to Hire", 0, 500, 100, step=10)
    with sim_col2:
        budget_increase_pct = st.slider("Budget Increase (%)", -20, 100, 25, step=5)
    
    scenario_total_staff = total_staff + additional_lecturers
    budget_adjusted = annual_budget * (1 + budget_increase_pct / 100)
    scenario_student_staff_ratio = total_enrollment / scenario_total_staff if scenario_total_staff > 0 else 0
    scenario_budget_per_student = budget_adjusted / total_enrollment if total_enrollment > 0 else 0
    
    grad_rate_baseline = grad_rate_pred
    grad_rate_effect = (3.1 / max(scenario_student_staff_ratio, 1)) * 4
    scenario_grad_rate = min(grad_rate_baseline + grad_rate_effect, 100)
    
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        change = scenario_student_staff_ratio - student_staff_ratio
        st.metric("Projected Student-Staff Ratio", f"{scenario_student_staff_ratio:.2f}:1", delta=f"{change:.2f}", delta_color="inverse")
    
    with result_col2:
        change = scenario_budget_per_student - budget_per_student
        st.metric("Budget per Student", f"₦{scenario_budget_per_student:,.0f}", delta=f"₦{change:,.0f}")
    
    with result_col3:
        change = scenario_grad_rate - grad_rate_pred
        st.metric("Projected Graduation Rate", f"{scenario_grad_rate:.1f}%", delta=f"{change:.1f}pp")


    # RESOURCE OPTIMISATION

    st.markdown("---")
    st.header("🎯 Optimised Resource Allocation Recommendation")
    
    # Optimization controls
    st.markdown("### ⚙️ Optimisation Parameters")
    opt_col1, opt_col2, opt_col3 = st.columns(3)
    
    with opt_col1:
        budget_limit = st.number_input(
            "Maximum Additional Budget (₦)", 
            min_value=0, 
            max_value=1_000_000_000, 
            value=100_000_000,
            step=10_000_000,
            help="Maximum additional budget available for year 1"
        )
    
    with opt_col2:
        target_grad_rate = st.slider(
            "Target Graduation Rate (%)", 
            min_value=75, 
            max_value=95, 
            value=85,
            help="Desired graduation rate to achieve"
        )
    
    with opt_col3:
        max_staff_ratio = st.slider(
            "Maximum Student-Staff Ratio", 
            min_value=10, 
            max_value=25, 
            value=18,
            help="Maximum acceptable student-staff ratio"
        )
    
    # Run optimization button
    if st.button("🚀 Run Optimisation", type="primary", use_container_width=True):
        with st.spinner("🔄 Running optimisation algorithm... This may take some seconds"):
            
            # Call optimization function
            optimal_plan = optimize_resource_allocation(
                current_enrollment=total_enrollment,
                projected_enrollment=projected_enrollment,
                current_staff=total_staff,
                current_male_staff=male_staff,
                current_female_staff=female_staff,
                current_budget=annual_budget,
                budget_limit=budget_limit,
                target_grad_rate=target_grad_rate,
                max_ratio=max_staff_ratio,
                current_grad_rate=grad_rate_pred,
                faculty=faculty,
                resource_model=resource_model,
                resource_features=resource_features
            )
        
        # Key metrics
        st.markdown("### 📊 Optimal Solution Summary")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                "Total New Staff (Year 1)",
                f"{optimal_plan['total_new_staff']:.0f}",
                delta=f"{optimal_plan['male_hires']:.0f}M / {optimal_plan['female_hires']:.0f}F"
            )
        
        with metric_col2:
            st.metric(
                "Optimal Annual Budget",
                f"₦{optimal_plan['optimal_budget']:,.0f}",
                delta=f"+₦{optimal_plan['budget_increase']:,.0f}"
            )
        
        with metric_col3:
            st.metric(
                "Achieved Graduation Rate",
                f"{optimal_plan['achieved_grad_rate']:.1f}%",
                delta=f"+{optimal_plan['grad_rate_improvement']:.1f}pp"
            )
        
        with metric_col4:
            st.metric(
                "Final Student-Staff Ratio",
                f"{optimal_plan['final_ratio']:.1f}:1",
                delta=f"{optimal_plan['ratio_change']:.1f}",
                delta_color="inverse"
            )
        
        # Year-by-year plan
        st.markdown("### 📅 Year 1 Implementation Plan")
        plan_df = pd.DataFrame(optimal_plan['yearly_plan'])
        st.dataframe(plan_df, use_container_width=True, hide_index=True)
        
        # Budget breakdown (updated - only salaries + buffer)
        st.markdown("### 💰 Additional Budget Breakdown (Year 1)")

        # Show total as a nice metric
        st.metric(
            label="Total Additional Budget Needed",
            value=f"₦{optimal_plan['total_investment']:,.0f}",
            delta=f"+₦{optimal_plan['budget_increase']:,.0f}",
            delta_color="normal"
        )

        # Calculate salary cost for new staff
        total_salary_cost = optimal_plan['total_new_staff'] * 1.8e6  # ₦1.8M per staff
        other_costs = optimal_plan['budget_increase'] - total_salary_cost

        st.markdown(f"""
        **Budget Breakdown (Year 1):**
        - **Salaries for new staff**: ₦{total_salary_cost/1e6:.1f}M
        - **Other costs (buffer)**: ₦{other_costs/1e6:.1f}M

        *Note: Additional budget is primarily for new staff salaries + 10% buffer for onboarding, training, etc.*
        """)  
        
        # Keep the bar chart here - it works perfectly with the new optimization
        st.markdown("### 📊 Current vs Optimised Performance")
        comparison_metrics = ['Graduation Rate (%)', 'Student-Staff Ratio', 'Budget per Student (₦/1000)']
        current_values = [grad_rate_pred, student_staff_ratio, budget_per_student/1000]
        optimized_values = [optimal_plan['achieved_grad_rate'], optimal_plan['final_ratio'], 
                   optimal_plan['optimal_budget_per_student']/1000]

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name='Current',
            x=comparison_metrics,
            y=current_values,
            marker_color='#667eea'
        ))
        fig_compare.add_trace(go.Bar(
            name='Optimized',
            x=comparison_metrics,
            y=optimized_values,
            marker_color='#10b981'
        ))
        fig_compare.update_layout(
            title='Current vs Optimised Performance',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Cost-benefit analysis
        st.markdown("### 📈 Cost-Benefit Analysis")
        cba_col1, cba_col2, cba_col3 = st.columns(3)
        
        with cba_col1:
            st.markdown(f"""
            <div style='background: #f0f9ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6;'>
                <strong style='color: #1e40af;'>Total Investment</strong><br>
                <span style='font-size: 1.8rem; color: #3b82f6;'>₦{optimal_plan['total_investment']/1e6:.1f}M</span><br>
                <small style='color: #64748b;'>For year 1</small>
            </div>
            """, unsafe_allow_html=True)
        
        with cba_col2:
            st.markdown(f"""
            <div style='background: #f0fdf4; padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981;'>
                <strong style='color: #065f46;'>Additional Graduates</strong><br>
                <span style='font-size: 1.8rem; color: #10b981;'>+{optimal_plan['additional_graduates']:.0f}</span><br>
                <small style='color: #64748b;'>Students</small>
            </div>
            """, unsafe_allow_html=True)
                
        # Download button for detailed plan
        st.markdown("### 📥 Export Optimisation Results")
        
        detailed_report = generate_optimization_report(optimal_plan, faculty, year)
        
        st.download_button(
            label="📄 Download Detailed Implementation Plan (CSV)",
            data=detailed_report,
            file_name=f"optimal_resource_plan_{faculty}_{year}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Implementation recommendations
        st.markdown("### ✅ Implementation Recommendations")
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.markdown("#### 🚀 Immediate Actions (Year 1)")
            for action in optimal_plan['immediate_actions']:
                st.success(f"✓ {action}")
        
        with rec_col2:
            st.markdown("#### 📅 Implementation Strategy")
            for strategy in optimal_plan['long_term_strategies']:
                st.info(f"→ {strategy}")
        
        # Constraints met/violated
        st.markdown("### ⚖️ Constraint Satisfaction")
        constraints_col1, constraints_col2 = st.columns(2)
        
        with constraints_col1:
            st.markdown("**✅ Constraints Met:**")
            for constraint in optimal_plan['constraints_met']:
                st.success(constraint)
        
        with constraints_col2:
            if optimal_plan['constraints_violated']:
                st.markdown("**⚠️ Constraints Challenged:**")
                for constraint in optimal_plan['constraints_violated']:
                    st.warning(constraint)
            else:
                st.success("✅ All constraints satisfied!")
    
    else:
        # Show placeholder when optimization hasn't been run
        st.info("👆 Configure optimisation parameters above and click **'Run Optimisation'** to find the optimal resource allocation plan")
        
        st.markdown("""
        ### 🎯 What You'll Get:
        
        - **Optimal hiring plan** for year 1 (male/female breakdown)
        - **Budget allocation** across categories (salaries, infrastructure, teaching, research)
        - **Performance projections** (graduation rates, student-staff ratios)
        - **Cost-benefit analysis** with ROI calculations
        - **Sensitivity analysis** showing robustness of solutions
        - **Alternative solutions** with different trade-offs
        - **Implementation roadmap** with actionable recommendations
        - **Downloadable detailed plan** for presentation to administrators
        """)
        
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; padding: 2rem 0;'>
        <p><strong>© University of Ibadan 2026</strong></p>
        <p>Developed as part of M.Info.Sci research on Machine Learning for Enrolment Prediction and Resource Optimisation</p>
        <p style='margin-top: 1rem; font-size: 0.8rem; color: #999;'>All Rights Reserved © 2026</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# MAIN APP LOGIC
# --------------------------------------------------
render_sidebar()

if st.session_state.page == 'landing':
    landing_page()
elif st.session_state.page == 'eda':
    eda_dashboard()
elif st.session_state.page == 'prediction':
    prediction_tool()
