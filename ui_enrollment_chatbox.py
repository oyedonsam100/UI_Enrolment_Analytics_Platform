"""
Q&A Chatbox for University of Ibadan Enrollment Prediction System
Fixed version with reliable question matching
"""

import streamlit as st
import json
from datetime import datetime
import pandas as pd
from typing import Optional, Dict, Any, List
import io
import base64

class UIEnrollmentChatbox:
    """
    Q&A chatbox for University of Ibadan's enrollment prediction platform.
    """
    
    def __init__(self):
        """Initialize the chatbox with UI-specific Q&A pairs."""
        
        # Use a list of (patterns, answer) instead of dictionary for better control
        self.qa_patterns = [
            # EXACT MATCHES FIRST (most reliable)
            (["what is this", "what is this app", "what is this platform"], {
                "answer": "This is the University of Ibadan Enrollment Prediction and Resource Optimization Platform. It helps predict student enrollment trends and optimize resource allocation using AI and machine learning.",
                "category": "general"
            }),
            
            (["about", "about this app", "tell me about this", "what does this do"], {
                "answer": "This platform provides three main features: 1) EDA Dashboard for exploring enrollment data with visualizations, 2) AI-powered enrollment predictions for forecasting growth, and 3) Resource optimization tools for planning staff and budget allocation.",
                "category": "general"
            }),
            
            (["what is enrollment", "define enrollment", "enrollment meaning"], {
                "answer": "Enrollment is the total number of students admitted and registered at the university. This platform predicts future enrollment by analyzing historical trends, economic factors (GDP, unemployment), faculty capacity, budget, and policy factors like strike duration and hostel availability.",
                "category": "general"
            }),
            
            (["what is prediction", "what are predictions", "explain prediction"], {
                "answer": "Predictions are AI-generated forecasts of future outcomes. This app makes two types: 1) Enrollment Growth Rate - predicting how enrollment will change (e.g., +5% next year), and 2) Graduation Rate - predicting what percentage of students will graduate based on current resources.",
                "category": "general"
            }),
            
            (["what is enrollment prediction", "enrollment forecasting"], {
                "answer": "Enrollment prediction uses machine learning models to forecast how many students will enroll next year. The system analyzes 10+ factors including GDP growth, unemployment rates, budget per student, student-staff ratios, and historical enrollment patterns to generate 1-year projections with uncertainty ranges.",
                "category": "prediction"
            }),
            
            # APP USAGE
            (["how do i use", "how to use", "how do i start", "getting started"], {
                "answer": "Start by clicking 'EDA Dashboard' in the sidebar to explore enrollment data. Then go to 'Prediction Tool' to make forecasts. Select your faculty, input current parameters (enrollment, budget, staff), set economic indicators, and click buttons to generate predictions and optimization recommendations.",
                "category": "usage"
            }),
            
            (["navigate", "navigation", "how to navigate"], {
                "answer": "Use the sidebar menu on the left: 🏠 Home for overview, 📊 EDA Dashboard to analyze data, 🎯 Prediction Tool to forecast and optimize. Each section has clear instructions and input fields.",
                "category": "usage"
            }),
            
            # DATA
            (["what data", "data needed", "data required", "data format"], {
                "answer": "Upload CSV files with these columns: YEAR, GENDER, FACULTY, DEPARTMENT, MODE_OF_ENTRY, ENROLED, ANNUAL_BUDGET_DEPT(₦), FAC_STAFF_COUNT_MALE, FAC_STAFF_COUNT_FEMALE, HOSTEL_ALLOCATION_PROBABILITY, STRIKE_DURATION_MONTHS, GDP_GROWTH_PERCENTAGE, UNEMPLOYMENT_RATE_PERCENTAGE, DEPT_POST_UTME_CUT_OFF.",
                "category": "data"
            }),
            
            (["csv", "upload csv", "file upload", "upload data"], {
                "answer": "In the EDA Dashboard, you have three options: 1) Upload CSV File - browse and select your file, 2) Use Sample Data - load pre-configured example data, or 3) Paste CSV Data - copy-paste your data directly. If upload fails, try re-saving as 'CSV UTF-8' format in Excel.",
                "category": "data"
            }),
            
            (["csv not working", "upload fails", "cant upload", "upload error"], {
                "answer": "Common fixes: 1) Open file in Excel, Save As → 'CSV UTF-8', 2) Try the 'Paste CSV Data' option instead, 3) Use 'Sample Data' to test the system, 4) Check file is under 200MB, 5) Remove special characters from data.",
                "category": "troubleshooting"
            }),
            
            (["sample data", "test data", "demo data"], {
                "answer": "Click 'Use Sample Data' in the EDA Dashboard to load pre-configured enrollment data. This lets you explore all features without uploading your own files. Great for testing and learning how the system works.",
                "category": "data"
            }),
            
            # PREDICTIONS
            (["how accurate", "accuracy", "prediction accuracy"], {
                "answer": "Predictions include uncertainty ranges (typically ±5-8 percentage points). For example, a 15% ±5pp prediction means growth could be 10-20%. Accuracy depends on data quality and economic stability. The models are trained on historical UI data.",
                "category": "prediction"
            }),
            
            (["what affects enrollment", "enrollment factors", "factors affecting"], {
                "answer": "Key factors: GDP growth rate (economic health), unemployment rate (job market), faculty demand (e.g., Science/Medicine are high-demand), student-staff ratio (capacity), budget per student (resources), strike duration (disruptions), hostel probability (accommodation), and Post-UTME cut-off marks.",
                "category": "prediction"
            }),
            
            (["how far ahead", "forecast period", "prediction timeframe"], {
                "answer": "The system provides 1-year enrollment projections. This is the most reliable timeframe for accurate predictions. For longer-term planning (3-5 years), you can run multiple scenarios with different assumptions about economic conditions and policy changes.",
                "category": "prediction"
            }),
            
            (["uncertainty range", "confidence interval", "prediction range"], {
                "answer": "The uncertainty range (±X pp) shows the confidence interval. Example: 15% ±5pp means enrollment growth could be 10-20%. Larger ranges indicate higher uncertainty due to volatile economic conditions or limited historical data.",
                "category": "prediction"
            }),
            
            # GRADUATION RATES
            (["graduation rate", "graduation prediction"], {
                "answer": "The system predicts graduation rates based on student-staff ratio, budget per student, faculty size, and resource quality. Predictions are compared against the national average (65%) and NUC target (85%). Small faculties get automatic adjustments for their unique characteristics.",
                "category": "graduation"
            }),
            
            (["improve graduation", "increase graduation rate"], {
                "answer": "To improve graduation rates: 1) Reduce student-staff ratio to below 20:1, 2) Increase budget per student above ₦60,000, 3) Hire qualified lecturers, 4) Minimize disruptions (strikes), 5) Use the optimization tool to find the best resource allocation strategy.",
                "category": "graduation"
            }),
            
            # OPTIMIZATION
            (["how optimization works", "optimization algorithm"], {
                "answer": "The optimizer uses differential evolution to find the best hiring and budget allocation plan. It maximizes graduation rates while staying within your budget limit, maintaining good student-staff ratios (15-25:1), and achieving gender balance (30-50% female staff).",
                "category": "optimization"
            }),
            
            (["what to optimize", "optimize what", "resources optimize"], {
                "answer": "You can optimize: 1) Number of new lecturers to hire (male/female breakdown), 2) Budget allocation across salaries, infrastructure, teaching materials, and research, 3) Timing of hires across years, 4) Resource distribution to achieve target graduation rates.",
                "category": "optimization"
            }),
            
            (["optimization parameters", "what to set"], {
                "answer": "Set three parameters: 1) Maximum Additional Budget - total funds available (₦), 2) Target Graduation Rate - desired outcome (75-95%), 3) Maximum Student-Staff Ratio - quality threshold (typically 15-25:1). Then click 'Run Optimization'.",
                "category": "optimization"
            }),
            
            (["optimization time", "how long", "optimization speed"], {
                "answer": "Optimization typically takes 10-30 seconds. The algorithm evaluates thousands of possible solutions using differential evolution to find the optimal staff hiring and budget allocation strategy for your constraints.",
                "category": "optimization"
            }),
            
            # SCENARIO SIMULATION
            (["scenario simulation", "what if analysis", "simulation"], {
                "answer": "Scenario simulation lets you test 'what-if' situations. Adjust sliders for additional lecturers (0-500) and budget increase (-20% to +100%) to see immediate impacts on student-staff ratio, budget per student, and graduation rate projections.",
                "category": "simulation"
            }),
            
            # FACULTIES
            (["faculties", "which faculties", "list of faculties"], {
                "answer": "All 16 UI faculties are included: Agriculture, Arts, Basic Medical Sciences, Clinical Sciences, Dentistry, Education, Environmental Design & Management, Law, Pharmacy, Public Health, Renewable Natural Resources, Science, Social Sciences, Technology, and Veterinary Medicine.",
                "category": "structure"
            }),
            
            (["departments", "select department"], {
                "answer": "Predictions are made at the faculty level. Select your faculty from the sidebar dropdown in Prediction Tool. For department-level analysis, use the EDA Dashboard if your uploaded data includes department information in the DEPARTMENT column.",
                "category": "structure"
            }),
            
            # ECONOMIC INDICATORS
            (["economic indicators", "gdp unemployment"], {
                "answer": "GDP Growth Rate measures economic expansion (typically -5% to +10% in Nigeria, current ~2.5%). Unemployment Rate reflects job market conditions (current ~20%). Both strongly influence enrollment - higher GDP and lower unemployment typically increase enrollment.",
                "category": "economics"
            }),
            
            (["set indicators", "input economic data"], {
                "answer": "In Prediction Tool sidebar, adjust: 1) GDP Growth Rate slider (use recent NBS statistics, default 2.5%), 2) Unemployment Rate slider (current average ~20%). These critically affect enrollment predictions, so use accurate current data for best results.",
                "category": "economics"
            }),
            
            # EXPORTS
            (["download", "export results", "save report"], {
                "answer": "After running optimization, click '📄 Download Detailed Implementation Plan (CSV)' button. This exports a comprehensive report with hiring plans, budget breakdown, yearly schedule, immediate actions, long-term strategies, and constraint satisfaction analysis.",
                "category": "export"
            }),
            
            (["whats in export", "export contents"], {
                "answer": "The CSV export contains: Summary metrics (staff, budget, graduation rates), year-by-year implementation plan, budget breakdown by category, immediate actions for Year 1, long-term strategies for Years 2-5, and analysis of which constraints were met or challenged.",
                "category": "export"
            }),
            
            # TROUBLESHOOTING
            (["models not loading", "model error", "pkl error"], {
                "answer": "Ensure all 5 .pkl model files are in the 'models' folder: ui_enrollment_features.pkl, ui_enrollment_prediction_model.pkl, ui_resource_allocation_model.pkl, ui_resource_features.pkl, ui_system_metadata.pkl. Check the models folder is in the same directory as app.py.",
                "category": "troubleshooting"
            }),
            
            (["predictions wrong", "incorrect predictions", "bad predictions"], {
                "answer": "Verify: 1) Input parameters are realistic (check enrollment, budget, staff numbers), 2) Faculty selection is correct, 3) Economic indicators match current conditions (GDP ~2.5%, unemployment ~20%), 4) Remember predictions include uncertainty ranges showing possible variation.",
                "category": "troubleshooting"
            }),
        ]
        
        # Initialize session state
        if 'ui_chat_history' not in st.session_state:
            st.session_state.ui_chat_history = []
        if 'ui_chat_context' not in st.session_state:
            st.session_state.ui_chat_context = {}
    
    def find_predefined_answer(self, question: str) -> Optional[Dict[str, str]]:
        """
        Find matching answer using simple, reliable pattern matching.
        """
        question_lower = question.lower().strip().replace('?', '').replace('!', '').replace('.', '')
        
        # Try each pattern
        for patterns, answer_data in self.qa_patterns:
            for pattern in patterns:
                # Direct substring match (most reliable)
                if pattern in question_lower:
                    return answer_data
        
        # If no match, try keyword-based matching as last resort
        question_words = set(question_lower.split())
        
        # Remove common words
        stop_words = {'is', 'are', 'the', 'a', 'an', 'what', 'how', 'why', 'when', 
                      'where', 'can', 'do', 'does', 'did', 'this', 'that', 'these',
                      'i', 'my', 'me', 'you', 'your', 'it', 'its', 'to', 'for', 'of'}
        question_words = question_words - stop_words
        
        if len(question_words) == 0:
            return None
        
        best_match = None
        best_score = 0
        
        for patterns, answer_data in self.qa_patterns:
            for pattern in patterns:
                pattern_words = set(pattern.split()) - stop_words
                if len(pattern_words) == 0:
                    continue
                
                overlap = len(question_words.intersection(pattern_words))
                
                # Only consider if significant overlap
                if overlap >= 2 and overlap >= len(pattern_words) * 0.6:
                    score = overlap / len(pattern_words)
                    if score > best_score:
                        best_score = score
                        best_match = answer_data
        
        return best_match if best_score > 0.5 else None
    
    def _generate_fallback_response(self, question: str) -> str:
        """Generate helpful fallback when no match found."""
        question_lower = question.lower()
        
        # Topic detection
        if any(word in question_lower for word in ['data', 'upload', 'csv', 'file']):
            return """**Need help with data?** Try asking:

• "What data do I need?"
• "How to upload CSV?"
• "CSV not working"

Or go to **EDA Dashboard** → Upload your data"""
        
        elif any(word in question_lower for word in ['enroll', 'forecast', 'growth']):
            return """**Need help with predictions?** Try asking:

• "How accurate are predictions?"
• "What affects enrollment?"
• "What is enrollment prediction?"

Or go to **Prediction Tool** → Generate forecasts"""
        
        elif any(word in question_lower for word in ['optimize', 'resource', 'staff', 'budget']):
            return """**Need help with optimization?** Try asking:

• "How does optimization work?"
• "What can I optimize?"
• "How to improve graduation rate?"

Or use **Prediction Tool** → Scroll to optimization section"""
        
        elif any(word in question_lower for word in ['faculty', 'department']):
            return """**Need help with faculties?** Try asking:

• "Which faculties are included?"
• "How to select department?"

**16 Faculties Available:** Agriculture, Arts, Sciences, Engineering, Medicine, Law, and more."""
        
        else:
            return """**I can help with:**

📊 **App Usage:** "How do I use this?"
📁 **Data:** "What data format?"
🎯 **Predictions:** "How accurate are predictions?"
💰 **Optimization:** "How does optimization work?"
🏛️ **Faculties:** "Which faculties are included?"

**What would you like to know?**"""
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to chat history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        st.session_state.ui_chat_history.append(message)
    
    def save_chat_history(self):
        """Save chat history to JSON."""
        chat_data = {
            "university": "University of Ibadan",
            "export_date": datetime.now().isoformat(),
            "messages": st.session_state.ui_chat_history
        }
        return json.dumps(chat_data, indent=2)
    
    def render(self, app_context: Optional[Dict[str, Any]] = None, compact: bool = False):
        """Render the chatbox UI."""
        
        if app_context:
            st.session_state.ui_chat_context.update(app_context)
        
        if not compact:
            st.markdown("### 💬 Q&A Assistant")
            st.markdown("Ask me about enrollment predictions, resource optimization, or how to use the platform!")
        else:
            st.markdown("#### 💬 Q&A")
        
        # Controls
        if not compact:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("🗑️ Clear Chat", key="clear_ui_chat"):
                    st.session_state.ui_chat_history = []
                    st.rerun()
            
            with col2:
                if st.session_state.ui_chat_history:
                    chat_json = self.save_chat_history()
                    st.download_button(
                        label="📥 Export",
                        data=chat_json,
                        file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )
        
        # Chat display
        chat_container = st.container()
        
        with chat_container:
            messages_to_show = st.session_state.ui_chat_history[-5:] if compact else st.session_state.ui_chat_history
            
            if not messages_to_show and not compact:
                st.info("👋 **Hi! I'm your UI Enrollment Assistant.**\n\nAsk me anything about enrollment predictions, data analysis, or resource optimization.")
            
            for msg in messages_to_show:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        # Chat input
        user_question = st.chat_input("Ask about enrollment, predictions, or optimization...")
        
        if user_question:
            self.add_message("user", user_question)
            
            with st.chat_message("user"):
                st.markdown(user_question)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    predefined = self.find_predefined_answer(user_question)
                    
                    if predefined:
                        response = f"**{predefined['category'].title()}:** {predefined['answer']}"
                        metadata = {"source": "predefined", "category": predefined['category']}
                    else:
                        response = self._generate_fallback_response(user_question)
                        metadata = {"source": "fallback"}
                    
                    st.markdown(response)
                    self.add_message("assistant", response, metadata)
            
            st.rerun()
        
        # Quick questions
        if not compact:
            st.markdown("---")
            st.markdown("**Quick Questions:**")
            
            quick_questions = [
                "What is this app?",
                "What data do I need?",
                "How accurate are predictions?",
                "How does optimization work?"
            ]
            
            cols = st.columns(2)
            for idx, qq in enumerate(quick_questions):
                with cols[idx % 2]:
                    if st.button(qq, key=f"qq_ui_{idx}", use_container_width=True):
                        self.add_message("user", qq)
                        predefined = self.find_predefined_answer(qq)
                        if predefined:
                            response = f"**{predefined['category'].title()}:** {predefined['answer']}"
                            self.add_message("assistant", response, {"source": "predefined"})
                        st.rerun()


def render_ui_chatbox(app_context: Optional[Dict[str, Any]] = None, compact: bool = False):
    """Convenience function to render the chatbox."""
    if 'ui_chatbox' not in st.session_state:
        st.session_state.ui_chatbox = UIEnrollmentChatbox()
    
    st.session_state.ui_chatbox.render(app_context, compact)
