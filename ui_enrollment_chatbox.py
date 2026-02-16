"""
Q&A Chatbox for University of Ibadan Enrollment Prediction System
Customized for enrollment trends and resource optimization
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
    Q&A chatbox specifically for University of Ibadan's enrollment prediction 
    and resource optimization platform.
    """
    
    def __init__(self):
        """Initialize the chatbox with UI-specific Q&A pairs."""
        self.predefined_qa = {
            # GENERAL QUESTIONS
            "what is this": {
                "answer": "This is the University of Ibadan Enrollment Prediction and Resource Optimization Platform. It's an AI-powered system that helps predict student enrollment trends and optimize resource allocation for better planning and decision-making.",
                "category": "general"
            },
            "what is this app": {
                "answer": "This platform provides three main features: 1) EDA Dashboard for exploring enrollment data, 2) AI-powered enrollment growth predictions, and 3) Resource optimization tools to help with staffing and budget planning at University of Ibadan.",
                "category": "general"
            },
            "what is this platform": {
                "answer": "This is a comprehensive analytics platform for University of Ibadan that combines data analysis, machine learning predictions, and optimization algorithms to forecast enrollment and plan resources effectively.",
                "category": "general"
            },
            "about this app": {
                "answer": "The UI Enrollment Platform helps university administrators make data-driven decisions by forecasting enrollment trends, predicting graduation rates, and recommending optimal resource allocation strategies.",
                "category": "general"
            },
            "what is enrollment": {
                "answer": "Enrollment refers to the number of students admitted and registered at the university. This platform predicts enrollment growth rates based on historical data, economic indicators, faculty capacity, and policy factors.",
                "category": "general"
            },
            "what is enrollment prediction": {
                "answer": "Enrollment prediction uses machine learning to forecast how many students will enroll in future years. The system analyzes factors like GDP growth, unemployment rates, budget per student, and historical trends to make 1-year projections with uncertainty ranges.",
                "category": "general"
            },
            "what is prediction": {
                "answer": "Prediction in this app refers to forecasting future enrollment numbers and graduation rates using AI models trained on historical University of Ibadan data. Predictions include uncertainty ranges to show confidence levels.",
                "category": "general"
            },
            "explain predictions": {
                "answer": "The platform makes two types of predictions: 1) Enrollment Growth Rate - how much enrollment will change next year (e.g., +5.2%), and 2) Graduation Rate - what percentage of students are expected to graduate based on current resources.",
                "category": "general"
            },
            
            # APP USAGE
            "how do i use this app": {
                "answer": "Navigate using the sidebar: Start with the EDA Dashboard to explore data, then use the Prediction Tool to forecast enrollment and optimize resources. Select your faculty, input parameters, and get AI-powered predictions.",
                "category": "usage"
            },
            "how to navigate": {
                "answer": "Use the sidebar on the left: 🏠 Home for overview, 📊 EDA Dashboard for data exploration, 🎯 Prediction Tool for forecasts and optimization.",
                "category": "usage"
            },
            "what can this platform do": {
                "answer": "This platform provides: 1) Exploratory data analysis of enrollment trends, 2) AI-powered enrollment growth predictions, 3) Graduation rate forecasting, 4) Resource allocation optimization, and 5) Scenario simulations for planning.",
                "category": "usage"
            },
            
            # DATA REQUIREMENTS
            "what data do i need": {
                "answer": "For EDA: Upload CSV files with Year, Enrollment, Budget, Staff counts, and demographic data. For Predictions: Input current enrollment, budget, staff numbers, and economic indicators like GDP growth and unemployment rates.",
                "category": "data"
            },
            "what format should data be": {
                "answer": "Upload CSV files with columns like: YEAR, GENDER, FACULTY, DEPARTMENT, MODE_OF_ENTRY, ANNUAL_BUDGET_DEPT(₦), FAC_STAFF_COUNT_MALE, FAC_STAFF_COUNT_FEMALE, GDP_GROWTH_PERCENTAGE, UNEMPLOYMENT_RATE_PERCENTAGE.",
                "category": "data"
            },
            "csv upload not working": {
                "answer": "Try these solutions: 1) Re-save your file as 'CSV UTF-8' in Excel, 2) Use the 'Paste CSV Data' option instead, 3) Use 'Sample Data' to test the system, 4) Check file size is under 200MB, 5) Remove special characters from your data.",
                "category": "data"
            },
            "sample data": {
                "answer": "Click 'Use Sample Data' in the EDA Dashboard to load pre-configured enrollment data. This helps you explore features without uploading your own data first.",
                "category": "data"
            },
            
            # ENROLLMENT PREDICTIONS
            "how accurate are predictions": {
                "answer": "Enrollment predictions show an uncertainty range (typically ±5-8 percentage points). Accuracy depends on data quality, economic stability, and policy consistency. The model is trained on historical UI data and macroeconomic indicators.",
                "category": "prediction"
            },
            "what affects enrollment growth": {
                "answer": "Key factors include: GDP growth rate, unemployment rate, faculty demand, student-staff ratio, budget per student, strike duration, hostel availability, and Post-UTME cut-off marks. Economic conditions have the strongest impact.",
                "category": "prediction"
            },
            "how far ahead can i predict": {
                "answer": "The system provides 1-year enrollment projections. For longer-term planning, you can run multiple scenarios with different parameter assumptions.",
                "category": "prediction"
            },
            "what is uncertainty range": {
                "answer": "The uncertainty range (±X pp) shows the confidence interval for predictions. For example, a 15% ±5pp prediction means enrollment growth could be between 10-20%. Larger ranges indicate higher uncertainty.",
                "category": "prediction"
            },
            
            # GRADUATION RATES
            "graduation rate prediction": {
                "answer": "The system predicts graduation rates based on student-staff ratio, budget per student, faculty size, and resource allocation. Rates are adjusted for small faculties and compared against the national average (65%) and NUC target (85%).",
                "category": "graduation"
            },
            "how to improve graduation rate": {
                "answer": "Key strategies: 1) Reduce student-staff ratio below 20:1, 2) Increase budget per student above ₦60,000, 3) Hire additional qualified lecturers, 4) Minimize academic disruptions (strikes), 5) Use the optimization tool to find optimal resource allocation.",
                "category": "graduation"
            },
            
            # RESOURCE OPTIMIZATION
            "how does optimization work": {
                "answer": "The optimizer uses differential evolution to find the best balance of staff hiring and budget allocation. It maximizes graduation rates while staying within budget limits, maintaining acceptable student-staff ratios, and achieving gender balance targets (30-50% female staff).",
                "category": "optimization"
            },
            "what resources can be optimized": {
                "answer": "The system optimizes: 1) Number of new lecturers to hire (male/female split), 2) Budget allocation across salaries and infrastructure, 3) Timing of hires, 4) Resource distribution to meet graduation rate targets.",
                "category": "optimization"
            },
            "optimization parameters": {
                "answer": "Set three key parameters: 1) Maximum Additional Budget (available funds), 2) Target Graduation Rate (desired outcome, 75-95%), 3) Maximum Student-Staff Ratio (quality threshold, typically 15-25:1).",
                "category": "optimization"
            },
            "how long does optimization take": {
                "answer": "Optimization typically runs for 10-30 seconds. The algorithm evaluates thousands of possible solutions to find the best resource allocation strategy.",
                "category": "optimization"
            },
            
            # SCENARIO SIMULATION
            "what is scenario simulation": {
                "answer": "Scenario simulation lets you test 'what-if' situations. Adjust the number of additional lecturers and budget increase percentage to see projected impacts on student-staff ratio, budget per student, and graduation rate.",
                "category": "simulation"
            },
            "how to use simulation": {
                "answer": "In the Prediction Tool, scroll to 'Scenario Simulation'. Use sliders to set additional lecturers (0-500) and budget increase (-20% to +100%). Results update immediately showing projected metrics.",
                "category": "simulation"
            },
            
            # FACULTIES AND DEPARTMENTS
            "which faculties are included": {
                "answer": "All 16 UI faculties: Agriculture, Arts, Basic Medical Sciences, Clinical Sciences, Dentistry, Education, Environmental Design, Law, Pharmacy, Public Health, Renewable Natural Resources, Science, Social Sciences, Technology, and Veterinary Medicine.",
                "category": "structure"
            },
            "how to select department": {
                "answer": "Currently, predictions are made at the faculty level. Select your faculty from the sidebar dropdown in the Prediction Tool. Department-level analysis can be done in the EDA Dashboard if your uploaded data includes department information.",
                "category": "structure"
            },
            
            # ECONOMIC INDICATORS
            "what are economic indicators": {
                "answer": "GDP Growth Rate measures economic expansion (typically -5% to +10% in Nigeria). Unemployment Rate reflects job market conditions (0-40%). Both significantly influence enrollment decisions as higher GDP and lower unemployment correlate with increased enrollment.",
                "category": "economics"
            },
            "how to set economic indicators": {
                "answer": "In the Prediction Tool sidebar, adjust: 1) GDP Growth Rate slider (current Nigerian average ~2.5%), 2) Unemployment Rate slider (current ~20%), 3) Use recent official statistics from NBS for accuracy.",
                "category": "economics"
            },
            
            # REPORTS AND EXPORTS
            "how to download results": {
                "answer": "After running optimization, click '📄 Download Detailed Implementation Plan (CSV)' to export a comprehensive report including hiring plan, budget breakdown, yearly implementation schedule, and recommendations.",
                "category": "export"
            },
            "what's in the export": {
                "answer": "The CSV export contains: Summary metrics, year-by-year implementation plan, budget breakdown, immediate actions, long-term strategies, and constraint satisfaction analysis.",
                "category": "export"
            },
            
            # TROUBLESHOOTING
            "models not loading": {
                "answer": "Ensure all 5 .pkl model files are in the 'models' folder: ui_enrollment_features.pkl, ui_enrollment_prediction_model.pkl, ui_resource_allocation_model.pkl, ui_resource_features.pkl, ui_system_metadata.pkl.",
                "category": "troubleshooting"
            },
            "predictions seem wrong": {
                "answer": "Check: 1) Input parameters are realistic, 2) Faculty selection is correct, 3) Economic indicators match current conditions, 4) Current enrollment and staff numbers are accurate. Predictions include uncertainty ranges to account for variability.",
                "category": "troubleshooting"
            },
            
            # PLANNING AND IMPLEMENTATION
            "how to use predictions for planning": {
                "answer": "Use predictions to: 1) Forecast infrastructure needs, 2) Plan recruitment cycles, 3) Request budget allocations, 4) Set admission targets, 5) Prepare for capacity expansion. Export optimization results for administrative presentations.",
                "category": "planning"
            },
            "immediate vs long-term actions": {
                "answer": "Immediate actions (Year 1) focus on hiring, budget allocation, and orientation. Long-term strategies address sustained growth, quality assurance, faculty development, and infrastructure planning over 3-5 years.",
                "category": "planning"
            },
        }
        
        # Initialize session state
        if 'ui_chat_history' not in st.session_state:
            st.session_state.ui_chat_history = []
        if 'ui_chat_context' not in st.session_state:
            st.session_state.ui_chat_context = {}
    
    def find_predefined_answer(self, question: str) -> Optional[Dict[str, str]]:
        """Search for predefined answers using improved keyword matching."""
        question_lower = question.lower().strip()
        
        # Remove question marks and extra spaces
        question_lower = question_lower.replace('?', '').strip()
        
        # Exact match first
        if question_lower in self.predefined_qa:
            return self.predefined_qa[question_lower]
        
        # Check if the full key phrase appears in the question (most accurate)
        for key, value in self.predefined_qa.items():
            if key in question_lower:
                return value
        
        # Keyword matching with better scoring
        best_match = None
        best_score = 0
        min_threshold = 3  # Require at least 3 matching words
        
        question_words = set(question_lower.split())
        
        # Filter out common words that shouldn't count
        stop_words = {'is', 'are', 'the', 'a', 'an', 'what', 'how', 'why', 'when', 
                      'where', 'can', 'do', 'does', 'this', 'that', 'i', 'my', 'me'}
        question_words = question_words - stop_words
        
        for key, value in self.predefined_qa.items():
            key_words = set(key.split()) - stop_words
            
            # Calculate overlap
            overlap = len(question_words.intersection(key_words))
            
            # Calculate match percentage (more important for short questions)
            if len(question_words) > 0:
                match_percentage = overlap / len(question_words)
            else:
                match_percentage = 0
            
            # Scoring: prioritize high match percentage and overlap
            score = overlap + (match_percentage * 2)
            
            # Require minimum overlap AND good match percentage
            if score > best_score and overlap >= min(min_threshold, len(key_words)):
                # Additional check: avoid matching if key is much longer than question
                # (e.g., "prediction" shouldn't match "what is uncertainty range")
                if len(key_words) <= len(question_words) * 2:
                    best_score = score
                    best_match = value
        
        # Only return if we have a strong match
        if best_score >= min_threshold:
            return best_match
        
        return None
    
    def _generate_fallback_response(self, question: str) -> str:
        """Generate helpful fallback response when no match is found."""
        
        # Analyze question to provide relevant suggestions
        question_lower = question.lower()
        
        # Check for topic keywords
        if any(word in question_lower for word in ['data', 'upload', 'csv', 'file', 'format']):
            return """**Data & Upload Help:** I can help with data-related questions! Try asking:
            
- "What data do I need?"
- "What format should data be?"
- "CSV upload not working"
- "How to use sample data?"

Or navigate to the **EDA Dashboard** to upload and explore your data."""
        
        elif any(word in question_lower for word in ['predict', 'forecast', 'enrollment', 'growth']):
            return """**Enrollment Predictions:** I can help with prediction questions! Try asking:

- "How accurate are predictions?"
- "What affects enrollment growth?"
- "How far ahead can I predict?"
- "What is uncertainty range?"

Or go to the **Prediction Tool** to generate forecasts."""
        
        elif any(word in question_lower for word in ['optimize', 'resource', 'staff', 'budget', 'hire']):
            return """**Resource Optimization:** I can help with optimization! Try asking:

- "How does optimization work?"
- "What resources can be optimized?"
- "How to improve graduation rate?"
- "Optimization parameters"

Or use the **Prediction Tool** and scroll to the optimization section."""
        
        elif any(word in question_lower for word in ['faculty', 'faculties', 'department']):
            return """**University Structure:** I can help with organizational questions! Try asking:

- "Which faculties are included?"
- "How to select department?"

**Available Faculties:** Agriculture, Arts, Basic Medical Sciences, Clinical Sciences, Dentistry, Education, Environmental Design, Law, Pharmacy, Public Health, Renewable Natural Resources, Science, Social Sciences, Technology, Veterinary Medicine."""
        
        elif any(word in question_lower for word in ['graduation', 'graduate', 'rate']):
            return """**Graduation Rates:** I can help with graduation-related questions! Try:

- "Graduation rate prediction"
- "How to improve graduation rate?"

The system predicts graduation rates based on student-staff ratios, budget per student, and resource allocation."""
        
        elif any(word in question_lower for word in ['error', 'not working', 'problem', 'issue', 'fix']):
            return """**Troubleshooting:** I can help! Try asking:

- "Models not loading"
- "CSV upload not working"
- "Predictions seem wrong"

Or describe your specific issue and I'll try to help!"""
        
        else:
            # General fallback
            return """I'm here to help with University of Ibadan enrollment predictions and resource optimization! 

**Popular topics I can help with:**

📊 **App Usage:** "How do I use this app?"
📁 **Data:** "What data do I need?"
🎯 **Predictions:** "How accurate are predictions?"
💰 **Optimization:** "How does optimization work?"
🏛️ **Faculties:** "Which faculties are included?"

**Or try:**
- Click the quick question buttons below
- Ask about a specific feature you see on the page
- Describe what you're trying to do

What would you like to know?"""
    
    def get_ai_response(self, question: str, context: Dict[str, Any]) -> str:
        """
        Generate AI-powered response using context from the app.
        
        To enable AI responses, integrate with Anthropic Claude or OpenAI:
        
        import anthropic
        client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system="You are a helpful assistant for University of Ibadan's enrollment prediction system...",
            messages=[{
                "role": "user",
                "content": f"Context: {context_str}\n\nQuestion: {question}"
            }]
        )
        return message.content[0].text
        """
        
        # Build context string
        context_info = []
        
        if context.get('current_page'):
            context_info.append(f"User is on: {context.get('current_page')}")
        
        if context.get('selected_faculty'):
            context_info.append(f"Selected faculty: {context.get('selected_faculty')}")
        
        if context.get('current_enrollment'):
            context_info.append(f"Current enrollment: {context.get('current_enrollment'):,} students")
        
        if context.get('projected_enrollment'):
            context_info.append(f"Projected enrollment: {context.get('projected_enrollment'):,} students")
        
        if context.get('predicted_growth_rate'):
            context_info.append(f"Predicted growth: {context.get('predicted_growth_rate'):.1f}%")
        
        if context.get('graduation_rate'):
            context_info.append(f"Expected graduation rate: {context.get('graduation_rate'):.1f}%")
        
        context_str = ". ".join(context_info) if context_info else "No specific context available"
        
        # Placeholder response - replace with actual AI API call
        return f"**AI Assistant:** I understand you're asking about '{question}'. Based on the current state ({context_str}), I'm here to help with enrollment predictions and resource optimization for University of Ibadan. [To enable full AI responses, add your API key to .streamlit/secrets.toml]"
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to chat history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        st.session_state.ui_chat_history.append(message)
    
    def save_chat_history(self):
        """Save chat history to a downloadable JSON file."""
        chat_data = {
            "university": "University of Ibadan",
            "export_date": datetime.now().isoformat(),
            "context": st.session_state.ui_chat_context,
            "messages": st.session_state.ui_chat_history
        }
        return json.dumps(chat_data, indent=2)
    
    def render(self, app_context: Optional[Dict[str, Any]] = None, compact: bool = False):
        """
        Render the chatbox UI.
        
        Args:
            app_context: Dictionary containing app state and data
            compact: If True, render in compact mode for sidebar
        """
        # Update context
        if app_context:
            st.session_state.ui_chat_context.update(app_context)
        
        if not compact:
            st.markdown("### 💬 Q&A Assistant")
            st.markdown("Ask me about enrollment predictions, resource optimization, or how to use the platform!")
        else:
            st.markdown("#### 💬 Q&A")
        
        # Chat controls
        if not compact:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                if st.button("🗑️ Clear Chat", key="clear_chat_main"):
                    st.session_state.ui_chat_history = []
                    st.rerun()
            
            with col2:
                if st.session_state.ui_chat_history:
                    chat_json = self.save_chat_history()
                    st.download_button(
                        label="📥 Export",
                        data=chat_json,
                        file_name=f"ui_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        # Chat display area
        chat_container = st.container()
        
        with chat_container:
            # Limit displayed messages in compact mode
            messages_to_show = st.session_state.ui_chat_history[-5:] if compact else st.session_state.ui_chat_history
            
            if not messages_to_show and not compact:
                st.info("👋 Hi! I'm your UI Enrollment Assistant. Ask me anything about enrollment predictions, resource optimization, or using the platform.")
            
            for msg in messages_to_show:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        # Chat input
        user_question = st.chat_input("Ask about enrollment predictions, data, or resource optimization...")
        
        if user_question:
            # Add user message
            self.add_message("user", user_question)
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Try predefined answers first
                    predefined = self.find_predefined_answer(user_question)
                    
                    if predefined:
                        response = f"**{predefined['category'].title()}:** {predefined['answer']}"
                        metadata = {"source": "predefined", "category": predefined['category']}
                    else:
                        # Check if we should try AI or provide helpful fallback
                        if hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
                            # AI is configured, use it
                            response = self.get_ai_response(user_question, st.session_state.ui_chat_context)
                            metadata = {"source": "ai"}
                        else:
                            # No AI configured, provide helpful fallback
                            response = self._generate_fallback_response(user_question)
                            metadata = {"source": "fallback"}
                    
                    st.markdown(response)
                    self.add_message("assistant", response, metadata)
            
            st.rerun()
        
        # Quick action buttons
        if not compact:
            st.markdown("---")
            st.markdown("**Quick Questions:**")
            
            quick_questions = [
                "How do I use this app?",
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
                            self.add_message("assistant", response, {"source": "predefined", "category": predefined['category']})
                        st.rerun()


# Convenience function for easy integration
def render_ui_chatbox(app_context: Optional[Dict[str, Any]] = None, compact: bool = False):
    """
    Convenience function to render the chatbox.
    
    Usage:
        from ui_enrollment_chatbox import render_ui_chatbox
        
        # In your app
        render_ui_chatbox({
            'current_page': 'Prediction Tool',
            'selected_faculty': 'SCIENCE',
            'current_enrollment': 4000
        })
    """
    if 'ui_chatbox' not in st.session_state:
        st.session_state.ui_chatbox = UIEnrollmentChatbox()
    
    st.session_state.ui_chatbox.render(app_context, compact)
