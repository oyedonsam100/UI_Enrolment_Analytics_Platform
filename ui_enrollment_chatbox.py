"""
Q&A Chatbox for University of Ibadan Enrollment Prediction System
UPDATED VERSION - no category labels, cleaner & more direct answers
"""
import streamlit as st
import json
from datetime import datetime
from typing import Optional, Dict, Any, List


class UIEnrollmentChatbox:
    def __init__(self):
        """Initialize with clean, direct answers - no category prefixes"""
        self.qa_patterns = [
            # Exact high-priority matches first
            (
                ["what is enrollment", "define enrollment", "enrollment meaning", "what does enrollment mean"],
                {
                    "answer": "Enrollment is the total number of full-time undergraduate students admitted and registered at the University of Ibadan in a given academic year. This platform forecasts future enrollment trends and helps optimize staff, budget, and hostel resources accordingly."
                }
            ),
            (
                ["what is enrollment prediction", "enrollment forecasting", "what is enrollment forecasting", "enrollment prediction meaning"],
                {
                    "answer": "Enrollment prediction forecasts how many students will enroll at UI next year using machine learning (mainly Random Forest models). It combines 10+ years of historical UI data with factors like GDP growth, unemployment rate, departmental budget, staff numbers, strike duration, hostel availability, and Post-UTME cut-off marks to give 1-year projections with realistic uncertainty ranges."
                }
            ),
            (
                ["what is prediction", "what are predictions", "explain prediction", "predictions meaning"],
                {
                    "answer": "Predictions are AI-based forecasts of future university outcomes. This app gives two main types: 1) Enrollment growth rate — expected percentage change in student numbers next year, and 2) Graduation rate — the likely percentage of current students who will successfully complete their programs based on resources and trends."
                }
            ),
            (
                ["what affects enrollment", "factors affecting enrollment", "what factors affect enrollment", "enrollment factors"],
                {
                    "answer": "The main factors are: GDP growth rate, national unemployment rate, faculty/department popularity (especially high-demand fields like Medicine and Science), student-to-staff ratio, budget per student, duration of academic strikes, hostel allocation probability, and departmental Post-UTME cut-off marks. Internal university factors (staffing, budget, accommodation) usually have a stronger influence than short-term economic changes."
                }
            ),
            (
                ["uncertainty range", "confidence interval", "prediction range", "what is uncertainty range"],
                {
                    "answer": "The uncertainty range (e.g. ±5 pp) is the confidence interval around the prediction. A forecast of 15% ±5 pp means actual growth could reasonably be between 10% and 20%. Wider ranges appear when economic conditions are volatile or historical data is limited."
                }
            ),

            # Other patterns (kept clean and direct)
            (
                ["what is this", "what is this app", "what is this platform"],
                {
                    "answer": "This is the University of Ibadan Enrollment Prediction and Resource Optimization Platform. It uses machine learning to forecast student enrollment trends and recommend the best ways to allocate staff, budget, and other resources."
                }
            ),
            (
                ["how do i use", "how to use this app", "how to get started"],
                {
                    "answer": "1. Visit the EDA Dashboard to upload and explore data\n2. Go to Prediction Tool in the sidebar\n3. Choose your faculty\n4. Enter current enrollment, budget, staff numbers, and economic indicators\n5. Click the prediction and optimization buttons to see forecasts and recommendations."
                }
            ),
            (
                ["how accurate", "accuracy", "prediction accuracy"],
                {
                    "answer": "The models perform strongly (R² ≈ 0.93 on historical data). Forecasts include realistic uncertainty ranges, usually ±5–8 percentage points. Best results come from using up-to-date and accurate input values."
                }
            ),
            (
                ["how optimization works", "how does optimization work"],
                {
                    "answer": "The optimizer uses differential evolution to test thousands of possible combinations of new staff hires and budget allocations. It finds the plan that maximizes graduation rate while respecting your budget limit, target graduation goal, and acceptable student-to-staff ratio (usually 15–25:1)."
                }
            ),
            (
                ["how to improve graduation", "increase graduation rate"],
                {
                    "answer": "To raise graduation rates:\n1. Lower the student-to-staff ratio (ideally below 20:1)\n2. Increase budget per student (aim above ₦60,000)\n3. Hire additional qualified lecturers\n4. Reduce disruptions like strikes\n5. Run the optimization tool to find the most efficient combination under your constraints."
                }
            ),
            # Add your remaining patterns here in similar clean style...
        ]

        if 'ui_chat_history' not in st.session_state:
            st.session_state.ui_chat_history = []
        if 'ui_chat_context' not in st.session_state:
            st.session_state.ui_chat_context = {}

    def find_predefined_answer(self, question: str) -> Optional[Dict[str, str]]:
        original = question.lower().strip()
        q = original.rstrip('?.!').replace('  ', ' ')
        
        cleaned = q.replace("what is ", "").replace("what are ", "").replace("explain ", "").replace("tell me ", "").replace("what does ", "").strip()

        # Exact match first
        for patterns, answer_data in self.qa_patterns:
            for pattern in patterns:
                if pattern == q or pattern == cleaned:
                    return answer_data

        # Strong substring + overlap
        q_words = set(cleaned.split())
        for patterns, answer_data in self.qa_patterns:
            for pattern in patterns:
                if pattern in q or pattern in cleaned:
                    pat_words = set(pattern.split())
                    if len(pat_words) > 0:
                        overlap = len(pat_words.intersection(q_words))
                        if overlap / len(pat_words) >= 0.75:
                            return answer_data

        return None

    def _generate_fallback_response(self, question: str) -> str:
        q = question.lower().strip()

        if "enrollment" in q:
            return """Enrollment refers to the total number of full-time undergraduate students registered at UI in a given year.

You might also want to ask:
• "What is enrollment prediction?"
• "What affects enrollment?"
• "How accurate are predictions?" """

        return """Sorry, I didn't catch that exactly.

Try one of these:
• "What is enrollment?"
• "What is enrollment prediction?"
• "What affects enrollment?"
• "How accurate are predictions?"
• "How does optimization work?"

What would you like to know?"""

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        st.session_state.ui_chat_history.append(message)

    def save_chat_history(self):
        chat_data = {
            "university": "University of Ibadan",
            "export_date": datetime.now().isoformat(),
            "messages": st.session_state.ui_chat_history
        }
        return json.dumps(chat_data, indent=2)

    def render(self, app_context: Optional[Dict[str, Any]] = None, compact: bool = False):
        if app_context:
            st.session_state.ui_chat_context.update(app_context)

        if not compact:
            st.markdown("### 💬 Q&A Assistant")
            st.markdown("Ask anything about enrollment forecasts, resource planning, or how to use the platform.")
        else:
            st.markdown("#### 💬 Q&A")

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
                        label="📥 Export Chat",
                        data=chat_json,
                        file_name=f"ui-chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )

        chat_container = st.container()
        with chat_container:
            messages = st.session_state.ui_chat_history[-6:] if compact else st.session_state.ui_chat_history
            if not messages and not compact:
                st.info("👋 Hi! I'm your UI Enrollment Assistant.\nAsk me about predictions, data, optimization, or how the app works.")

            for msg in messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        user_question = st.chat_input("Ask about enrollment, predictions, optimization...")

        if user_question:
            self.add_message("user", user_question)

            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    match = self.find_predefined_answer(user_question)
                    if match:
                        response = match['answer']
                    else:
                        response = self._generate_fallback_response(user_question)

                    st.markdown(response)
                    self.add_message("assistant", response)

            st.rerun()

        if not compact:
            st.markdown("---")
            st.markdown("**Quick Questions:**")
            quick = [
                "What is this app?",
                "What data do I need?",
                "How accurate are predictions?",
                "How does optimization work?"
            ]
            cols = st.columns(2)
            for i, q in enumerate(quick):
                with cols[i % 2]:
                    if st.button(q, key=f"qq_ui_{i}", use_container_width=True):
                        self.add_message("user", q)
                        match = self.find_predefined_answer(q)
                        if match:
                            resp = match['answer']
                            self.add_message("assistant", resp)
                        st.rerun()


def render_ui_chatbox(app_context: Optional[Dict[str, Any]] = None, compact: bool = False):
    if 'ui_chatbox' not in st.session_state:
        st.session_state.ui_chatbox = UIEnrollmentChatbox()
    st.session_state.ui_chatbox.render(app_context, compact)
