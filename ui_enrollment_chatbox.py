"""
Q&A Chatbox for University of Ibadan Enrollment Prediction System
Fixed version with reliable question matching - 2025 improved edition
"""
import streamlit as st
import json
from datetime import datetime
from typing import Optional, Dict, Any, List


class UIEnrollmentChatbox:
    """
    Reliable Q&A chatbox for University of Ibadan's enrollment prediction platform.
    """

    def __init__(self):
        """Initialize with carefully ordered, specific-first Q&A patterns."""
        self.qa_patterns = [
            # ──────────────────────────────────────────────────────────────
            # MOST SPECIFIC QUESTIONS FIRST (prevents overlap/false matches)
            # ──────────────────────────────────────────────────────────────
            (
                ["what affects enrollment", "factors affecting enrollment", "what factors affect enrollment", "enrollment factors"],
                {
                    "answer": "Key factors include: GDP growth rate, unemployment rate, faculty demand (especially high-demand fields like Medicine and Science), student-staff ratio, budget per student, strike duration, hostel allocation probability, and departmental Post-UTME cut-off marks. Internal institutional factors (staffing, budget, accommodation) usually have stronger influence than short-term economic changes.",
                    "category": "prediction"
                }
            ),

            (
                ["uncertainty range", "confidence interval", "prediction range", "what is uncertainty range"],
                {
                    "answer": "The uncertainty range (±X pp) shows the confidence interval for the prediction. For example, a 15% ±5pp prediction means enrollment growth could realistically fall between 10% and 20%. Larger ranges indicate higher uncertainty (e.g., due to volatile economic conditions or limited historical data).",
                    "category": "prediction"
                }
            ),

            # ──────────────────────────────────────────────────────────────
            # Medium specificity
            # ──────────────────────────────────────────────────────────────
            (
                ["what is enrollment prediction", "enrollment forecasting", "what is enrollment forecasting"],
                {
                    "answer": "Enrollment prediction uses machine learning models (primarily Random Forest) to forecast how many students will enroll in the coming year. It analyzes 10+ years of UI data plus external factors like GDP growth, unemployment, budget, staffing, strikes, hostel availability, and Post-UTME policies to generate 1-year projections with uncertainty ranges.",
                    "category": "prediction"
                }
            ),

            (
                ["what is prediction", "what are predictions", "explain prediction", "predictions meaning"],
                {
                    "answer": "Predictions are AI-generated forecasts of future outcomes. This app provides two main types: 1) Enrollment Growth Rate — how much enrollment is expected to change next year, and 2) Graduation Rate — the expected percentage of students who will complete their programs based on current resources and trends.",
                    "category": "general"
                }
            ),

            (
                ["what is enrollment", "define enrollment", "enrollment meaning", "what does enrollment mean"],
                {
                    "answer": "Enrollment is the total number of students admitted and registered at the University of Ibadan for a given academic year (full-time undergraduates). This platform predicts future enrollment trends and helps optimize resources (staff, budget, hostels) to match expected numbers.",
                    "category": "general"
                }
            ),

            # ──────────────────────────────────────────────────────────────
            # App usage & structure
            # ──────────────────────────────────────────────────────────────
            (
                ["how do i use", "how to use this app", "how to get started", "getting started"],
                {
                    "answer": "1. Go to **EDA Dashboard** to explore & upload data\n2. Switch to **Prediction Tool** in the sidebar\n3. Select your faculty\n4. Adjust current enrollment, budget, staff numbers, economic indicators\n5. Click prediction & optimization buttons to see forecasts and recommendations.",
                    "category": "usage"
                }
            ),

            (
                ["what is this", "what is this app", "what is this platform", "tell me about this app"],
                {
                    "answer": "This is the **University of Ibadan Enrollment Prediction and Resource Optimization Platform**. It uses machine learning to forecast student enrollment trends and recommend optimal staff hiring, budget allocation, and resource planning to improve graduation rates and operational efficiency.",
                    "category": "general"
                }
            ),

            # ──────────────────────────────────────────────────────────────
            # Data & upload
            # ──────────────────────────────────────────────────────────────
            (
                ["what data", "data needed", "data required", "data format", "csv columns"],
                {
                    "answer": "Upload CSV with these key columns:\n• YEAR\n• GENDER\n• FACULTY\n• DEPARTMENT\n• MODE_OF_ENTRY\n• ENROLED\n• ANNUAL_BUDGET_DEPT(₦)\n• FAC_STAFF_COUNT_MALE / FEMALE\n• HOSTEL_ALLOCATION_PROBABILITY\n• STRIKE_DURATION_MONTHS\n• GDP_GROWTH_PERCENTAGE\n• UNEMPLOYMENT_RATE_PERCENTAGE\n• DEPT_POST_UTME_CUT_OFF",
                    "category": "data"
                }
            ),

            (
                ["csv not working", "upload fails", "cant upload", "upload error", "file upload problem"],
                {
                    "answer": "Quick fixes:\n1. In Excel → Save As → 'CSV UTF-8'\n2. Try **Paste CSV Data** option instead\n3. Use **Sample Data** button to test\n4. File < 200 MB\n5. Remove special characters / extra commas",
                    "category": "troubleshooting"
                }
            ),

            # ──────────────────────────────────────────────────────────────
            # Optimization & graduation
            # ──────────────────────────────────────────────────────────────
            (
                ["how optimization works", "optimization algorithm", "how does optimization work"],
                {
                    "answer": "The optimizer uses differential evolution to search thousands of possible hiring + budget scenarios. It tries to maximize graduation rate while respecting your constraints: max budget, target graduation rate, acceptable student-staff ratio (15–25:1), and gender balance goals.",
                    "category": "optimization"
                }
            ),

            (
                ["how to improve graduation", "increase graduation rate", "improve graduation rate"],
                {
                    "answer": "Top recommendations:\n1. Reduce student-staff ratio (aim <20:1)\n2. Increase budget per student (target >₦60,000)\n3. Hire more qualified lecturers\n4. Minimize academic disruptions\n5. Use the optimization tool to find the most cost-effective combination.",
                    "category": "graduation"
                }
            ),

            # ──────────────────────────────────────────────────────────────
            # General / catch-all at the bottom
            # ──────────────────────────────────────────────────────────────
            (
                ["how accurate", "accuracy", "prediction accuracy", "how reliable"],
                {
                    "answer": "Models achieve strong performance (R² ≈ 0.93 on historical data). Predictions include realistic uncertainty ranges (±5–8 pp typical). Accuracy is best when inputs (budget, staff, economic indicators) are up-to-date and realistic.",
                    "category": "prediction"
                }
            ),

            (
                ["faculties", "which faculties", "list of faculties"],
                {
                    "answer": "All 16 UI faculties are supported:\nAgriculture, Arts, Basic Medical Sciences, Clinical Sciences, Dentistry, Education, Environmental Design & Management, Law, Pharmacy, Public Health, Renewable Natural Resources, Science, Social Sciences, Technology, Veterinary Medicine, and more.",
                    "category": "structure"
                }
            ),
        ]

        # Session state initialization
        if 'ui_chat_history' not in st.session_state:
            st.session_state.ui_chat_history = []
        if 'ui_chat_context' not in st.session_state:
            st.session_state.ui_chat_context = {}

    def find_predefined_answer(self, question: str) -> Optional[Dict[str, str]]:
        """
        Stricter matching: specific → general, cleaned question, overlap check
        """
        q = question.lower().strip().rstrip('?.!').replace('  ', ' ')

        # Remove common prefixes for cleaner matching
        cleaned = q.replace("what is ", "").replace("what are ", "").replace("explain ", "").replace("tell me ", "").strip()

        for patterns, answer_data in self.qa_patterns:
            for pattern in patterns:
                # Pattern must appear in original or cleaned version
                if pattern in q or pattern in cleaned:
                    # Require reasonable word overlap for short patterns
                    pat_words = set(pattern.split())
                    q_words = set(cleaned.split())
                    overlap = len(pat_words.intersection(q_words))
                    if len(pat_words) > 0 and overlap / len(pat_words) >= 0.6:
                        # print(f"DEBUG MATCH: '{question}' → '{pattern}'")  # uncomment for debugging
                        return answer_data

        return None

    def _generate_fallback_response(self, question: str) -> str:
        q = question.lower()
        if any(w in q for w in ['data', 'upload', 'csv', 'file', 'format']):
            return """**Data / Upload help**  
Try asking:  
• "What data do I need?"  
• "CSV not working"  
Or go to **EDA Dashboard** → Upload section"""

        if any(w in q for w in ['predict', 'forecast', 'growth', 'enroll', 'accuracy']):
            return """**Predictions help**  
Try asking:  
• "What is enrollment prediction?"  
• "How accurate are predictions?"  
• "What affects enrollment?"  
Go to **Prediction Tool** → Generate forecast"""

        if any(w in q for w in ['optimi', 'resource', 'staff', 'budget', 'hire', 'graduation']):
            return """**Optimization / Graduation help**  
Try asking:  
• "How does optimization work?"  
• "How to improve graduation rate?"  
Use **Prediction Tool** → Optimization section"""

        return """**I'm here to help!**  
Popular questions:  
• "What is this app?"  
• "What data do I need?"  
• "How accurate are predictions?"  
• "How does optimization work?"  
• "Which faculties are included?"  

What would you like to know? 😊"""

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
            st.markdown("Ask anything about enrollment forecasting, resource optimization, or using the platform.")
        else:
            st.markdown("#### 💬 Q&A")

        # Clear & Export buttons
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

        # Chat messages
        chat_container = st.container()
        with chat_container:
            messages = st.session_state.ui_chat_history[-6:] if compact else st.session_state.ui_chat_history
            if not messages and not compact:
                st.info("👋 Hi! I'm your UI Enrollment Assistant.\nAsk me about predictions, data, optimization, or how to use the app.")

            for msg in messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Input
        user_question = st.chat_input("Ask about enrollment, predictions, optimization...")

        if user_question:
            self.add_message("user", user_question)

            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    match = self.find_predefined_answer(user_question)
                    if match:
                        response = f"**{match['category'].title()}**  \n{match['answer']}"
                        meta = {"source": "predefined", "category": match['category']}
                    else:
                        response = self._generate_fallback_response(user_question)
                        meta = {"source": "fallback"}

                    st.markdown(response)
                    self.add_message("assistant", response, meta)

            st.rerun()

        # Quick questions buttons
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
                            resp = f"**{match['category'].title()}**  \n{match['answer']}"
                            self.add_message("assistant", resp, {"source": "predefined"})
                        st.rerun()


def render_ui_chatbox(app_context: Optional[Dict[str, Any]] = None, compact: bool = False):
    if 'ui_chatbox' not in st.session_state:
        st.session_state.ui_chatbox = UIEnrollmentChatbox()
    st.session_state.ui_chatbox.render(app_context, compact)
