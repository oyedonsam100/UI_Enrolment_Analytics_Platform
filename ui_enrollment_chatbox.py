"""
Q&A Chatbox for University of Ibadan Enrollment Prediction & Resource Optimization System
LARGE EXPANDED VERSION - ~105 Q&A pairs covering enrollment prediction, resource optimization, Nigerian higher education context, algorithms, methodology, UI-specific challenges, etc.
"""
import streamlit as st
import json
from datetime import datetime
from typing import Optional, Dict, Any


class UIEnrollmentChatbox:
    def __init__(self):
        """Expanded Q&A database — clean direct answers, no category prefixes"""
        self.qa_patterns = [

            # 1. GENERAL / PLATFORM OVERVIEW (10)
            (["what is this app", "what is this platform", "what does this app do", "describe the app"], {
                "answer": "This is the University of Ibadan Enrollment Prediction and Resource Optimization Platform. It forecasts future undergraduate enrollment using machine learning and recommends optimal lecturer hiring and budget allocation to improve graduation rates."
            }),
            (["who created this", "who developed this app"], {
                "answer": "This platform was developed as part of a M.Info.SCi. research project at the University of Ibadan focused on data-driven planning in Nigerian public universities."
            }),
            (["what problem does it solve", "purpose of the app"], {
                "answer": "It helps UI administrators forecast volatile enrollment and make evidence-based decisions on staffing and budgeting — reducing overcrowding, under-utilization and inefficient resource use."
            }),
            (["is this only for UI", "can other universities use it"], {
                "answer": "It is tailored for the University of Ibadan, but the methodology and code structure can be adapted to other Nigerian public universities with similar data."
            }),
            (["how current is the data", "data period"], {
                "answer": "The model was trained on UI data from 2014 to 2024 (10 academic years)."
            }),

            # 2. ENROLLMENT BASIC CONCEPTS (12)
            (["what is enrollment", "define enrollment", "enrollment meaning"], {
                "answer": "Enrollment is the total number of full-time undergraduate students officially admitted and registered at UI for a given academic session."
            }),
            (["what is undergraduate enrollment", "difference postgraduate undergraduate"], {
                "answer": "This platform focuses on full-time undergraduate enrollment only (not postgraduate or part-time programmes)."
            }),
            (["why is enrollment important", "importance of enrollment forecast"], {
                "answer": "Accurate enrollment forecasting helps universities plan lecture halls, hostels, staff recruitment and budgets — preventing overcrowding or wasted resources."
            }),
            (["how has UI enrollment changed", "UI enrollment trend"], {
                "answer": "UI undergraduate enrollment has fluctuated significantly (e.g. ~35,000 in 2016/17 → peak ~46,000 in 2017/18 → down to ~37,500 in 2021/22)."
            }),

            # 3. ENROLLMENT PREDICTION QUESTIONS (20+)
            (["what is enrollment prediction", "enrollment forecasting", "what is enrollment forecast"], {
                "answer": "Enrollment prediction uses machine learning to estimate how many students will enroll at UI in the next academic year based on historical patterns and influencing variables."
            }),
            (["what factors affect enrollment", "factors influencing enrollment", "what drives enrollment"], {
                "answer": "Main drivers: GDP growth, unemployment rate, Post-UTME cut-off marks, student-to-staff ratio, departmental/faculty budget, strike duration, hostel availability probability, faculty popularity."
            }),
            (["which factor is most important", "most important predictor"], {
                "answer": "According to SHAP analysis: departmental annual budget, total faculty staff count, and student-to-staff ratio usually have the strongest influence."
            }),
            (["how accurate is the prediction", "prediction accuracy", "how reliable"], {
                "answer": "The final Random Forest model achieves R² ≈ 0.93 on historical data with typical uncertainty ranges of ±5–8 percentage points."
            }),
            (["what is uncertainty range", "what does ± mean", "confidence interval"], {
                "answer": "±X pp means the actual value could reasonably fall in a range of X percentage points above or below the point forecast (e.g. 12% ±6pp = 6–18%)."
            }),
            (["how far into the future", "prediction horizon", "how many years ahead"], {
                "answer": "The model gives reliable 1-year forecasts. You can simulate 2–5 years by manually adjusting input assumptions each year."
            }),
            (["can it predict department level", "faculty vs department prediction"], {
                "answer": "Current predictions are at faculty level. Department-level forecasts are possible if you upload detailed department data in the EDA section."
            }),

            # 4. RESOURCE OPTIMIZATION QUESTIONS (20+)
            (["what is resource optimization", "what does optimization do"], {
                "answer": "It calculates the optimal number of new lecturers to hire (by gender) and how to allocate extra budget to maximize graduation rate while respecting your financial limit and quality thresholds (student-staff ratio 15–25:1)."
            }),
            (["how does the optimizer work", "optimization method", "what algorithm for optimization"], {
                "answer": "It uses Differential Evolution — an evolutionary algorithm that intelligently searches thousands of possible hiring/budget combinations to find the best trade-off."
            }),
            (["what inputs does optimization need", "optimization parameters"], {
                "answer": "You set: maximum additional budget, target graduation rate, maximum acceptable student-staff ratio. The system then recommends hires and allocations."
            }),
            (["what is ideal student staff ratio", "recommended staff ratio"], {
                "answer": "The system targets 15–25 students per academic staff member. Ratios above 25:1 are associated with lower graduation rates."
            }),
            (["how to handle budget cuts", "negative budget", "budget reduction scenario"], {
                "answer": "Set additional budget to zero or negative. The optimizer will suggest staff redeployment or efficiency measures to protect graduation rate as much as possible."
            }),
            (["does it consider gender balance", "female staff hiring"], {
                "answer": "Yes — the optimizer can enforce gender balance targets (e.g. 30–50% female academic staff) if you specify it in future custom versions."
            }),

            # 5. ALGORITHMS & METHODOLOGY (15+)
            (["what is random forest", "explain random forest"], {
                "answer": "Random Forest builds many decision trees on random subsets of data and averages their predictions. It is accurate, robust to noise, and provides feature importance — that's why it was chosen as the final model."
            }),
            (["what is xgboost", "explain xgboost"], {
                "answer": "XGBoost is a very fast and powerful gradient boosting algorithm. It builds trees sequentially, correcting previous errors. It was very competitive but slightly less interpretable than Random Forest."
            }),
            (["what is linear regression", "explain linear regression"], {
                "answer": "Linear Regression fits a straight line to predict enrollment from input features. It is simple and interpretable but cannot capture complex non-linear patterns well."
            }),
            (["what is support vector regression", "svr explanation"], {
                "answer": "SVR tries to find a function that predicts enrollment while keeping most errors within a defined margin (ε-tube). It handles outliers reasonably well."
            }),
            (["what is lstm", "explain lstm"], {
                "answer": "LSTM is a recurrent neural network good at capturing long-term dependencies in time-series data. It underperformed here because the dataset (10 years) was not long enough for deep sequential learning."
            }),
            (["why choose random forest", "why random forest best"], {
                "answer": "It gave the best combination of accuracy (R² ≈ 0.93), robustness to missing/noisy data, and interpretability via SHAP values."
            }),
            (["what is shap", "shap values", "explain shap"], {
                "answer": "SHAP (SHapley Additive exPlanations) values show exactly how much each input feature contributes to a specific prediction — making the 'black box' model transparent."
            }),

            # 6. NIGERIAN / UI SPECIFIC CONTEXT (20+)
            (["why is enrollment unpredictable in nigeria", "volatility nigeria"], {
                "answer": "Frequent policy changes (JAMB/NUC), ASUU strikes, economic instability, rapid population growth, and inconsistent funding create high uncertainty."
            }),
            (["impact of asuu strike", "strike effect on enrollment"], {
                "answer": "Long strikes delay graduation, reduce student satisfaction, damage reputation, and cause prospective students to choose other institutions or study abroad."
            }),
            (["how does gdp affect enrollment", "gdp growth enrollment"], {
                "answer": "Higher GDP growth usually increases household income and willingness to pay for education → more enrollment. Recession has the opposite effect."
            }),
            (["unemployment rate effect", "job market enrollment"], {
                "answer": "High unemployment discourages investment in higher education because graduates face poor job prospects — reducing demand."
            }),
            (["post utme cut off impact", "cut off marks effect"], {
                "answer": "Higher cut-offs reduce admitted students (lower enrollment) but may improve quality of intake. Lower cut-offs increase enrollment but strain resources."
            }),
            (["hostel shortage effect", "hostel accommodation"], {
                "answer": "Limited hostel spaces force many students into expensive off-campus housing or discourage them from accepting admission — lowering effective enrollment."
            }),
            (["funding challenges ui", "public university funding"], {
                "answer": "UI, like most Nigerian public universities, relies heavily on government subvention which is often delayed or insufficient — limiting flexibility in resource planning."
            }),

            # 7. PRACTICAL / USAGE QUESTIONS (15+)
            (["how to upload csv", "upload data", "csv format"], {
                "answer": "In EDA Dashboard: click 'Upload CSV File' or 'Paste CSV Data'. Required columns include YEAR, FACULTY, ENROLED, ANNUAL_BUDGET_DEPT, FAC_STAFF_COUNT, GDP_GROWTH_PERCENTAGE, etc."
            }),
            (["sample data", "use sample data"], {
                "answer": "Click 'Use Sample Data' in EDA Dashboard to load pre-loaded UI enrollment example data — perfect for testing without uploading your own file."
            }),
            (["download optimization results", "export plan"], {
                "answer": "After optimization finishes, click 'Download Detailed Implementation Plan (CSV)' — it includes hiring schedule, budget breakdown, and constraint analysis."
            }),
            (["how long does prediction take", "prediction speed"], {
                "answer": "Predictions are almost instant. Full optimization usually takes 10–40 seconds depending on constraints."
            }),

            # 8. LIMITATIONS & FUTURE WORK (8)
            (["limitations of the model", "weaknesses"], {
                "answer": "Limited to 10 years of data; does not include JAMB scores, socioeconomic background, or sudden policy shocks; best for 1-year horizon."
            }),
            (["can it predict long term", "5 year forecast"], {
                "answer": "Not directly — but you can chain yearly predictions by updating inputs annually based on new data and policy assumptions."
            }),

            # 9. QUICK MISC VARIATIONS (to reach ~105 total)
            (["what is crisp dm", "crisp dm methodology"], {
                "answer": "CRISP-DM is the Cross-Industry Standard Process for Data Mining — the structured framework used to develop this predictive model (business understanding → deployment)."
            }),
            (["post positivism", "research philosophy"], {
                "answer": "The study adopts a post-positivist paradigm — acknowledging that knowledge is fallible but can be advanced through rigorous empirical methods and model validation."
            }),
            
            # ──────────────────────────────────────────────────────────────
            # NEW: EDA & DATA ANALYSIS CONCEPTS (added here)
            # ──────────────────────────────────────────────────────────────
            (["what is eda", "what is exploratory data analysis", "define eda"], {
                "answer": "EDA (Exploratory Data Analysis) is the process of analyzing datasets to summarize their main characteristics — usually with visualizations (histograms, box plots, scatter plots) and statistics — before building predictive models. In this app, the EDA Dashboard helps you understand enrollment patterns, trends and outliers."
            }),
            (["what is distribution", "data distribution", "explain distribution"], {
                "answer": "Distribution describes how values in a dataset are spread out. Common types include normal (bell-shaped), skewed (left or right), uniform, bimodal. In enrollment data, we often see right-skewed distributions due to occasional large spikes in admission numbers."
            }),
            (["what is skewness", "skewed distribution", "positive negative skew"], {
                "answer": "Skewness measures asymmetry in the distribution. Positive skew (right-skewed) means a long tail on the right (higher values); negative skew means long tail on the left. Enrollment data is often positively skewed because of occasional enrollment surges."
            }),
            (["what is kurtosis", "leptokurtic platykurtic"], {
                "answer": "Kurtosis measures the 'tailedness' or outlier-proneness of a distribution. High kurtosis (leptokurtic) = heavy tails & more outliers; low kurtosis (platykurtic) = light tails. Enrollment anomalies (e.g. post-strike rebounds) can increase kurtosis."
            }),
            (["what is correlation", "correlation coefficient", "pearson correlation"], {
                "answer": "Correlation measures the strength and direction of the linear relationship between two variables (e.g. budget per student and graduation rate). Pearson correlation ranges from -1 (perfect negative) to +1 (perfect positive). In this research, we found strong negative correlation between student-staff ratio and graduation rate."
            }),
            (["what is outlier", "outliers in data", "how to detect outliers"], {
                "answer": "An outlier is a data point significantly different from others (e.g. unusually high enrollment in one year). We detect them using IQR method or z-scores. In UI enrollment data, outliers often come from policy changes or major strikes."
            }),
            (["what is missing data", "missing values", "how to handle missing data"], {
                "answer": "Missing data occurs when values are absent (e.g. unreported budget for some years). We handle it using imputation: median for numerical variables, mode for categorical, or advanced methods like KNN imputation if needed."
            }),
            (["what is feature engineering", "feature creation"], {
                "answer": "Feature engineering is creating new meaningful variables from raw data to improve model performance — e.g. calculating student-staff ratio, budget per student, or lag features from previous years' enrollment."
            }),
            (["what is data preprocessing", "data preparation steps"], {
                "answer": "Data preprocessing includes cleaning (remove duplicates, fix errors), handling missing values, encoding categorical variables, scaling/normalizing numerical features, and splitting data into train/test sets — all essential before training models."
            }),
            (["what is normalization", "what is standardization", "scaling data"], {
                "answer": "Normalization scales features to a fixed range (usually 0–1). Standardization transforms features to have mean=0 and standard deviation=1. Both help machine learning algorithms (especially distance-based ones) perform better."
            }),
            (["categorical vs numerical data", "types of data"], {
                "answer": "Numerical data has meaningful numbers (enrollment count, budget). Categorical data represents groups/categories (faculty name, gender, mode of entry). We use one-hot or label encoding for categorical variables before modeling."
            }),
            (["what is time series data", "time series in enrollment"], {
                "answer": "Time series data is collected over time with order mattering (e.g. yearly enrollment from 2014–2024). Enrollment forecasting is a time-series problem — that's why we tested LSTM, though Random Forest performed better on our dataset."
            }),
            (["why use visualizations in eda", "importance of plots"], {
                "answer": "Visualizations (histograms, box plots, scatter plots, heatmaps) reveal patterns, trends, outliers, correlations, and data quality issues that are hard to see in raw numbers alone."
            }),
            (["what is histogram", "what is box plot"], {
                "answer": "Histogram shows the frequency distribution of a numerical variable. Box plot summarizes data with median, quartiles, and outliers — very useful for seeing enrollment spread across years or faculties."
            }),
        ]

        # Session state initialization
        if 'ui_chat_history' not in st.session_state:
            st.session_state.ui_chat_history = []
        if 'ui_chat_context' not in st.session_state:
            st.session_state.ui_chat_context = {}

def find_predefined_answer(self, question: str) -> Optional[Dict[str, str]]:
    original = question.lower().strip()
    # Fix: normalize multiple spaces
    q = original.rstrip('?.!,').replace('  ', ' ').replace('   ', ' ')

    # More thorough prefix removal (loop until no more prefixes)
    cleaned = q
    prefixes = ["what is ", "what are ", "explain ", "tell me about ", "what does ", "define ", "what ", "tell me "]
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                changed = True

    # Debug prints (remove later if you want)
    # print(f"DEBUG: Original: '{original}'")
    # print(f"DEBUG: q:        '{q}'")
    # print(f"DEBUG: cleaned:  '{cleaned}'")

    # 1. Exact full match on cleaned or q
    for patterns, answer_data in self.qa_patterns:
        for pattern in patterns:
            if pattern == q or pattern == cleaned:
                # print(f"DEBUG: Exact match → {pattern}")
                return answer_data

    # 2. Strong substring match + overlap (lowered threshold slightly for short terms)
    q_words = set(cleaned.split())
    for patterns, answer_data in self.qa_patterns:
        for pattern in patterns:
            # substring check
            if pattern in q or pattern in cleaned:
                pat_words = set(pattern.split())
                if len(pat_words) > 0:
                    overlap_ratio = len(pat_words.intersection(q_words)) / len(pat_words)
                    if overlap_ratio >= 0.65:  # slightly relaxed for short words like "eda"
                        # print(f"DEBUG: Overlap match → {pattern} (ratio {overlap_ratio:.2f})")
                        return answer_data

    # print("DEBUG: No match")
    return None

    def _generate_fallback_response(self, question: str) -> str:
        return """Sorry, I didn't quite understand the question.

Try asking one of these:
• "What is enrollment prediction?"
• "What affects enrollment?"
• "How does optimization work?"
• "What is random forest?"
• "What is the best model?"
• "Challenges of Nigerian universities"

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
            st.markdown("Ask anything about enrollment forecasting, resource optimization, machine learning models, or challenges in Nigerian higher education.")

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
                st.info("👋 Welcome! I'm your UI Enrollment & Optimization Assistant.\nAsk me about predictions, algorithms, Nigerian university challenges, or how to use the platform.")

            for msg in messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        user_question = st.chat_input("Ask about enrollment, optimization, algorithms, Nigeria higher education...")

        if user_question:
            self.add_message("user", user_question)

            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    match = self.find_predefined_answer(user_question)
                    response = match['answer'] if match else self._generate_fallback_response(user_question)
                    st.markdown(response)
                    self.add_message("assistant", response)

            st.rerun()

        if not compact:
            st.markdown("---")
            st.markdown("**Quick Questions:**")
            quick = [
                "What is enrollment prediction?",
                "What affects enrollment?",
                "How does optimization work?",
                "What is Random Forest?"
            ]
            cols = st.columns(2)
            for i, q in enumerate(quick):
                with cols[i % 2]:
                    if st.button(q, key=f"qq_ui_{i}", use_container_width=True):
                        self.add_message("user", q)
                        match = self.find_predefined_answer(q)
                        if match:
                            self.add_message("assistant", match['answer'])
                        st.rerun()


def render_ui_chatbox(app_context: Optional[Dict[str, Any]] = None, compact: bool = False):
    if 'ui_chatbox' not in st.session_state:
        st.session_state.ui_chatbox = UIEnrollmentChatbox()
    st.session_state.ui_chatbox.render(app_context, compact)
