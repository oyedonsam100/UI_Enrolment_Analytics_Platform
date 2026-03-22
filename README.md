# 🎓 University of Ibadan — Enrolment Analytics Platform

> **A data-driven forecasting and analytics system that turns 10 years of university enrolment data into actionable intelligence for institutional planning.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://uienrolmentanalyticsplatform-etygix4lc3xy6eosum9ard.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org)

---

## 🌐 Live Demo

**[→ Launch Platform](https://uienrolmentanalyticsplatform-etygix4lc3xy6eosum9ard.streamlit.app)**

---

## Overview

Nigerian universities face a recurring institutional challenge: enrolment numbers fluctuate unpredictably due to economic conditions, strikes, policy changes, and demographic shifts — yet resource allocation decisions for hostels, academic staff, budgets, and facilities are made months or years in advance. Poor forecasting leads to overstaffed faculties, underprovisioned hostels, and misaligned budgets.

This platform addresses that problem directly. Built on 10 years of real University of Ibadan enrolment data (2014–2024), it combines exploratory analytics with machine learning forecasting to give administrators a clear, evidence-based view of where enrolment is heading — and what it means for resource planning.

---

## The Problem

Universities struggle to predict student enrolment trends accurately, resulting in:

- **Hostel shortfalls** — insufficient bed space when enrolment spikes unexpectedly
- **Staffing mismatches** — faculty overloaded or underutilised due to poor demand forecasting
- **Budget inefficiency** — allocations based on last year's numbers rather than projected demand
- **Reactive planning** — decisions made after problems emerge rather than before

---

## Solution

A full-stack analytics platform that ingests historical enrolment data alongside economic and institutional variables, performs ML-based forecasting, and presents findings through an interactive dashboard built for non-technical decision-makers.

---

## Features

- **Exploratory Data Analysis** — enrolment trends by faculty, year, and programme with interactive visualisations
- **Feature Importance Analysis** — identifies which variables (GDP, strike duration, staff count, etc.) most drive enrolment changes
- **Time-Series Trend Analysis** — visualises long-term patterns and seasonal enrolment cycles
- **ML Forecasting Models** — predicts future enrolment figures with up to 95% accuracy
- **Resource Planning Projections** — translates forecasts into staffing, hostel, and budget recommendations
- **Interactive Streamlit Dashboard** — filters, charts, and predictions in one deployable interface

---

## Dataset

| Variable | Description |
|---|---|
| Annual enrolment figures | By faculty and department, 2014–2024 |
| Academic staff count | Per faculty, per year |
| Hostel space availability | Total bed capacity per session |
| Graduation outcomes | Completion rates by faculty |
| GDP growth rate | Nigeria annual GDP change |
| Departmental budgets | Annual faculty budget allocation |
| Unemployment rate | Graduate employment context |
| Strike duration | Academic calendar disruption (ASUU) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Processing | Python · Pandas · NumPy |
| Machine Learning | Scikit-learn |
| Visualisation | Plotly · Matplotlib |
| Dashboard | Streamlit |
| Database | SQL |
| Version Control | GitHub |

---

## Key Results

- Analysed **37,000+ enrolment records** across 10 academic sessions
- Built ML models achieving **95% forecast accuracy** on held-out test data
- Identified enrolment growth patterns and stress points across all major faculties
- Quantified the impact of ASUU strike duration on enrolment drop-off
- Highlighted staff-to-student ratio imbalances across 12 faculties
- Produced actionable projections for hostel allocation, staffing needs, and budget planning

---

## Business Impact

University administrators can use this platform to:

- **Plan staffing requirements** 1–3 years ahead based on projected faculty enrolment
- **Optimise hostel allocation** — match bed space provisioning to forecasted intake
- **Improve budget forecasting** — allocate departmental resources based on demand projections rather than historical averages
- **Monitor at-risk faculties** — identify departments with declining enrolment trends early enough to intervene

---

## Run Locally

```bash
# Clone the repository
git clone https://github.com/SamuelOyedokun/UI_Enrolment_Analytics_Platform
cd UI_Enrolment_Analytics_Platform

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

---

## Project Structure

```
UI_Enrolment_Analytics_Platform/
├── app.py                  # Streamlit dashboard
├── model.py                # ML forecasting models
├── data/                   # Enrolment datasets
├── notebooks/              # EDA and analysis notebooks
├── requirements.txt        # Dependencies
└── README.md
```

---

## Author

**Samuel Oyedokun** — Data Analyst & BI Engineer

- 🌐 Portfolio: [github.com/SamuelOyedokun](https://github.com/SamuelOyedokun)
- 💼 LinkedIn: [samuel-oyedokun-b41895142](https://www.linkedin.com/in/samuel-oyedokun-b41895142)
- 📧 thesamueloyedokun@gmail.com



*Built as part of Masters of Information Science research — University of Ibadan, 2026.*
