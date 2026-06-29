# app.py
import streamlit as st
import sqlite3
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(
    page_title="Service Excellence Prediction System",
    page_icon="📊",
    layout="wide"
)

# Load data
conn = sqlite3.connect("hotel_employees.db")
employee_data = pd.read_sql_query("SELECT * FROM employees", conn)
conn.close()

# Load model
with open("employee_performance_model.pkl", "rb") as f:
    model = pickle.load(f)

# App Header
st.title("📊 Service Excellence Prediction System")
st.caption("A Workforce Analytics Platform for Hospitality Operations")

st.markdown(
    """
    This dashboard predicts employee performance using attendance, task completion, 
    and customer service metrics. It also provides promotion readiness, merit increase 
    guidance, and operational recommendations.
    """
)

# Sidebar
st.sidebar.header("Employee Selection")
employee_names = employee_data["name"].tolist()
selected_employee = st.sidebar.selectbox("Choose an employee", employee_names)

employee = employee_data[employee_data["name"] == selected_employee].iloc[0]

# Inputs
st.sidebar.header("Adjust Performance Metrics")

new_attendance = st.sidebar.slider(
    "Attendance %",
    min_value=60,
    max_value=100,
    value=int(employee["attendance"])
)

new_task_completion = st.sidebar.slider(
    "Task Completion %",
    min_value=50,
    max_value=100,
    value=int(employee["task_completion"])
)

new_customer_service = st.sidebar.slider(
    "Customer Service %",
    min_value=40,
    max_value=100,
    value=int(employee["customer_service"])
)

# Prediction
predicted_performance = model.predict(
    [[new_attendance, new_task_completion, new_customer_service]]
)[0]

predicted_performance = float(np.clip(predicted_performance, 0, 100))

# Derived scores
promotion_readiness = round(
    (new_attendance * 0.25)
    + (new_task_completion * 0.35)
    + (new_customer_service * 0.40),
    2
)

if predicted_performance >= 90:
    performance_tier = "Elite Performer"
    merit_increase = "5% - 7%"
    recommendation = "Strong candidate for promotion, leadership development, or expanded responsibility."
elif predicted_performance >= 85:
    performance_tier = "High Performer"
    merit_increase = "3% - 5%"
    recommendation = "Consider for merit increase, cross-training, or advancement planning."
elif predicted_performance >= 75:
    performance_tier = "Solid Performer"
    merit_increase = "1% - 3%"
    recommendation = "Maintain current role while providing coaching for continued growth."
else:
    performance_tier = "Development Needed"
    merit_increase = "0% - 1%"
    recommendation = "Create a performance improvement plan focused on attendance, service, and task execution."

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

col1.metric("Predicted Performance", f"{predicted_performance:.2f}%")
col2.metric("Promotion Readiness", f"{promotion_readiness:.2f}%")
col3.metric("Performance Tier", performance_tier)
col4.metric("Merit Increase Range", merit_increase)

st.divider()

# Employee Profile
left, right = st.columns([1, 2])

with left:
    st.subheader("Employee Profile")
    st.write(f"**Name:** {selected_employee}")
    st.write(f"**Role:** {employee['role']}")
    st.write(f"**Sex:** {employee['sex']}")
    st.write(f"**Age:** {employee['age']} years")
    st.write(f"**Start Date:** {employee['start_date']}")
    st.write(f"**Current Attendance:** {employee['attendance']}%")
    st.write(f"**Current Task Completion:** {employee['task_completion']}%")
    st.write(f"**Current Customer Service:** {employee['customer_service']}%")

with right:
    st.subheader("Prediction Explanation")

    drivers = []

    if new_attendance >= 90:
        drivers.append("Attendance is a strong positive driver.")
    elif new_attendance < 75:
        drivers.append("Attendance is lowering the performance prediction.")

    if new_task_completion >= 90:
        drivers.append("Task completion shows strong operational consistency.")
    elif new_task_completion < 75:
        drivers.append("Task completion needs improvement.")

    if new_customer_service >= 90:
        drivers.append("Customer service is a key strength.")
    elif new_customer_service < 75:
        drivers.append("Customer service may be impacting the overall score.")

    if not drivers:
        drivers.append("Performance is balanced across attendance, task completion, and customer service.")

    for item in drivers:
        st.write(f"✅ {item}")

    st.info(recommendation)

st.divider()

# Visuals
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Adjusted Performance Metrics")
    metrics_df = pd.DataFrame({
        "Metric": ["Attendance", "Task Completion", "Customer Service"],
        "Score": [new_attendance, new_task_completion, new_customer_service]
    })

    fig, ax = plt.subplots()
    ax.bar(metrics_df["Metric"], metrics_df["Score"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Current Adjusted Metrics")
    st.pyplot(fig)

with chart_col2:
    st.subheader("Performance Progression")

    months = np.arange(1, 13)
    base_score = predicted_performance
    performance_progress = np.clip(
        base_score + np.random.normal(0, 4, 12),
        60,
        100
    )

    fig2, ax2 = plt.subplots()
    ax2.plot(months, performance_progress, marker="o")
    ax2.axhline(85, linestyle="--", label="High Performance Threshold")
    ax2.axhline(70, linestyle="--", label="Needs Improvement Threshold")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Performance Score (%)")
    ax2.set_title(f"{selected_employee}'s Simulated Performance Trend")
    ax2.legend()
    st.pyplot(fig2)

st.divider()

# Team Overview
st.subheader("Team Performance Overview")

team_avg_attendance = employee_data["attendance"].mean()
team_avg_tasks = employee_data["task_completion"].mean()
team_avg_service = employee_data["customer_service"].mean()

team_col1, team_col2, team_col3 = st.columns(3)

team_col1.metric("Team Avg Attendance", f"{team_avg_attendance:.2f}%")
team_col2.metric("Team Avg Task Completion", f"{team_avg_tasks:.2f}%")
team_col3.metric("Team Avg Customer Service", f"{team_avg_service:.2f}%")

st.subheader("Employee Ranking Snapshot")

ranking_df = employee_data.copy()
ranking_df["estimated_score"] = (
    ranking_df["attendance"] * 0.30
    + ranking_df["task_completion"] * 0.35
    + ranking_df["customer_service"] * 0.35
)

ranking_df = ranking_df.sort_values("estimated_score", ascending=False)

st.dataframe(
    ranking_df[["name", "role", "attendance", "task_completion", "customer_service", "estimated_score"]],
    use_container_width=True
)

st.caption(
    "Note: Promotion readiness and merit increase guidance are analytical estimates for portfolio demonstration purposes, not official HR decisions."
)
