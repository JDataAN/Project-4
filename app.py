# app.py
import streamlit as st
import sqlite3
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from faker import Faker

st.set_page_config(
    page_title="Service Excellence Prediction System",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
div[data-testid="stMetricValue"] {
    font-size: 24px !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 13px !important;
}
.tier-box {
    padding: 10px;
    border-radius: 6px;
    font-size: 18px;
    font-weight: 700;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Load data
conn = sqlite3.connect("hotel_employees.db")
employee_data = pd.read_sql_query("SELECT * FROM employees", conn)
conn.close()

# Load model
with open("employee_performance_model.pkl", "rb") as f:
    model = pickle.load(f)

# Add unique employee names and departments for portfolio display
fake = Faker()
Faker.seed(42)
np.random.seed(42)

departments = [
    "Front Desk",
    "Housekeeping",
    "Maintenance",
    "Valet",
    "Janitorial",
    "Restaurant",
    "Breakfast",
    "Engineering",
    "Sales",
    "Events",
    "Security",
    "Management"
]

employee_data = employee_data.reset_index(drop=True)

employee_data["Employee ID"] = [
    f"EMP-{1001 + i}"
    for i in range(len(employee_data))
]

employee_data["Employee Name"] = [
    fake.name()
    for _ in range(len(employee_data))
]

employee_data["Department"] = [
    np.random.choice(departments)
    for _ in range(len(employee_data))
]

def get_status(score):
    if score < 70:
        return "Below Requirements", "red"
    elif score < 85:
        return "Needs Improvement", "#D4A017"
    else:
        return "At or Above Expectations", "green"

st.title("📊 Service Excellence Prediction System")
st.caption("A Workforce Analytics Platform for Hospitality Operations")

st.write(
    "This dashboard predicts employee performance using attendance, task completion, "
    "and customer service metrics. It also provides promotion readiness, merit increase "
    "guidance, and operational recommendations."
)

# Sidebar
st.sidebar.header("Employee Selection")
selected_employee = st.sidebar.selectbox(
    "Choose an employee",
    employee_data["Employee Name"].tolist()
)

employee = employee_data[employee_data["Employee Name"] == selected_employee].iloc[0]

st.sidebar.header("Adjust Performance Metrics")

new_attendance = st.sidebar.slider("Attendance %", 60, 100, int(employee["attendance"]))
new_task_completion = st.sidebar.slider("Task Completion %", 50, 100, int(employee["task_completion"]))
new_customer_service = st.sidebar.slider("Customer Service %", 40, 100, int(employee["customer_service"]))

# Prediction
predicted_performance = model.predict(
    [[new_attendance, new_task_completion, new_customer_service]]
)[0]
predicted_performance = float(np.clip(predicted_performance, 0, 100))

performance_tier, tier_color = get_status(predicted_performance)

promotion_readiness = round(
    (new_attendance * 0.25)
    + (new_task_completion * 0.35)
    + (new_customer_service * 0.40),
    2
)

if predicted_performance < 70:
    merit_increase = "0% - 1%"
    recommendation = "Create a performance improvement plan focused on attendance, task completion, and customer service."
elif predicted_performance < 85:
    merit_increase = "1% - 3%"
    recommendation = "Provide coaching, follow-up training, and measurable short-term goals."
else:
    merit_increase = "3% - 7%"
    recommendation = "Strong performer. Consider recognition, advancement planning, or leadership development."

# KPI cards
col1, col2, col3, col4 = st.columns(4)

col1.metric("Predicted Performance", f"{predicted_performance:.2f}%")
col2.metric("Promotion Readiness", f"{promotion_readiness:.2f}%")

with col3:
    st.markdown("**Performance Tier**")
    st.markdown(
        f"<div class='tier-box' style='background-color:{tier_color}; color:white;'>{performance_tier}</div>",
        unsafe_allow_html=True
    )

col4.metric("Merit Increase Range", merit_increase)

st.divider()

# Employee profile
left, right = st.columns([1, 2])

with left:
    st.subheader("Employee Profile")
    st.write(f"**Employee Name:** {employee['Employee Name']}")
    st.write(f"**Employee ID:** {employee['Employee ID']}")
    st.write(f"**Department:** {employee['Department']}")
    st.write(f"**Original Role:** {employee['role']}")
    st.write(f"**Age:** {employee['age']} years")
    st.write(f"**Start Date:** {employee['start_date']}")
    st.write(f"**Attendance:** {employee['attendance']}%")
    st.write(f"**Task Completion:** {employee['task_completion']}%")
    st.write(f"**Customer Service:** {employee['customer_service']}%")

with right:
    st.subheader("Prediction Explanation")

    for label, score in {
        "Attendance": new_attendance,
        "Task Completion": new_task_completion,
        "Customer Service": new_customer_service
    }.items():
        status, color = get_status(score)
        st.markdown(
            f"<span style='color:{color}; font-size:22px;'>●</span> "
            f"**{label}:** {score}% — {status}",
            unsafe_allow_html=True
        )

    st.info(recommendation)

st.divider()

# Charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Adjusted Performance Metrics")

    metrics = ["Attendance", "Task Completion", "Customer Service"]
    scores = [new_attendance, new_task_completion, new_customer_service]
    colors = [get_status(score)[1] for score in scores]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(metrics, scores, color=colors)
    ax.set_ylim(0, 100)
    ax.axhline(70, color="red", linestyle="--", label="Below Requirements")
    ax.axhline(85, color="green", linestyle="--", label="At/Above Expectations")
    ax.set_ylabel("Score (%)")
    ax.set_title("Adjusted Metrics by Performance Category")
    ax.legend()
    st.pyplot(fig)

with chart_col2:
    st.subheader("Performance Progression")

    months = np.arange(1, 13)
    performance_progress = np.clip(
        predicted_performance + np.random.normal(0, 4, 12),
        60,
        100
    )

    fig2, ax2 = plt.subplots(figsize=(7, 4))

    for i in range(len(months) - 1):
        color = get_status(performance_progress[i])[1]
        ax2.plot(
            months[i:i + 2],
            performance_progress[i:i + 2],
            marker="o",
            linewidth=3,
            color=color
        )

    ax2.axhline(70, color="red", linestyle="--", label="Below Requirements")
    ax2.axhline(85, color="green", linestyle="--", label="At/Above Expectations")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Performance Score (%)")
    ax2.set_title(f"{selected_employee}'s Simulated Performance Trend")
    ax2.legend()
    st.pyplot(fig2)

st.divider()

# Ranking
st.subheader("Employee Ranking Snapshot")

ranking_df = employee_data.copy()
ranking_df["Estimated Score"] = (
    ranking_df["attendance"] * 0.30
    + ranking_df["task_completion"] * 0.35
    + ranking_df["customer_service"] * 0.35
)

ranking_df["Performance Category"] = ranking_df["Estimated Score"].apply(lambda x: get_status(x)[0])

department_options = ["Whole Hotel"] + sorted(ranking_df["Department"].unique().tolist())
selected_department = st.selectbox("Filter rankings by department", department_options)

if selected_department != "Whole Hotel":
    ranking_df = ranking_df[ranking_df["Department"] == selected_department]

ranking_df = ranking_df.sort_values("Estimated Score", ascending=False)

def add_badge(row):
    score = row["Estimated Score"]
    status = row["Performance Category"]

    if score >= 90:
        return "🏆 Top Performer"
    elif score >= 85:
        return "⭐ Promotion Candidate"
    elif status == "Needs Improvement":
        return "⚠️ Coaching Needed"
    else:
        return "🔴 Immediate Support"

ranking_df["Talent Badge"] = ranking_df.apply(add_badge, axis=1)

def color_status_cell(value):
    if value == "Below Requirements":
        return "background-color: #8B0000; color: white; font-weight: 700;"
    elif value == "Needs Improvement":
        return "background-color: #B8860B; color: black; font-weight: 700;"
    elif value == "At or Above Expectations":
        return "background-color: #006400; color: white; font-weight: 700;"
    return ""

display_df = ranking_df[
    [
        "Employee ID",
        "Employee Name",
        "Department",
        "attendance",
        "task_completion",
        "customer_service",
        "Estimated Score",
        "Performance Category",
        "Talent Badge"
    ]
].rename(columns={
    "attendance": "Attendance",
    "task_completion": "Task Completion",
    "customer_service": "Customer Service"
})

styled_df = display_df.style.map(
    color_status_cell,
    subset=["Performance Category"]
)

st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True
)

st.caption("Legend: Red = Below Requirements | Yellow = Needs Improvement | Green = At or Above Expectations.")
