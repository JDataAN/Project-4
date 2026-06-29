# app.py
import streamlit as st
import sqlite3
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

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

# Add professional employee names if names are placeholders
first_last_names = [
    "Alicia Carter", "Marcus Johnson", "Taylor Brooks", "Jordan Williams",
    "Sophia Martinez", "Brandon Lee", "Danielle Parker", "Christopher Allen",
    "Maya Thompson", "Derrick Harris", "Natalie Reed", "Kevin Morgan",
    "Jasmine Scott", "Anthony Bell", "Brianna Cooper", "Malik Robinson",
    "Olivia Bennett", "Caleb Turner", "Naomi Foster", "Isaiah Mitchell"
]

employee_data = employee_data.reset_index(drop=True)
employee_data["employee_id"] = employee_data.index + 1001
employee_data["display_name"] = [
    first_last_names[i % len(first_last_names)] for i in range(len(employee_data))
]

# App Header
st.title("📊 Service Excellence Prediction System")
st.caption("A Workforce Analytics Platform for Hospitality Operations")

st.markdown(
    """
    This dashboard predicts employee performance using attendance, task completion, 
    and customer service metrics. It also provides promotion readiness, merit increase 
    guidance, employee ranking, and operational recommendations.
    """
)

# Sidebar
st.sidebar.header("Employee Selection")
employee_names = employee_data["display_name"].tolist()
selected_employee = st.sidebar.selectbox("Choose an employee", employee_names)

employee = employee_data[employee_data["display_name"] == selected_employee].iloc[0]

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

promotion_readiness = round(
    (new_attendance * 0.25)
    + (new_task_completion * 0.35)
    + (new_customer_service * 0.40),
    2
)

# Color-coded performance logic
if predicted_performance < 70:
    performance_tier = "Below Requirements"
    tier_color = "red"
    merit_increase = "0% - 1%"
    recommendation = "Create a performance improvement plan focused on attendance, task completion, and guest service."
elif predicted_performance < 85:
    performance_tier = "Needs Improvement"
    tier_color = "gold"
    merit_increase = "1% - 3%"
    recommendation = "Provide coaching, follow-up training, and clear short-term performance goals."
else:
    performance_tier = "At or Above Expectations"
    tier_color = "green"
    merit_increase = "3% - 7%"
    recommendation = "Strong performer. Consider for recognition, advancement planning, or leadership development."

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

col1.metric("Predicted Performance", f"{predicted_performance:.2f}%")
col2.metric("Promotion Readiness", f"{promotion_readiness:.2f}%")

with col3:
    st.markdown("**Performance Tier**")
    st.markdown(
        f"<h4 style='color:{tier_color}; margin-top:0;'>{performance_tier}</h4>",
        unsafe_allow_html=True
    )

col4.metric("Merit Increase Range", merit_increase)

st.divider()

# Employee Profile
left, right = st.columns([1, 2])

with left:
    st.subheader("Employee Profile")
    st.write(f"**Employee Name:** {selected_employee}")
    st.write(f"**Employee ID:** {employee['employee_id']}")
    st.write(f"**Role/Department:** {employee['role']}")
    st.write(f"**Sex:** {employee['sex']}")
    st.write(f"**Age:** {employee['age']} years")
    st.write(f"**Start Date:** {employee['start_date']}")
    st.write(f"**Attendance:** {employee['attendance']}%")
    st.write(f"**Task Completion:** {employee['task_completion']}%")
    st.write(f"**Customer Service:** {employee['customer_service']}%")

with right:
    st.subheader("Prediction Explanation")

    drivers = []

    if new_attendance >= 85:
        drivers.append("Attendance is at or above expectations.")
    elif new_attendance >= 70:
        drivers.append("Attendance needs improvement.")
    else:
        drivers.append("Attendance is below requirements.")

    if new_task_completion >= 85:
        drivers.append("Task completion is at or above expectations.")
    elif new_task_completion >= 70:
        drivers.append("Task completion needs improvement.")
    else:
        drivers.append("Task completion is below requirements.")

    if new_customer_service >= 85:
        drivers.append("Customer service is at or above expectations.")
    elif new_customer_service >= 70:
        drivers.append("Customer service needs improvement.")
    else:
        drivers.append("Customer service is below requirements.")

    for item in drivers:
        st.write(f"• {item}")

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

    bar_colors = [
        "red" if score < 70 else "gold" if score < 85 else "green"
        for score in metrics_df["Score"]
    ]

    fig, ax = plt.subplots()
    ax.bar(metrics_df["Metric"], metrics_df["Score"], color=bar_colors)
    ax.set_ylim(0, 100)
    ax.axhline(70, linestyle="--", color="red", label="Below Requirements")
    ax.axhline(85, linestyle="--", color="green", label="At/Above Expectations")
    ax.set_ylabel("Score (%)")
    ax.set_title("Color-Coded Performance Metrics")
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

    fig2, ax2 = plt.subplots()

    for i in range(len(months) - 1):
        score = performance_progress[i]
        color = "red" if score < 70 else "gold" if score < 85 else "green"
        ax2.plot(months[i:i+2], performance_progress[i:i+2], marker="o", color=color)

    ax2.axhline(70, linestyle="--", color="red", label="Below Requirements")
    ax2.axhline(85, linestyle="--", color="green", label="At/Above Expectations")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Performance Score (%)")
    ax2.set_title(f"{selected_employee}'s Simulated Performance Trend")
    ax2.legend()
    st.pyplot(fig2)

st.divider()

# Team Overview
st.subheader("Team Performance Overview")

team_col1, team_col2, team_col3 = st.columns(3)

team_col1.metric("Team Avg Attendance", f"{employee_data['attendance'].mean():.2f}%")
team_col2.metric("Team Avg Task Completion", f"{employee_data['task_completion'].mean():.2f}%")
team_col3.metric("Team Avg Customer Service", f"{employee_data['customer_service'].mean():.2f}%")

# Employee Ranking with department filter
st.subheader("Employee Ranking Snapshot")

ranking_df = employee_data.copy()
ranking_df["estimated_score"] = (
    ranking_df["attendance"] * 0.30
    + ranking_df["task_completion"] * 0.35
    + ranking_df["customer_service"] * 0.35
)

ranking_df["performance_category"] = ranking_df["estimated_score"].apply(
    lambda x: "Below Requirements" if x < 70 else "Needs Improvement" if x < 85 else "At or Above Expectations"
)

departments = ["All Departments"] + sorted(ranking_df["role"].unique().tolist())
selected_department = st.selectbox("Filter ranking by department/role", departments)

if selected_department != "All Departments":
    ranking_df = ranking_df[ranking_df["role"] == selected_department]

ranking_df = ranking_df.sort_values("estimated_score", ascending=False)

st.dataframe(
    ranking_df[
        [
            "employee_id",
            "display_name",
            "role",
            "attendance",
            "task_completion",
            "customer_service",
            "estimated_score",
            "performance_category"
        ]
    ].rename(columns={
        "employee_id": "Employee ID",
        "display_name": "Employee Name",
        "role": "Department/Role",
        "attendance": "Attendance",
        "task_completion": "Task Completion",
        "customer_service": "Customer Service",
        "estimated_score": "Estimated Score",
        "performance_category": "Performance Category"
    }),
    use_container_width=True
)

st.caption(
    "Legend: Red = Below Requirements | Yellow = Needs Improvement | Green = At or Above Expectations. "
    "Promotion readiness and merit guidance are portfolio demonstration estimates, not official HR decisions."
)
