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

# Professional demo employee names
demo_names = [
    "Alicia Carter", "Marcus Johnson", "Taylor Brooks", "Jordan Williams",
    "Sophia Martinez", "Brandon Lee", "Danielle Parker", "Christopher Allen",
    "Maya Thompson", "Derrick Harris", "Natalie Reed", "Kevin Morgan",
    "Jasmine Scott", "Anthony Bell", "Brianna Cooper", "Malik Robinson",
    "Olivia Bennett", "Caleb Turner", "Naomi Foster", "Isaiah Mitchell",
    "Lauren Hughes", "Devin Price", "Morgan Ellis", "Rachel Simmons",
    "Cameron Wright", "Tiffany Adams", "Andre Coleman", "Erica James"
]

department_map = [
    "Front Desk", "Housekeeping", "Maintenance", "Valet",
    "Janitorial", "Restaurant", "Breakfast", "Guest Services"
]

employee_data = employee_data.reset_index(drop=True)
employee_data["Employee ID"] = employee_data.index + 1001
employee_data["Employee Name"] = [demo_names[i % len(demo_names)] for i in range(len(employee_data))]
employee_data["Department"] = [department_map[i % len(department_map)] for i in range(len(employee_data))]

def category(score):
    if score < 70:
        return "Below Requirements", "red"
    elif score < 85:
        return "Needs Improvement", "#D4A017"
    else:
        return "At or Above Expectations", "green"

st.title("📊 Service Excellence Prediction System")
st.caption("A Workforce Analytics Platform for Hospitality Operations")

st.markdown(
    "This dashboard predicts employee performance using attendance, task completion, "
    "and customer service metrics. It also provides promotion readiness, merit increase "
    "guidance, employee ranking, and operational recommendations."
)

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

predicted_performance = model.predict(
    [[new_attendance, new_task_completion, new_customer_service]]
)[0]

predicted_performance = float(np.clip(predicted_performance, 0, 100))
performance_tier, tier_color = category(predicted_performance)

promotion_readiness = round(
    (new_attendance * 0.25) +
    (new_task_completion * 0.35) +
    (new_customer_service * 0.40),
    2
)

if predicted_performance < 70:
    merit_increase = "0% - 1%"
    recommendation = "Create a performance improvement plan focused on attendance, task completion, and guest service."
elif predicted_performance < 85:
    merit_increase = "1% - 3%"
    recommendation = "Provide coaching, follow-up training, and clear short-term performance goals."
else:
    merit_increase = "3% - 7%"
    recommendation = "Strong performer. Consider recognition, advancement planning, or leadership development."

col1, col2, col3, col4 = st.columns(4)

col1.metric("Predicted Performance", f"{predicted_performance:.2f}%")
col2.metric("Promotion Readiness", f"{promotion_readiness:.2f}%")

with col3:
    st.markdown("**Performance Tier**")
    st.markdown(
        f"<div style='color:{tier_color}; font-size:20px; font-weight:700; line-height:1.2;'>{performance_tier}</div>",
        unsafe_allow_html=True
    )

col4.metric("Merit Increase Range", merit_increase)

st.divider()

left, right = st.columns([1, 2])

with left:
    st.subheader("Employee Profile")
    st.write(f"**Employee Name:** {employee['Employee Name']}")
    st.write(f"**Employee ID:** {employee['Employee ID']}")
    st.write(f"**Department:** {employee['Department']}")
    st.write(f"**Original Role:** {employee['role']}")
    st.write(f"**Sex:** {employee['sex']}")
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
        tier, color = category(score)
        st.markdown(
            f"<span style='color:{color}; font-weight:700;'>●</span> "
            f"{label}: {score}% — {tier}",
            unsafe_allow_html=True
        )

    st.info(recommendation)

st.divider()

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Adjusted Performance Metrics")

    metrics = ["Attendance", "Task Completion", "Customer Service"]
    scores = [new_attendance, new_task_completion, new_customer_service]
    colors = [category(score)[1] for score in scores]

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
        color = category(performance_progress[i])[1]
        ax2.plot(
            months[i:i+2],
            performance_progress[i:i+2],
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

st.subheader("Employee Ranking Snapshot")

ranking_df = employee_data.copy()
ranking_df["Estimated Score"] = (
    ranking_df["attendance"] * 0.30 +
    ranking_df["task_completion"] * 0.35 +
    ranking_df["customer_service"] * 0.35
)

ranking_df["Performance Category"] = ranking_df["Estimated Score"].apply(lambda x: category(x)[0])

department_options = ["Whole Hotel"] + sorted(ranking_df["Department"].unique().tolist())

selected_department = st.selectbox(
    "Filter rankings by department",
    department_options
)

if selected_department != "Whole Hotel":
    ranking_df = ranking_df[ranking_df["Department"] == selected_department]

ranking_df = ranking_df.sort_values("Estimated Score", ascending=False)

st.dataframe(
    ranking_df[
        [
            "Employee ID",
            "Employee Name",
            "Department",
            "attendance",
            "task_completion",
            "customer_service",
            "Estimated Score",
            "Performance Category"
        ]
    ].rename(columns={
        "attendance": "Attendance",
        "task_completion": "Task Completion",
        "customer_service": "Customer Service"
    }),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Legend: Red = Below Requirements | Yellow = Needs Improvement | Green = At or Above Expectations."
)
