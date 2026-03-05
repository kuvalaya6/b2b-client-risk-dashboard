import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Client Risk Intelligence System", layout="wide")

st.title("B2B Customer Risk & Churn Prediction Dashboard")

# -------------------------------
# Load Dataset
# -------------------------------

@st.cache_data
def get_data():
    data = pd.read_csv("B2B_Client_Churn_5000.csv")
    return data

data = get_data()

# -------------------------------
# Risk Score Calculation
# -------------------------------

def calculate_risk(row):

    risk_points = 0

    # Payment delay logic
    if row["Payment_Delay_Days"] > 30:
        risk_points += 3
    elif row["Payment_Delay_Days"] > 15:
        risk_points += 2
    elif row["Payment_Delay_Days"] > 5:
        risk_points += 1

    # Usage logic
    if row["Monthly_Usage_Score"] < 35:
        risk_points += 3
    elif row["Monthly_Usage_Score"] < 55:
        risk_points += 2
    elif row["Monthly_Usage_Score"] < 70:
        risk_points += 1

    # Contract duration
    if row["Contract_Length_Months"] < 6:
        risk_points += 3
    elif row["Contract_Length_Months"] < 12:
        risk_points += 2
    elif row["Contract_Length_Months"] < 18:
        risk_points += 1

    # Support complaints
    if row["Support_Tickets_Last30Days"] > 7:
        risk_points += 3
    elif row["Support_Tickets_Last30Days"] > 4:
        risk_points += 2
    elif row["Support_Tickets_Last30Days"] > 1:
        risk_points += 1

    return risk_points


data["Risk_Score"] = data.apply(calculate_risk, axis=1)


# -------------------------------
# Risk Categorization
# -------------------------------

def assign_risk_level(score):

    if score >= 9:
        return "High Risk"
    elif score >= 5:
        return "Medium Risk"
    else:
        return "Low Risk"

data["Risk_Category"] = data["Risk_Score"].apply(assign_risk_level)

# -------------------------------
# Sidebar Filters
# -------------------------------

st.sidebar.header("Dashboard Filters")

region_options = sorted(data["Region"].unique())
industry_options = sorted(data["Industry"].unique())

selected_regions = st.sidebar.multiselect(
    "Select Region",
    region_options,
    default=region_options
)

selected_industries = st.sidebar.multiselect(
    "Select Industry",
    industry_options,
    default=industry_options
)

selected_risk = st.sidebar.multiselect(
    "Risk Category",
    ["Low Risk", "Medium Risk", "High Risk"],
    default=["Low Risk", "Medium Risk", "High Risk"]
)

filtered_df = data[
    (data["Region"].isin(selected_regions)) &
    (data["Industry"].isin(selected_industries)) &
    (data["Risk_Category"].isin(selected_risk))
]

# -------------------------------
# KPI Section
# -------------------------------

total_clients = len(filtered_df)
high_risk_clients = (filtered_df["Risk_Category"] == "High Risk").sum()

average_revenue = filtered_df["Monthly_Revenue_USD"].mean()

churn_rate = (filtered_df["Renewal_Status"] == "No").mean() * 100

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Clients", total_clients)
k2.metric("High Risk Clients", high_risk_clients)
k3.metric("Predicted Churn Rate", f"{churn_rate:.2f}%")
k4.metric("Average Revenue per Client", f"${average_revenue:,.2f}")

st.divider()

# -------------------------------
# Visualization Section
# -------------------------------

colA, colB = st.columns(2)

with colA:

    st.subheader("Client Risk Distribution")

    risk_counts = filtered_df["Risk_Category"].value_counts()

    fig1, ax1 = plt.subplots()

    ax1.bar(risk_counts.index, risk_counts.values)

    ax1.set_xlabel("Risk Level")
    ax1.set_ylabel("Number of Clients")

    st.pyplot(fig1)


with colB:

    st.subheader("Industry vs Risk Table")

    industry_risk_table = pd.pivot_table(
        filtered_df,
        index="Industry",
        columns="Risk_Category",
        values="Client_ID",
        aggfunc="count",
        fill_value=0
    )

    st.dataframe(industry_risk_table, use_container_width=True)

st.divider()

colC, colD = st.columns(2)

with colC:

    st.subheader("Revenue vs Risk Score")

    fig2, ax2 = plt.subplots()

    ax2.scatter(
        filtered_df["Monthly_Revenue_USD"],
        filtered_df["Risk_Score"]
    )

    ax2.set_xlabel("Monthly Revenue (USD)")
    ax2.set_ylabel("Risk Score")

    st.pyplot(fig2)


with colD:

    st.subheader("Contract Duration vs Churn")

    churn_numeric = filtered_df["Renewal_Status"].map({"Yes": 0, "No": 1})

    fig3, ax3 = plt.subplots()

    ax3.scatter(
        filtered_df["Contract_Length_Months"],
        churn_numeric
    )

    ax3.set_xlabel("Contract Length (Months)")
    ax3.set_ylabel("Churn (1 = Yes, 0 = No)")

    st.pyplot(fig3)

st.divider()

# -------------------------------
# Machine Learning Model
# -------------------------------

st.subheader("Decision Tree Model for Churn Prediction")

target = data["Renewal_Status"].map({"Yes": 1, "No": 0})

features = data[[
    "Monthly_Usage_Score",
    "Payment_Delay_Days",
    "Contract_Length_Months",
    "Support_Tickets_Last30Days",
    "Monthly_Revenue_USD",
    "Risk_Score"
]]

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.20,
    random_state=42,
    stratify=target
)

dt_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

dt_model.fit(X_train, y_train)

predictions = dt_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

st.write("Model Accuracy:", round(accuracy, 4))

st.write("Confusion Matrix")

st.write(confusion_matrix(y_test, predictions))

# Feature Importance

importance = pd.Series(
    dt_model.feature_importances_,
    index=features.columns
).sort_values(ascending=False)

st.subheader("Feature Importance Analysis")

st.bar_chart(importance)

st.divider()

# -------------------------------
# High Risk Client Table
# -------------------------------

st.subheader("Top 20 High Risk Clients")

top_clients = filtered_df.sort_values(
    ["Risk_Score", "Monthly_Revenue_USD"],
    ascending=[False, False]
).head(20)

st.dataframe(top_clients, use_container_width=True)

st.divider()

# -------------------------------
# Retention Strategy Section
# -------------------------------

st.subheader("AI-Driven Client Retention Suggestions")

if st.button("Generate Retention Strategy"):

    st.write("• Provide flexible payment options for clients with long payment delays.")

    st.write("• Conduct training sessions to improve product usage for low engagement clients.")

    st.write("• Assign dedicated account managers to clients raising frequent support tickets.")

    st.write("• Offer discounts or benefits for long-term contract renewals.")

    st.write("• For high revenue high-risk clients, initiate executive-level relationship management.")

st.divider()

# -------------------------------
# Responsible AI Section
# -------------------------------

st.subheader("Responsible AI: Ethical Considerations")

st.write("""
• **Algorithmic Bias:** Predictive models may unintentionally favor or disadvantage certain industries or regions.

• **Client Labeling Risk:** Labeling companies as "High Risk" may negatively affect how teams interact with them.

• **Data Privacy:** Sensitive data such as financial transactions and usage behavior must be protected.

• **Human Oversight:** AI predictions should support decision-making rather than replace human judgment.

• **Model Monitoring:** Organizations should continuously evaluate models for fairness and accuracy.
""")
