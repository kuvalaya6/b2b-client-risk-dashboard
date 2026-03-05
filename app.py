%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("B2B_Client_Churn_5000-2.csv")

st.title("B2B Client Risk Intelligence Dashboard")

# ===== Part B: Risk Score =====
# Example thresholds – adjust as needed
def compute_risk_score(row):
    score = 0

    # Payment delay
    if row["Payment_Delay_Days"] > 30:
        score += 3
    elif row["Payment_Delay_Days"] > 15:
        score += 2
    elif row["Payment_Delay_Days"] > 0:
        score += 1

    # Monthly usage (low usage = high risk)
    if row["Monthly_Usage"] < df["Monthly_Usage"].quantile(0.25):
        score += 3
    elif row["Monthly_Usage"] < df["Monthly_Usage"].quantile(0.5):
        score += 2
    else:
        score += 1

    # Contract length (short = high risk)
    if row["Contract_Length"] <= 6:
        score += 3
    elif row["Contract_Length"] <= 12:
        score += 2
    else:
        score += 1

    # Support tickets
    if row["Support_Tickets"] > df["Support_Tickets"].quantile(0.75):
        score += 3
    elif row["Support_Tickets"] > df["Support_Tickets"].quantile(0.5):
        score += 2
    else:
        score += 1

    return score

df["Risk_Score"] = df.apply(compute_risk_score, axis=1)

# Categorize risk
def categorize_risk(score):
    if score <= 6:
        return "Low Risk"
    elif score <= 9:
        return "Medium Risk"
    else:
        return "High Risk"

df["Risk_Category"] = df["Risk_Score"].apply(categorize_risk)

# ===== Part C: Decision Tree Model =====
df_ml = df.copy()
df_ml["Renewal_Status_Bin"] = df_ml["Renewal_Status"].map({"Yes": 1, "No": 0})

X = pd.get_dummies(
    df_ml.drop(columns=["Client_ID", "Renewal_Status", "Renewal_Status_Bin"]),
    drop_first=True
)
y = df_ml["Renewal_Status_Bin"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
feature_importances = pd.Series(model.feature_importances_, index=X.columns)\
                      .sort_values(ascending=False)

st.subheader("Model Performance")
st.write(f"Accuracy: {accuracy:.2f}")

fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

st.subheader("Feature Importance")
st.bar_chart(feature_importances.head(10))

# ===== Part D: Filters =====
st.sidebar.header("Filters")
regions = st.sidebar.multiselect(
    "Region", options=df["Region"].unique(), default=list(df["Region"].unique())
)
industries = st.sidebar.multiselect(
    "Industry", options=df["Industry"].unique(), default=list(df["Industry"].unique())
)
risk_cats = st.sidebar.multiselect(
    "Risk Category", options=df["Risk_Category"].unique(),
    default=list(df["Risk_Category"].unique())
)

filtered_df = df[
    (df["Region"].isin(regions)) &
    (df["Industry"].isin(industries)) &
    (df["Risk_Category"].isin(risk_cats))
]

# ===== KPI Cards =====
st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

total_clients = len(filtered_df)
high_risk_clients = (filtered_df["Risk_Category"] == "High Risk").sum()

# Predict churn on filtered set
X_filtered = X.loc[filtered_df.index]
y_filtered_pred = model.predict(X_filtered)
pred_churn_rate = (1 - y_filtered_pred.mean()) * 100  # assuming 1 = renew

avg_revenue = filtered_df["Revenue"].mean()

col1.metric("Total Clients", total_clients)
col2.metric("High Risk Clients", int(high_risk_clients))
col3.metric("Predicted Churn Rate %", f"{pred_churn_rate:.1f}%")
col4.metric("Avg Revenue per Client", f"{avg_revenue:,.2f}")

# ===== Visualizations =====
st.subheader("Risk Category Distribution")
st.bar_chart(filtered_df["Risk_Category"].value_counts())

st.subheader("Industry-wise Risk Analysis")
industry_risk = filtered_df.groupby(["Industry", "Risk_Category"]).size().unstack(fill_value=0)
st.bar_chart(industry_risk)

st.subheader("Revenue vs Risk Scatter")
fig_scatter, ax_scatter = plt.subplots()
sns.scatterplot(
    data=filtered_df,
    x="Monthly_Usage",
    y="Revenue",
    hue="Risk_Category",
    ax=ax_scatter
)
st.pyplot(fig_scatter)

st.subheader("Contract Length vs Churn")
contract_churn = df_ml.groupby("Contract_Length")["Renewal_Status_Bin"].mean()
st.line_chart(contract_churn)

# ===== Top 20 High-Risk Clients =====
st.subheader("Top 20 High-Risk Clients")
top20 = df[df["Risk_Category"] == "High Risk"].sort_values(
    "Risk_Score", ascending=False
).head(20)
st.dataframe(top20[["Client_ID", "Industry", "Region", "Risk_Score",
                    "Revenue", "Renewal_Status"]])

# ===== Retention Suggestions =====
st.subheader("AI-Based Retention Suggestions")
if st.button("Generate Retention Strategy"):
    st.write("- Offer payment plans or discounts for clients with Payment_Delay_Days > 30.")
    st.write("- Assign dedicated account managers to clients with high Support_Tickets.")
    st.write("- Provide longer contract incentives to high-risk but high-revenue clients.")
    st.write("- Schedule quarterly business reviews for high-risk clients.")
    st.write("- Improve onboarding and training for low-usage clients.")

# ===== Responsible AI Section =====
st.subheader("Ethical Implications of Predicting Client Churn")
st.markdown(
    """
- Predictive models may encode bias against certain industries or regions if the data is unbalanced.
- Labeling clients as "High Risk" can affect how they are treated and may damage long-term relationships.
- Sensitive financial and usage data must be handled with strong privacy and security controls.
- Human decision-makers should use the model as support, not as the only basis for actions.
"""
)
