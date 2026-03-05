import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="B2B Client Risk Intelligence Dashboard", layout="wide")
st.title("🛡️ B2B Client Risk Intelligence Dashboard")

# Part A: Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("B2B_Client_Churn_5000-2.csv")
    return df

df = load_data()

# Part B: Risk Scoring Logic
@st.cache_data
def compute_risk_score(df):
    df = df.copy()
    
    # Business logic-based risk score (0-12 points)
    def risk_score(row):
        score = 0
        
        # High payment delay → High risk
        if row['Payment_Delay_Days'] > 30:
            score += 3
        elif row['Payment_Delay_Days'] > 15:
            score += 2
        elif row['Payment_Delay_Days'] > 0:
            score += 1
            
        # Low usage → High risk
        usage_q = df['Monthly_Usage'].quantile(0.25)
        if row['Monthly_Usage'] < usage_q:
            score += 3
        elif row['Monthly_Usage'] < df['Monthly_Usage'].median():
            score += 2
        else:
            score += 1
            
        # Short contract → High risk
        if row['Contract_Length'] <= 6:
            score += 3
        elif row['Contract_Length'] <= 12:
            score += 2
        else:
            score += 1
            
        # High complaints → High risk
        tickets_q = df['Support_Tickets'].quantile(0.75)
        if row['Support_Tickets'] > tickets_q:
            score += 3
        elif row['Support_Tickets'] > df['Support_Tickets'].median():
            score += 2
        else:
            score += 1
            
        return score
    
    df['Risk_Score'] = df.apply(risk_score, axis=1)
    
    # Categorize into Low/Medium/High Risk
    def categorize(score):
        if score <= 6:
            return 'Low Risk'
        elif score <= 9:
            return 'Medium Risk'
        else:
            return 'High Risk'
    
    df['Risk_Category'] = df['Risk_Score'].apply(categorize)
    return df

df = compute_risk_score(df)

# Part C: Machine Learning - Decision Tree Classifier
@st.cache_data
def train_model(df):
    df_ml = df.copy()
    df_ml['Renewal_Status_Bin'] = df_ml['Renewal_Status'].map({'Yes': 1, 'No': 0})
    
    # Features (exclude ID and target)
    feature_cols = ['Monthly_Usage', 'Payment_Delay_Days', 'Contract_Length', 
                   'Support_Tickets', 'Revenue']
    X = pd.get_dummies(df_ml[['Industry', 'Region'] + feature_cols], drop_first=True)
    y = df_ml['Renewal_Status_Bin']
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42, max_depth=8)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, accuracy, confusion_matrix(y_test, y_pred), importance, X.columns

model, accuracy, cm, importance_df, feature_names = train_model(df)

# Sidebar filters (Part D)
st.sidebar.header("🔍 Filters")
region_options = st.sidebar.multiselect("Region", df['Region'].unique(), default=df['Region'].unique())
industry_options = st.sidebar.multiselect("Industry", df['Industry'].unique(), default=df['Industry'].unique())
risk_options = st.sidebar.multiselect("Risk Category", df['Risk_Category'].unique(), default=df['Risk_Category'].unique())

# Filter dataframe
filtered_df = df[
    (df['Region'].isin(region_options)) &
    (df['Industry'].isin(industry_options)) &
    (df['Risk_Category'].isin(risk_options))
]

# Part D: KPI Cards
col1, col2, col3, col4 = st.columns(4)
total_clients = len(filtered_df)
high_risk_count = (filtered_df['Risk_Category'] == 'High Risk').sum()
avg_revenue = filtered_df['Revenue'].mean()

# Predicted churn rate on filtered data
X_filtered = pd.get_dummies(filtered_df[['Industry', 'Region', 'Monthly_Usage', 
                                       'Payment_Delay_Days', 'Contract_Length', 
                                       'Support_Tickets', 'Revenue']], drop_first=True)
if len(X_filtered.columns) == len(feature_names):
    X_filtered = X_filtered.reindex(columns=feature_names, fill_value=0)
    churn_pred = model.predict(X_filtered)
    churn_rate = (1 - churn_pred.mean()) * 100

with col1:
    st.metric("Total Clients", total_clients)
with col2:
    st.metric("High Risk Clients", high_risk_count)
with col3:
    st.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")
with col4:
    st.metric("Avg Revenue/Client", f"${avg_revenue:,.0f}")

# Part D: Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Risk Category Distribution")
    risk_dist = filtered_df['Risk_Category'].value_counts()
    fig_risk = px.bar(x=risk_dist.index, y=risk_dist.values, 
                     title="Risk Distribution", color=risk_dist.index)
    st.plotly_chart(fig_risk, use_container_width=True)

with col2:
    st.subheader("🏭 Industry-wise Risk Analysis")
    industry_risk = filtered_df.groupby(['Industry', 'Risk_Category']).size().unstack(fill_value=0)
    fig_industry = px.bar(industry_risk, barmode='group', title="Industry Risk")
    st.plotly_chart(fig_industry, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("💰 Revenue vs Risk")
    fig_scatter = px.scatter(filtered_df, x='Monthly_Usage', y='Revenue', 
                           color='Risk_Category', size='Support_Tickets',
                           title="Revenue vs Usage (colored by Risk)")
    st.plotly_chart(fig_scatter, use_container_width=True)

with col4:
    st.subheader("📅 Contract Length vs Churn")
    contract_churn = filtered_df.groupby('Contract_Length')['Renewal_Status'].value_counts().unstack(fill_value=0)
    contract_churn_pct = contract_churn.div(contract_churn.sum(axis=1), axis=0) * 100
    fig_contract = px.line(contract_churn_pct, title="Churn Rate by Contract Length")
    st.plotly_chart(fig_contract, use_container_width=True)

# Top 20 High-Risk Clients Table
st.subheader("🚨 Top 20 High-Risk Clients")
top20 = filtered_df[filtered_df['Risk_Category'] == 'High Risk'].nlargest(20, 'Risk_Score')
st.dataframe(top20[['Client_ID', 'Industry', 'Region', 'Risk_Score', 'Revenue', 
                   'Payment_Delay_Days', 'Renewal_Status']].style.format({'Revenue': '${:,.0f}'}))

# Part C: ML Model Results
st.subheader("🤖 Machine Learning Model Performance")
col_ml1, col_ml2, col_ml3 = st.columns(3)
col_ml1.metric("Model Accuracy", f"{accuracy:.2%}")
col_ml2.metric("High Risk Clients", f"{high_risk_count}")
col_ml3.metric("Top Churn Factor", importance_df.iloc[0]['feature'])

st.subheader("Feature Importance (What drives churn most?)")
st.bar_chart(importance_df.head(10).set_index('feature')['importance'])

fig_cm, ax = plt.subplots(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Renewed', 'Churned'], 
            yticklabels=['Renewed', 'Churned'])
ax.set_title('Confusion Matrix')
st.pyplot(fig_cm)

# Part E: Retention Suggestions
st.subheader("💡 AI-Based Retention Strategies")
if st.button("🎯 Generate Retention Strategy"):
    st.markdown("""
    **Personalized Recommendations for High-Risk Clients:**
    
    1. **Payment Delay Relief**: Offer 10% discount or flexible payment plans for clients with Payment_Delay_Days > 30 days.
    2. **Account Manager Assignment**: Assign dedicated account managers to clients with Support_Tickets > median.
    3. **Contract Extension Incentives**: Provide 15% discount for extending contracts >12 months for short-term clients.
    4. **Usage Boost Program**: Offer free training sessions for clients with low Monthly_Usage (bottom 25%).
    5. **Revenue Recovery Focus**: Prioritize high-revenue high-risk clients with quarterly business reviews.
    """)

# Part F: Responsible AI Section
with st.expander("⚖️ Ethical Implications of Predicting Client Churn"):
    st.markdown("""
    **Key Ethical Considerations:**
    
    **1. Model Bias**: Decision trees may amplify biases in training data, unfairly flagging certain industries/regions as high-risk.
    
    **2. Labeling Impact**: "High Risk" labels could lead to reduced service quality or aggressive sales tactics, damaging relationships.
    
    **3. Data Privacy**: Financial data (Revenue, Payment_Delay_Days) must be handled with strict GDPR/CCPA compliance and access controls.
    
    **4. Responsible Use**: Predictions should support human judgment, not replace it. Regular model audits and transparency are essential.
    
    **5. Self-Fulfilling Prophecy**: Aggressive retention actions based on predictions might actually *cause* churn through poor customer experience.
    
    *This tool provides insights for proactive customer success, not automated decisions.*
    """)

# Footer
st.markdown("---")
st.markdown("*B2B Client Risk Intelligence Dashboard | Applied Programming Tools Assignment*")
