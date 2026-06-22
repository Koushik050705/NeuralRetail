import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="NeuralRetail", page_icon="🛍️", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('data/clean_df.csv', parse_dates=['InvoiceDate'])
    rfm = pd.read_csv('data/rfm.csv')
    forecast = pd.read_csv('data/forecast.csv', parse_dates=['ds'])
    inventory = pd.read_csv('data/inventory.csv')
    return df, rfm, forecast, inventory

df, rfm, forecast, inventory = load_data()

st.sidebar.title("NeuralRetail")
st.sidebar.caption("AI-Powered Sales Intelligence")
page = st.sidebar.radio("Navigation", [
    "Executive Overview", "Sales Analytics", "Customer Hub",
    "Demand Explorer", "Churn Risk", "Inventory Health"
])

# ---------------- PAGE 1 ----------------
if page == "Executive Overview":
    st.title("📊 Executive Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${df['TotalPrice'].sum():,.2f}")
    c2.metric("Total Orders", df['Invoice'].nunique())
    c3.metric("Total Customers", df['Customer ID'].nunique())
    c4.metric("Avg Order Value", f"${df.groupby('Invoice')['TotalPrice'].sum().mean():,.2f}")

    country_rev = df.groupby('Country')['TotalPrice'].sum().nlargest(10).reset_index()
    st.plotly_chart(px.bar(country_rev, x='Country', y='TotalPrice',
                            title='Revenue by Country'), use_container_width=True)

# ---------------- PAGE 2 ----------------
elif page == "Sales Analytics":
    st.title("📈 Sales Analytics")
    df['Month'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    monthly = df.groupby('Month')['TotalPrice'].sum().reset_index()
    st.plotly_chart(px.line(monthly, x='Month', y='TotalPrice',
                             title='Monthly Revenue Trend'), use_container_width=True)

    top_products = df.groupby('Description')['TotalPrice'].sum().nlargest(10).reset_index()
    st.plotly_chart(px.bar(top_products, x='TotalPrice', y='Description',
                            orientation='h', title='Top Selling Products'),
                     use_container_width=True)

# ---------------- PAGE 3 ----------------
elif page == "Customer Hub":
    st.title("👥 Customer Hub")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.scatter(rfm, x='Recency', y='Frequency', color='Cluster',
                                    title='Customer Segments'), use_container_width=True)
    with col2:
        st.plotly_chart(px.pie(rfm, names='Segment', title='Segment Distribution'),
                         use_container_width=True)

# ---------------- PAGE 4 ----------------
elif page == "Demand Explorer":
    st.title("📉 Demand Explorer")
    st.subheader("Sales Forecast (Next 90 Days)")
    fig = px.line(forecast, x='ds', y=['yhat', 'yhat_lower', 'yhat_upper'],
                  title='90-Day Forecast with Confidence Interval')
    st.plotly_chart(fig, use_container_width=True)

# ---------------- PAGE 5 ----------------
elif page == "Churn Risk":
    st.title("⚠️ Churn Risk Assessment")
    high_risk = rfm[rfm['ChurnProbability'] > 0.85].sort_values(
        'ChurnProbability', ascending=False).head(20)
    st.subheader("High Risk Customers")
    st.dataframe(high_risk[['Customer ID', 'Recency', 'Frequency',
                             'Monetary', 'ChurnProbability']])
    st.download_button("Download high-risk list as CSV",
                        high_risk.to_csv(index=False), "high_risk_customers.csv")
    st.plotly_chart(px.histogram(rfm, x='ChurnProbability',
                                  title='Churn Probability Distribution'),
                     use_container_width=True)

# ---------------- PAGE 6 ----------------
elif page == "Inventory Health":
    st.title("📦 Inventory Health & Optimization")
    abc_counts = inventory['ABC_Category'].value_counts().reset_index()
    st.plotly_chart(px.pie(abc_counts, names='ABC_Category',
                            title='ABC Analysis Summary'), use_container_width=True)
    st.subheader("Recommended Reorder Quantities (EOQ)")
    st.dataframe(inventory[['StockCode', 'SalesPrice', 'ABC_Category', 'EOQ']].head(20))