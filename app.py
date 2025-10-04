import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# Streamlit Dashboard Template
# ==============================

st.set_page_config(page_title="Data Dashboard", layout="wide")

st.title("📊 Data Analysis Dashboard")
st.markdown("Upload your CSV file and explore your data interactively!")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    # --- Dataset Overview ---
    st.header("🔎 Dataset Overview")
    st.write("**Shape:**", df.shape)
    st.write("**Preview:**")
    st.dataframe(df.head())

    # --- Column Summary ---
    st.header("📈 Summary Statistics")
    st.write(df.describe(include="all"))

    # --- Data Visualization ---
    st.header("📊 Data Visualization")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    tab1, tab2 = st.tabs(["📉 Numeric Analysis", "🔤 Categorical Analysis"])

    with tab1:
        if numeric_cols:
            x_col = st.selectbox("Select X-axis", numeric_cols)
            y_col = st.selectbox("Select Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
            
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            st.pyplot(fig)

            st.write("**Correlation Heatmap**")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns found.")

    with tab2:
        if cat_cols:
            col = st.selectbox("Select a categorical column", cat_cols)
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind='bar', ax=ax)
            plt.title(f"Distribution of {col}")
            st.pyplot(fig)
        else:
            st.warning("No categorical columns found.")

    # --- Data Download ---
    st.header("💾 Download Processed Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='processed_data.csv',
        mime='text/csv',
    )

else:
    st.info("👆 Upload a CSV file to start exploring your data.")
