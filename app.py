# app.py

import streamlit as st
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# --- Title ---
st.title("📊 Statistical Hypothesis Testing App")
st.write("""
This app helps you analyze your data and automatically selects the appropriate statistical test based on the flowchart.
""")

# --- Display Reference Flowchart ---
st.header("📚 Reference Flowchart")
st.write("Use the flowchart below to understand how statistical tests are selected based on data type and assumptions.")
st.image('images/Flow Chart for Cont. and Disc..png', caption='Statistical Test Decision Tree', use_container_width=True)

# --- Step 1: Data Input ---
st.header("1️⃣ Data Input")
data_input_method = st.radio("How would you like to input data?", ('Upload CSV', 'Manual Entry'))

if data_input_method == 'Upload CSV':
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("📊 **Dataset Preview:**")
        st.write(data.head())
else:
    manual_data = st.text_area("Enter your data manually (comma-separated values):")
    if manual_data:
        try:
            data = pd.DataFrame({'ManualData': [float(x.strip()) for x in manual_data.split(',')]})
            st.write("📊 **Manual Data Preview:**")
            st.write(data)
        except ValueError:
            st.error("❌ Invalid input! Ensure you enter comm numeric values.")

# --- Step 2: Data Type Selection ---
if 'data' in locals():
    st.header("2️⃣ Data Type Selection")
    data_type = st.radio("What type of data are you analyzing?", ('Continuous', 'Discrete'))

    ## --- Continuous Data Workflow ---
    if data_type == 'Continuous':
        st.header("3️⃣ Assumption Check")
        
        # Normality Test (Shapiro-Wilk)
        st.subheader("Normality Test (Shapiro-Wilk)")
        shapiro_p = stats.shapiro(data.iloc[:, 0])[1]
        st.write(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
        normal = shapiro_p > 0.05
        
        # Homogeneity of Variance Test (Levene)
        homogeneous = True
        if data.shape[1] > 1:
            st.subheader("Homogeneity of Variance (Levene)")
            levene_p = stats.levene(*[data[col] for col in data.columns])[1]
            st.write(f"Levene p-value: {levene_p:.4f}")
            homogeneous = levene_p > 0.05
        
        parametric = normal and homogeneous
        
        st.write("✅ All assumptions hold. Proceed with **Parametric Tests**." if parametric 
                 else "❌ Assumptions violated. Proceed with **Non-Parametric Tests**.")
        
        # Group Selection
        st.header("4️⃣ Group Selection")
        group_selection = st.radio("Number of Groups:", ("One Sample", "Two Samples", "More than Two Samples"))
        if group_selection in ["Two Samples", "More than Two Samples"]:
            paired = st.radio("Are the groups Paired or Unpaired?", ("Paired", "Unpaired"))
        
        # Test Execution
        st.header("5️⃣ Run Test")
        if st.button("Run Test"):
            if parametric:
                if group_selection == "One Sample":
                    stat, p = stats.ttest_1samp(data.iloc[:, 0], 0)
                    st.write(f"One-Sample t-test p-value: {p:.4f}")
                elif group_selection == "Two Samples":
                    stat, p = (stats.ttest_rel if paired == "Paired" else stats.ttest_ind)(
                        data.iloc[:, 0], data.iloc[:, 1]
                    )
                    st.write(f"{'Paired' if paired == 'Paired' else 'Independent'} t-test p-value: {p:.4f}")
                elif group_selection == "More than Two Samples":
                    stat, p = stats.f_oneway(*[data[col] for col in data.columns])
                    st.write(f"{'Repeated Measures ANOVA' if paired == 'Paired' else 'One-Way ANOVA'} p-value: {p:.4f}")
            else:
                if group_selection == "One Sample":
                    stat, p = stats.wilcoxon(data.iloc[:, 0])
                    st.write(f"One-Sample Wilcoxon Test p-value: {p:.4f}")
                elif group_selection == "Two Samples":
                    stat, p = (stats.wilcoxon if paired == "Paired" else stats.mannwhitneyu)(
                        data.iloc[:, 0], data.iloc[:, 1]
                    )
                    st.write(f"{'Wilcoxon Signed-Rank Test' if paired == 'Paired' else 'Mann-Whitney U Test'} p-value: {p:.4f}")
                elif group_selection == "More than Two Samples":
                    stat, p = (stats.friedmanchisquare if paired == "Paired" else stats.kruskal)(
                        *[data[col] for col in data.columns]
                    )
                    st.write(f"{'Friedman Test' if paired == 'Paired' else 'Kruskal-Wallis Test'} p-value: {p:.4f}")
    
    ## --- Discrete Data Workflow ---
    if data_type == 'Discrete':
        st.header("3️⃣ Group Selection")
        group_selection = st.radio("Number of Groups:", ("One Sample", "Two Samples", "More than Two Samples"))
        if group_selection in ["Two Samples", "More than Two Samples"]:
            paired = st.radio("Are the groups Paired or Unpaired?", ("Paired", "Unpaired"))
        
        st.header("4️⃣ Run Test")
        if st.button("Run Test"):
            if group_selection == "One Sample":
                success = st.number_input("Enter number of successes:", min_value=0, value=1)
                trials = st.number_input("Enter number of trials:", min_value=1, value=10)
                p = stats.binom_test(success, trials)
                st.write(f"Binomial Test p-value: {p:.4f}")
            elif group_selection == "Two Samples":
                table = [[st.number_input('Cell 1,1'), st.number_input('Cell 1,2')],
                         [st.number_input('Cell 2,1'), st.number_input('Cell 2,2')]]
                stat, p = (stats.mcnemar if paired == "Paired" else stats.fisher_exact)(table)
                st.write(f"{'McNemar Test' if paired == 'Paired' else 'Fisher\'s Exact Test'} p-value: {p:.4f}")
            elif group_selection == "More than Two Samples":
                stat, p = (stats.cochrans_q if paired == "Paired" else stats.chisquanre)(
             *[data[col] for col in data.columns]
                )
                st.write(f"{'Cochran\'s Q Test' if paired == 'Paired' else 'Chi-Square Test'} p-value: {p:.4f}")
