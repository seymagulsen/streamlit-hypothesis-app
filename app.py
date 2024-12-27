# app.py

import streamlit as st
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import ast

# --- Sidebar for Title and Flowchart ---
with st.sidebar:
    # TEDU Logo
    st.image('images/tedu_logo.png', caption='TED University', use_container_width=True)
    
    # App Title
    st.title("ADS 511: Statistical Inference Methods Web Application")
    st.write("This app helps you analyze your data and automatically selects the appropriate statistical test based on the flowchart.")
    st.markdown("---")

    # Author Information
    with st.expander("👩‍🎓 About the Developer"):
        st.write("Developed by **Şeyma Gülşen Akkuş**")
        st.write("MSc. in Data Science, TED University")
    
    st.markdown("---")
    
    # Reference Flowchart
    with st.expander("🗺️ Hypothesis Testing Map"):
        st.image('images/Flow Chart for Cont. and Disc..png', caption='Statistical Test Decision Tree', use_container_width=True)
        st.write("""Understand how hypothesis testing decisions are made based on assumptions and data type.""")
    
    st.markdown("---")
    
    # Quick Links
    with st.expander("🔗 Quick Links"):
        st.markdown("- [📖 SciPy Documentation](https://scipy.org)")
        st.markdown("- [💻 Streamlit Documentation](https://streamlit.io)")
        st.markdown("- [📊 Matplotlib Documentation](https://matplotlib.org)")

# --- Main Tabs ---
st.title("ADS 511: Statistical Inference Methods Web Application")

# Step Selector
step = st.selectbox(
    "Select Step:",
    ["1️⃣ Data Input", "2️⃣ Data Type Selection", "3️⃣ Assumption Check", "4️⃣ Group Selection", "5️⃣ Run Test"]
)

# Progress Bar
progress = ["1️⃣ Data Input", "2️⃣ Data Type Selection", "3️⃣ Assumption Check", "4️⃣ Group Selection", "5️⃣ Run Test"].index(step) + 1
st.progress(progress / 5)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Data Input", 
    "📊 Data Type Selection", 
    "🔍 Assumption Check", 
    "🧑‍🤝‍🧑 Group Selection", 
    "🚀 Run Test"
])

# --- Tab 1: Data Input ---
with tab1:
    st.header("1️⃣ Data Input")
    data_input_method = st.radio("How would you like to input data?", ('Upload CSV', 'Manual Entry'))

    if data_input_method == 'Upload CSV':
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.write("📊 **Dataset Preview:**")
            st.dataframe(data)
            st.bar_chart(data)
    else:
        manual_data = st.text_area("Enter your data manually (e.g., [2,3,4], [1,2,3]):")
        if manual_data:
            try:
                data_groups = ast.literal_eval(f'[{manual_data}]')
                if isinstance(data_groups, list) and all(isinstance(group, list) for group in data_groups):
                    data = pd.DataFrame({f'Group_{i+1}': group for i, group in enumerate(data_groups)})
                    st.write("📊 **Manual Data Preview:**")
                    st.dataframe(data)
                    st.line_chart(data)
                else:
                    raise ValueError("Invalid input format. Please use square brackets for groups.")
            except (ValueError, SyntaxError):
                st.error("❌ Invalid input! Ensure you use the correct format (e.g., [2,3,4], [1,2,3]).")

# --- Tab 2: Data Type Selection ---
with tab2:
    if 'data' in locals():
        st.header("2️⃣ Data Type Selection")
        data_type = st.radio("What type of data are you analyzing?", ('Continuous', 'Discrete'))

# --- Tab 3: Assumption Check ---
with tab3:
    if 'data' in locals() and data_type == 'Continuous':
        st.header("3️⃣ Assumption Check")
        
        # Normality Test (Shapiro-Wilk)
        with st.expander("🔍 Normality Test (Shapiro-Wilk)"):
            shapiro_p = stats.shapiro(data.iloc[:, 0])[1]
            st.write(f"**Shapiro-Wilk p-value:** {shapiro_p:.4f}")
            normal = shapiro_p > 0.05
            st.success("✅ Data is normally distributed." if normal else "❌ Data is not normally distributed.")
        
        # Homogeneity of Variance Test (Levene)
        homogeneous = True
        with st.expander("🔍 Homogeneity of Variance (Levene)"):
            if data.shape[1] > 1:
                levene_p = stats.levene(*[data[col] for col in data.columns])[1]
                st.write(f"**Levene p-value:** {levene_p:.4f}")
                homogeneous = levene_p > 0.05
                st.success("✅ Variances are homogeneous." if homogeneous else "❌ Variances are not homogeneous.")
        
        # Parametric Test Decision
        parametric = normal and homogeneous
        
        st.markdown("---")
        if parametric:
            st.success("✅ **All assumptions hold. Proceed with Parametric Tests.**")
        else:
            st.warning("❌ **Assumptions violated. Proceed with Non-Parametric Tests.**")


# --- Tab 4: Group Selection ---
with tab4:
    if 'data' in locals():
        st.header("4️⃣ Group Selection")
        group_selection = st.radio("Number of Groups:", ("One Sample", "Two Samples", "More than Two Samples"))
        if group_selection in ["Two Samples", "More than Two Samples"]:
            paired = st.radio("Are the groups Paired or Unpaired?", ("Paired", "Unpaired"))

# --- Tab 5: Run Test ---
with tab5:
    if 'data' in locals():
        st.header("5️⃣ Run Test")

        # Users can input additional parameters for the test
        additional_params = {}

        # Continuous Data Tests
        if data_type == 'Continuous':
            st.write("📊 **Continuous Data Summary:**")
            st.write(data.describe().T)
            if group_selection == "One Sample":
                st.write("📝 **One Sample Test Parameters:**")
                additional_params['population_mean'] = st.number_input("Enter the Population Mean (μ₀) for comparison:",
                                                                      min_value=-1000.0,
                                                                      max_value=1000.0,
                                                                      value=0.0,
                                                                      step=0.1)
            elif group_selection == "Two Samples":
                if paired == "Paired":
                    st.write("📝 **Paired Test Parameters:**")
                else:
                    st.write("📝 **Independent Test Parameters:**")
            elif group_selection == "More than Two Samples":
                if paired == "Paired":
                    st.write("📝 **Repeated Measures ANOVA Parameters:**")
                else:
                    st.write("📝 **One-Way ANOVA Parameters:**")
        
        # Discrete Data Tests
        if data_type == 'Discrete':
            st.write("📊 **Discrete Data Summary:**")
            st.write(data.describe().T)
            if group_selection == "One Sample":
                st.write("📝 **One Sample Test Parameters:**")
                additional_params['success'] = st.number_input("Enter number of successes:", min_value=0, value=1)
                additional_params['trials'] = st.number_input("Enter number of trials:", min_value=1, value=10)
            elif group_selection == "Two Samples":
                st.write("📝 **Two Sample Test Parameters:**")
                table = [[st.number_input('Cell 1,1'), st.number_input('Cell 1,2')],
                         [st.number_input('Cell 2,1'), st.number_input('Cell 2,2')]]
                additional_params['table'] = table
            elif group_selection == "More than Two Samples":
                if paired == "Paired":
                    st.write("📝 **Repeated Measures Test Parameters:**")
                else:
                    st.write("📝 **Chi-Square Test Parameters:**"
                               "Enter the data for each group in the format: [1,2,3], [4,5,6], ...")
                    additional_params['data'] = st.text_area("Enter the data for each group:")

        # Run Test Button
        if st.button("Run Test"):
            try:
                ## --- Continuous Data Workflow ---
                if data_type == 'Continuous':
                    if parametric:  # Parametric Tests
                        if group_selection == "One Sample":
                            population_mean = additional_params['population_mean']
                            stat, p = stats.ttest_1samp(data.iloc[:, 0], population_mean)
                            # Display the results of the one-sample t-test
                            st.toast("✅ Test Completed Successfully!", icon="🎯")
                            st.write(f"**Population Mean (μ₀):** {population_mean:.4f}")
                            st.write(f"**Test Statistic:** {stat:.4f}")
                            st.write(f"**One-Sample t-test p-value:** {p:.4f}")

                        elif group_selection == "Two Samples":
                            if paired == "Paired":
                                stat, p = stats.ttest_rel(data.iloc[:, 0], data.iloc[:, 1])
                                st.write(f"**Paired t-test p-value:** {p:.4f}")
                            else:
                                stat, p = stats.ttest_ind(data.iloc[:, 0], data.iloc[:, 1])
                                st.write(f"**Independent t-test p-value:** {p:.4f}")
                        elif group_selection == "More than Two Samples":
                            if paired == "Paired":
                                stat, p = stats.f_oneway(*[data[col] for col in data.columns])
                                st.write(f"**Repeated Measures ANOVA p-value:** {p:.4f}")
                            else:
                                stat, p = stats.f_oneway(*[data[col] for col in data.columns])
                                st.write(f"**One-Way ANOVA p-value:** {p:.4f}")
                    
                    else:  # Non-Parametric Tests
                        if group_selection == "One Sample":
                            stat, p = stats.wilcoxon(data.iloc[:, 0])
                            st.write(f"**One-Sample Wilcoxon Signed-Rank Test p-value:** {p:.4f}")
                        elif group_selection == "Two Samples":
                            if paired == "Paired":
                                stat, p = stats.wilcoxon(data.iloc[:, 0], data.iloc[:, 1])
                                st.write(f"**Wilcoxon Signed-Rank Test p-value:** {p:.4f}")
                            else:
                                stat, p = stats.mannwhitneyu(data.iloc[:, 0], data.iloc[:, 1])
                                st.write(f"**Mann-Whitney U Test p-value:** {p:.4f}")
                        elif group_selection == "More than Two Samples":
                            if paired == "Paired":
                                stat, p = stats.friedmanchisquare(*[data[col] for col in data.columns])
                                st.write(f"**Friedman Test p-value:** {p:.4f}")
                            else:
                                stat, p = stats.kruskal(*[data[col] for col in data.columns])
                                st.write(f"**Kruskal-Wallis H Test p-value:** {p:.4f}")
                
                ## --- Discrete Data Workflow ---
                if data_type == 'Discrete':
                    if group_selection == "One Sample":
                        success = st.number_input("Enter number of successes:", min_value=0, value=1)
                        trials = st.number_input("Enter number of trials:", min_value=1, value=10)
                        p = stats.binom_test(success, trials)
                        st.write(f"**Binomial Test p-value:** {p:.4f}")
                    
                    elif group_selection == "Two Samples":
                        table = [[st.number_input('Cell 1,1'), st.number_input('Cell 1,2')],
                                 [st.number_input('Cell 2,1'), st.number_input('Cell 2,2')]]
                        if paired == "Paired":
                            stat, p = stats.mcnemar(table)
                            st.write(f"**McNemar Test p-value:** {p:.4f}")
                        else:
                            stat, p = stats.fisher_exact(table)
                            st.write(f"**Fisher's Exact Test p-value:** {p:.4f}")
                    
                    elif group_selection == "More than Two Samples":
                        if paired == "Paired":
                            stat, p = stats.cochrans_q(*[data[col] for col in data.columns])
                            st.write(f"**Cochran's Q Test p-value:** {p:.4f}")
                        else:
                            stat, p = stats.chisquare(*[data[col] for col in data.columns])
                            st.write(f"**Chi-Square Test p-value:** {p:.4f}")
                
                ## --- Final Result Message ---
                if p < 0.05:
                    st.success("✅ **Statistically Significant Result:** Reject Null Hypothesis")
                    st.info("This means there is sufficient evidence to support the alternative hypothesis.")
                else:
                    st.warning("❌ **Not Statistically Significant:** Fail to Reject Null Hypothesis")
                    st.info("There is insufficient evidence to support the alternative hypothesis.")

            except Exception as e:
                st.error(f"❌ **Error:** {e}")
                st.info("Please ensure you have selected the correct data type and group selection.")