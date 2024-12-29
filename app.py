# app.py

import streamlit as st
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
from scipy.stats import fisher_exact, chi2_contingency
import ast
import sys
import os

# Explicitly add the site-packages path to sys.path
sys.path.append(os.path.expanduser('~/.local/lib/python3.12/site-packages'))

# Now import scikit-posthocs
import scikit_posthocs as sp


# --- Session State Initialization ---

if 'step_completed' not in st.session_state:
    st.session_state['step_completed'] = {
        'Data Input': False,
        'Data Type': False,
        'Assumption Check': False,
        'Group Selection': False,
        'Run Test': False
    }

if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'data_type' not in st.session_state:
    st.session_state['data_type'] = None
if 'group_selection' not in st.session_state:
    st.session_state['group_selection'] = None
if 'parametric' not in st.session_state:
    st.session_state['parametric'] = False
if 'paired' not in st.session_state:
    st.session_state['paired'] = None
if 'current_tab' not in st.session_state:
    st.session_state['current_tab'] = st.query_params.get("tab", "📂 Data Input")


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
        st.write("**Şeyma Gülşen Akkuş**")
        st.write("MSc. in Applied Data Science, TED University")
    
    st.markdown("---")

    st.write("**Progress:**")
    for step, completed in st.session_state['step_completed'].items():
        status = "✅" if completed else "⏳"
        st.write(f"{status} {step}")
    st.markdown("---")
    
    # Hypothesis Testing Steps
    with st.expander("📚 Hypothesis Testing Steps"):
        st.write("### 📝 Define the Hypotheses:")
        st.write("- Define the null (H₀) and alternative (H₁) hypotheses clearly.")
        
        st.write("### ✅ Verify Assumptions:")
        st.write("""
        - Check the necessary conditions for the statistical test:
            - Normality of data distribution.
            - Independence and identical distribution of samples.
            - Absence of significant outliers.
            - Homogeneity of variances (for certain tests).
        """)
        
        st.write("### 🧠 Select the Appropriate Test:")
        st.write("""
        - Parametric tests: Used when all assumptions are met.
        - Non-parametric tests: Used when assumptions are not met.
        """)
        
        st.write("### 📊 Calculate the Test Statistic and p-Value:")
        st.write("""
        - Compute the test statistic from the sample data.
        - Derive the p-value, which measures the probability of observing the data under H₀.
        """)
        
        st.write("### 🎯 Decision Making:")
        st.write("""
        - Compare the test statistic to a critical value or the p-value to the significance level (α).
        - α is the probability of rejecting the null hypothesis when it is true.
        
        **Two possible outcomes:**
        - Reject (H₀): Evidence supports the alternative hypothesis.
        - Fail to reject (H₀): Insufficient evidence to support the alternative hypothesis.
        """)

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

st.title("ADS 511: Statistical Inference Methods Web Application")
tabs = ["📂 Data Input", "📊 Data Type Selection", "🔍 Assumption Check", "🧑‍🤝‍🧑 Group Selection", "🚀 Run Test"]
if st.session_state['current_tab'] not in tabs:
    st.session_state['current_tab'] = "📂 Data Input"  # Ensure it defaults to the first tab if mismatched

# Tab Selection
# Navigation
selected_tab = st.radio("Navigation", tabs, index=tabs.index(st.session_state['current_tab']))
st.session_state['current_tab'] = selected_tab
st.query_params["tab"] = selected_tab


# --- Tab 1: Data Input ---
if selected_tab == "📂 Data Input":
    st.header("1️⃣ Data Input")
    data_input_method = st.radio("Choose Data Input Method:", ('Upload CSV', 'Manual Entry'))

    if data_input_method == 'Upload CSV':
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        if uploaded_file:
            st.session_state['data'] = pd.read_csv(uploaded_file)
            st.write("📊 Dataset Preview:")
            st.dataframe(st.session_state['data'])
            st.bar_chart(st.session_state['data'])
    else:
        manual_data = st.text_area("Enter data manually (e.g., [2,3,4], [1,2,3]):")
        if manual_data:
            try:
                parsed_data = ast.literal_eval(f'[{manual_data}]')
                if all(isinstance(group, list) for group in parsed_data):
                    st.session_state['data'] = pd.DataFrame({f'Group_{i+1}': group for i, group in enumerate(parsed_data)})
                    st.write("📊 Dataset Preview:")
                    st.dataframe(st.session_state['data'])
                    st.line_chart(st.session_state['data'])
                else:
                    st.error("Invalid format! Ensure all groups are lists.")
            except Exception as e:
                st.error(f"Data Parsing Error: {e}")
    
    if st.session_state['data'] is not None:
        if st.button("Next: Data Type Selection"):
            st.session_state['step_completed']['Data Input'] = True
            st.session_state['current_tab'] = '📊 Data Type Selection'
            st.query_params["tab"] = "📊 Data Type Selection"
            st.write("Data Input Completed! Proceed to the next step.")
            st.rerun()  # Refresh the page to update the sidebar progress

# --- Tab 2: Data Type Selection ---
if selected_tab == "📊 Data Type Selection":
    st.header("2️⃣ Data Type Selection")
    if not st.session_state['step_completed']['Data Input']:
        st.warning("⚠️ Please complete 'Data Input' first.")
    else:
        
        st.session_state['data_type'] = st.radio("Select Data Type:", ['Continuous', 'Discrete'])

        if st.button("Next"):
            st.session_state['step_completed']['Data Type'] = True
            if st.session_state['data_type'] == "Discrete":
                # Skip Assumption Check for Discrete Data
                st.session_state['step_completed']['Assumption Check'] = True
                st.session_state['current_tab'] = '🧑‍🤝‍🧑 Group Selection'
                st.query_params["tab"] = "🧑‍🤝‍🧑 Group Selection"
                st.write("Discrete data selected. Skipping Assumption Check. Proceeding to Group Selection...")
            else:
                # Proceed to Assumption Check for Continuous Data
                st.session_state['current_tab'] = '🔍 Assumption Check'
                st.query_params["tab"] = "🔍 Assumption Check"
                st.write("Data Type Selection Completed! Proceed to Assumption Check.")
            st.rerun()


# --- Tab 3: Assumption Check ---
if selected_tab == "🔍 Assumption Check":
    st.header("3️⃣ Assumption Check")
    
    if not st.session_state['step_completed']['Data Type']:
        st.warning("⚠️ Please complete 'Data Type Selection' first.")
    else:

        st.write("Performing assumption checks...")

        # Initialize assumption flags
        normal = False
        homogeneous = False
        independence_check = False
        outliers = 0

        # Normality Test (Shapiro-Wilk Test)
        with st.expander("🔍 Normality Test (Shapiro-Wilk)"):
            try:
                shapiro_p = stats.shapiro(st.session_state['data'].iloc[:, 0])[1]
                st.write(f"**Shapiro-Wilk p-value:** {shapiro_p:.4f}")
                normal = shapiro_p > 0.05
                st.success("✅ Data is normally distributed." if normal else "❌ Data is not normally distributed.")
            except Exception as e:
                st.error(f"Normality Test Error: {e}")

        # Homogeneity of Variances (Levene's Test)
        with st.expander("🔍 Homogeneity of Variances (Levene Test)"):
            try:
                if st.session_state['data'].shape[1] > 1:
                    levene_p = stats.levene(*[st.session_state['data'][col] for col in st.session_state['data'].columns])[1]
                    st.write(f"**Levene p-value:** {levene_p:.4f}")
                    homogeneous = levene_p > 0.05
                    st.success("✅ Variances are homogeneous." if homogeneous else "❌ Variances are not homogeneous.")
                else:
                    st.info("ℹ️ At least two groups are needed to test homogeneity of variances.")
                    homogeneous = True  # Assume True if only one group is present
            except Exception as e:
                st.error(f"Homogeneity Test Error: {e}")

        # Independence Check
        with st.expander("🔗 Independence and Identically Distributed Samples"):
            st.write("""
            - Samples should be collected independently.
            - Each observation should not influence another observation.
            - Samples should follow the same distribution.
            """)
            independence_check = st.checkbox("✅ Check if samples are independent and identically distributed")
            st.success("✅ Samples are independent and identically distributed." if independence_check else "❌ Samples might not be independent or identically distributed.")

        # Outlier Detection (Z-Score Method)
        with st.expander("⚠️ Absence of Significant Outliers"):
            try:
                z_scores = (st.session_state['data'] - st.session_state['data'].mean()) / st.session_state['data'].std()
                outliers = (z_scores.abs() > 3).sum().sum()
                st.write(f"**Number of Outliers Detected:** {outliers}")
                st.success("✅ No significant outliers detected." if outliers == 0 else f"⚠️ {outliers} significant outlier(s) detected.")
            except Exception as e:
                st.error(f"Outlier Detection Error: {e}")

        # Final Decision on Parametric/Non-Parametric Tests
        st.markdown("---")
        st.session_state['parametric'] = normal and homogeneous and independence_check and outliers == 0
        
        if st.session_state['parametric']:
            st.success("✅ **All assumptions hold. Proceed with Parametric Tests.**")
        else:
            st.warning("❌ **Assumptions violated. Proceed with Non-Parametric Tests.**")

        # Proceed to the Next Step
        if st.button("Next: Group Selection"):
            st.session_state['step_completed']['Assumption Check'] = True
            st.session_state['current_tab'] = '🧑‍🤝‍🧑 Group Selection'
            st.query_params["tab"] = "🧑‍🤝‍🧑 Group Selection"
            st.write("Assumption Check Completed! Proceed to the next step.")
            st.rerun()

# --- Tab 4: Group Selection ---
if selected_tab == "🧑‍🤝‍🧑 Group Selection":
    st.header("4️⃣ Group Selection")
    
    if not st.session_state['step_completed']['Assumption Check']:
        st.warning("⚠️ Please complete 'Assumption Check' first.")
    else:
        st.write("Select the appropriate group type based on your dataset and test requirements.")
        
        # Group Selection
        st.session_state['group_selection'] = st.radio(
            "Select Group Type:", 
            ["One Sample", "Two Samples", "More than Two Samples"],
            index=0
        )
        
        # Conditional Options Based on Group Selection
        if st.session_state['group_selection'] == "Two Samples":
            st.session_state['paired'] = st.radio(
                "Are the two samples paired or unpaired?",
                ["Paired", "Unpaired"],
                index=1,
                help="Select 'Paired' if the two samples are related or matched (e.g., pre-test/post-test)."
            )
        
        elif st.session_state['group_selection'] == "More than Two Samples":
            st.session_state['paired'] = st.radio(
                "Are the groups paired or unpaired?",
                ["Paired", "Unpaired"],
                index=1,
                help="Select 'Paired' for repeated measures (e.g., same subjects measured multiple times)."
            )
        
        # Display Selection Summary
        st.markdown("---")
        st.write("### 📊 **Your Selection:**")
        st.write(f"- **Group Type:** {st.session_state['group_selection']}")
        if st.session_state['group_selection'] in ["Two Samples", "More than Two Samples"]:
            st.write(f"- **Sample Type:** {st.session_state['paired']}")
        
        # Proceed Button
        if st.button("Next: Run Test"):
            if st.session_state['group_selection'] == "Two Samples" and 'paired' not in st.session_state:
                st.error("⚠️ Please select whether the samples are Paired or Unpaired.")
            elif st.session_state['group_selection'] == "More than Two Samples" and 'paired' not in st.session_state:
                st.error("⚠️ Please select whether the groups are Paired or Unpaired.")
            else:
                st.session_state['step_completed']['Group Selection'] = True
                st.session_state['current_tab'] = '🚀 Run Test'
                st.query_params["tab"] = "🚀 Run Test"
                st.write("Group Selection Completed! Proceed to the next step.")
                st.rerun()


# --- Tab 5: Run Test ---
if selected_tab == "🚀 Run Test":
    st.header("5️⃣ Run Statistical Test")
    
    if not st.session_state['step_completed']['Group Selection']:
        st.warning("⚠️ Please complete 'Group Selection' first.")
    else:
        st.write("### 🧠 **Test Configuration**")
        
        # Hypothesis Type Selection
        alternative = st.selectbox(
            "Choose Hypothesis Type:",
            ("two-sided", "greater", "less"),
            index=0,
            help=(
                "- **Two-sided:** Detects any difference.\n"
                "- **Greater:** Tests if the observed value is greater.\n"
                "- **Less:** Tests if the observed value is less."
            )
        )

        # Initialize Parameters
        data_type = st.session_state['data_type']
        group_selection = st.session_state['group_selection']
        paired = st.session_state.get('paired', None)
        data = st.session_state['data']
        parametric = st.session_state['parametric']

        additional_params = {}
        
        
        # 📊 **Continuous Data Tests**
        if data_type == "Continuous":
            st.subheader("📊 Continuous Data Test Parameters")
            if group_selection == "One Sample":
                additional_params['population_mean'] = st.number_input(
                    "Enter Population Mean (μ₀) for comparison:",
                    min_value=-1000.0,
                    max_value=10000.0,
                    value=0.0,
                    step=0.1
                )
            
            elif group_selection in ["Two Samples", "More than Two Samples"]:
                st.write("🧠 Sample Type:")
                st.write(f"- **Group Selection:** {group_selection}")
                st.write(f"- **Sample Type:** {paired}")
        
        # 📊 **Discrete Data Tests**
        if data_type == "Discrete":
            st.subheader("📊 Discrete Data Test Parameters")

            #---- One Sample ----
            if group_selection == "One Sample":
                st.write("One Sample Test Parameters:")
                additional_params['success'] = st.number_input("Enter Number of Successes:", 
                                                               min_value=0, 
                                                               value=1)
                additional_params['trials'] = st.number_input("Enter Number of Trials:", 
                                                              min_value=1, 
                                                              value=1)
            elif group_selection == "Two Samples":

                if paired == "Paired":
                    st.write("McNemar Test Parameters:")
                    additional_params['yes_yes'] = st.number_input("Condition Met in Both Scenarios (Yes-Yes):", min_value=0, value=0)
                    additional_params['yes_no'] = st.number_input("Condition Changed from Yes to No (Yes-No):", min_value=0, value=0)
                    additional_params['no_yes'] = st.number_input("Condition Changed from No to Yes (No-Yes):", min_value=0, value=0)
                    additional_params['no_no'] = st.number_input("Condition Not Met in Both Scenarios (No-No):", min_value=0, value=0)
                else:
                    st.write("Fisher's Exact Test Parameters:")
                    st.write("Enter values for a 2x2 contingency table:")
                    additional_params['group1_yes'] = st.number_input("Group 1: Condition Met (Yes):", min_value=0, value=0)
                    additional_params['group1_no'] = st.number_input("Group 1: Condition Not Met (No):", min_value=0, value=0)
                    additional_params['group2_yes'] = st.number_input("Group 2: Condition Met (Yes):", min_value=0, value=0)
                    additional_params['group2_no'] = st.number_input("Group 2: Condition Not Met (No):", min_value=0, value=0)
            
            elif group_selection == "More than Two Samples":

                if paired == "Paired":
                    st.write("Cochran's Q Test Parameters:")
                    st.write("Enter values for a contingency table:")
                    additional_params['paired_rows'] = st.number_input("Number of Rows (Groups)", min_value=2, value=2)
                    additional_params['paired_cols'] = st.number_input("Number of Columns (Samples)", min_value=1, value=2)
                    additional_params['paired_data'] =  [
                        [st.number_input(f"Value for Group {i+1}, Sample {j+1}", min_value=0, value=0)
                         for j in range(additional_params['paired_cols'])] 
                         for i in range(additional_params['paired_rows'])
                         ]
                else:
                    st.write("Chi-Square Test Parameters:")
                    st.write("Enter values for a contingency table:")
                    additional_params['chi2_rows'] = st.number_input("Number of Rows (Groups)", min_value=2, value=2)
                    additional_params['chi2_cols'] = st.number_input("Number of Columns (Samples)", min_value=1, value=2)
                    additional_params['chi2_table'] =  [
                        [st.number_input(f"Value for Group {i+1}, Sample {j+1}", min_value=0, value=0)
                         for j in range(additional_params['chi2_cols'])] 
                         for i in range(additional_params['chi2_rows'])
                         ]

        # Run Test Button
        if st.button("Run Test"):
            try:

                ## --- Continuous Data Workflow ---
                if data_type == 'Continuous':
                    if parametric:
                        # Parametric Tests
                        if group_selection == "One Sample":
                            st.subheader("🧪 **One Sample Test: One Sample t-test**")
                            population_mean = additional_params['population_mean']
                            stat, p = stats.ttest_1samp(data.iloc[:, 0], population_mean, alternative=alternative)
                            st.write(f"**Population Mean (μ₀):** {population_mean:.4f}")
                            st.write(f"**One Sample t-test Statistic:** {stat:.4f},**p-value:** {p:.4f}")

                        elif group_selection == "Two Samples":
                            if paired == "Paired":
                                st.subheader("🧪 **Two Sample Paired Test: Paired t-test**")
                                stat, p = stats.ttest_rel(data.iloc[:, 0], data.iloc[:, 1], alternative=alternative)
                                st.write(f"**Paired t-test Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                            else:
                                st.subheader("🧪 **Two Sample Unpaired Test: Independent t-test**")
                                stat, p = stats.ttest_ind(data.iloc[:, 0], data.iloc[:, 1], alternative=alternative)
                                st.write(f"**Independent t-test Statistic:** {stat:.4f}, **p-value:** {p:.4f}")

                        elif group_selection == "More than Two Samples":
                            if paired == "Paired":
                                # Repeated Measures ANOVA
                                st.subheader("🧪 **Repeated Measures ANOVA: One-Way ANOVA**")
                                stat, p = stats.f_oneway(*[data[col] for col in data.columns])
                                st.write(f"**One-Way ANOVA F-Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                                if p < 0.05:
                                    import scikit_posthocs as sp
                                    # Ensure proper input formatting
                                    import scikit_posthocs as sp
                                    long_data = pd.melt(data.reset_index(), id_vars='index', var_name='group', value_name='value')
                                    long_data = long_data.dropna()  # Remove NaN values
                                    posthoc_df = sp.posthoc_ttest(long_data, val_col='value',group_col='group', p_adjust='bonferroni')
                                    # Style the results manually for highlighting
                                    def highlight_significant(val):
                                        return 'background-color: lightblue' if val < 0.05 else ''
                                    posthoc_styled = posthoc_df.style.format("{:.6f}").applymap(highlight_significant)
                                    st.success("✅ Significant Differences Found! Performing Pairwise T-tests...")  
                                    st.write("🔍 **Pairwise T-Test Results (Bonferroni Corrected):**")
                                    st.write(posthoc_styled.to_html(), unsafe_allow_html=True) 

                            else:
                                # One-Way ANOVA
                                st.subheader("🧪 **One-Way ANOVA: One-Way ANOVA**")
                                stat, p = stats.f_oneway(*[data[col] for col in data.columns])
                                st.write(f"**One-Way ANOVA F-Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                                if p < 0.05:
                                    import scikit_posthocs as sp
                                    long_data = pd.melt(data.reset_index(), id_vars='index', var_name='group', value_name='value')
                                    long_data = long_data.dropna()  # Remove NaN values
                                    posthoc_df = sp.posthoc_ttest(long_data, val_col='value',group_col='group', p_adjust='bonferroni')
                                    # Style the results manually for highlighting
                                    def highlight_significant(val):
                                        return 'background-color: lightblue' if val < 0.05 else ''
                                    posthoc_styled = posthoc_df.style.format("{:.6f}").applymap(highlight_significant)
                                    st.success("✅ Significant Differences Found! Performing Pairwise T-tests...")  
                                    st.write("🔍 **Pairwise T-Test Results (Bonferroni Corrected):**")
                                    st.write(posthoc_styled.to_html(), unsafe_allow_html=True)  

                    else:
                        # Non-Parametric Tests
                        if group_selection == "One Sample":
                            st.subheader("🧪 **Non-Parametric One Sample Test: Wilcoxon Signed-Rank Test**")
                            population_mean = additional_params['population_mean']
                            diff = data.iloc[:, 0] - population_mean
                            stat, p = stats.wilcoxon(diff, alternative=alternative)
                            st.write(f"**Wilcoxon Signed-Rank Test Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                        elif group_selection == "Two Samples":
                            if paired == "Paired":
                                st.subheader("🧪 **Non-Parametric Two-Sample Paired Test: Wilcoxon Signed-Rank Test**")
                                stat, p = stats.wilcoxon(data.iloc[:, 0], data.iloc[:, 1], alternative=alternative)
                                st.write(f"**Wilcoxon Signed-Rank Test Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                            else:
                                st.subheader("🧪 **Non-Parametric Two-Sample Unpaired Test: Mann-Whitney U Test**")
                                stat, p = stats.mannwhitneyu(data.iloc[:, 0], data.iloc[:, 1], alternative=alternative)
                                st.write(f"**Mann-Whitney U Test Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                        elif group_selection == "More than Two Samples":
                            if paired == "Paired":
                                st.subheader("🧪 **Non-Parametric Repeated Measures Test: Friedman Test**")
                                stat, p = stats.friedmanchisquare(*[data[col] for col in data.columns])
                                st.write(f"**Friedman Test Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                                if p < 0.05:
                                    import scikit_posthocs as sp
                                    long_data = pd.melt(data.reset_index(), id_vars='index', var_name='group', value_name='value')
                                    long_data = long_data.dropna()  # Remove NaN values
                                    posthoc_df = sp.posthoc_ttest(long_data, val_col='value',group_col='group', p_adjust='bonferroni')
                                    # Style the results manually for highlighting
                                    def highlight_significant(val):
                                        return 'background-color: lightblue' if val < 0.05 else ''
                                    posthoc_styled = posthoc_df.style.format("{:.6f}").applymap(highlight_significant)
                                    st.success("✅ Significant Differences Found! Performing Pairwise T-tests...")  
                                    st.write("🔍 **Pairwise T-Test Results (Bonferroni Corrected):**")
                                    st.write(posthoc_styled.to_html(), unsafe_allow_html=True) 
                            else:
                                st.subheader("🧪 **Non-Parametric One-Way Test: Kruskal-Wallis H Test**")
                                stat, p = stats.kruskal(*[data[col] for col in data.columns])
                                st.write(f"**Kruskal-Wallis H Test Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                                if p < 0.05:
                                    import scikit_posthocs as sp
                                    long_data = pd.melt(data.reset_index(), id_vars='index', var_name='group', value_name='value')
                                    long_data = long_data.dropna()  # Remove NaN values
                                    posthoc_df = sp.posthoc_ttest(long_data, val_col='value',group_col='group', p_adjust='bonferroni')
                                    # Style the results manually for highlighting
                                    def highlight_significant(val):
                                        return 'background-color: lightblue' if val < 0.05 else ''
                                    posthoc_styled = posthoc_df.style.format("{:.6f}").applymap(highlight_significant)
                                    st.success("✅ Significant Differences Found! Performing Pairwise T-tests...")  
                                    st.write("🔍 **Pairwise T-Test Results (Bonferroni Corrected):**")
                                    st.write(posthoc_styled.to_html(), unsafe_allow_html=True) 

                ## --- Discrete Data Workflow ---
                if data_type == 'Discrete':
                    ## --- One Sample ---
                    if group_selection == "One Sample":
                        st.subheader("🧪 **One Sample Test: Binomial Test**")
                       
                        try:
                            result = stats.binomtest(
                                additional_params['success'],
                                additional_params['trials'],
                                p=0.5,
                                alternative=alternative)
                            
                            p = result.pvalue
                            st.write(f"**Number of Successes:** {additional_params['success']}")
                            st.write(f"**Number of Trials:** {additional_params['trials']}")
                            st.write(f"**Binomial Test p-value:** {p:.4f}")
                        except Exception as e:
                            st.error(f"❌ **Error:** {e}")
                            st.info("Please ensure you have entered valid values for the binomial test.")

                    ## --- Two Samples ---
                    elif group_selection == "Two Samples":
                        if paired == "Paired":
                            st.subheader("🧪 **Paired Test: McNemar Test**")
                            
                            try:
                                table = [[additional_params['yes_yes'], additional_params['yes_no']], 
                                         [additional_params['no_yes'], additional_params['no_no']]]
                                result = mcnemar(table, exact=True)
                                stat = result.statistic
                                p = result.pvalue
                                st.write(f"**McNemar Test Statistic:** {stat:.4f}")
                                st.write(f"**McNemar Test p-value:** {p:.4f}")
                            except Exception as e:
                                st.error(f"❌ **Error:** {e}")
                                st.info("Please ensure you have entered valid values for the contingency table.")
                        else:
                            st.subheader("🧪 **Unpaired Test: Fisher's Exact Test**")
                            
                            try:
                                table = [[additional_params['group1_yes'], additional_params['group1_no']], 
                                         [additional_params['group2_yes'], additional_params['group2_no']]]
                                odds_ratio, p = fisher_exact(table)
                                st.write(f"**Odds Ratio:** {odds_ratio:.4f}")
                                st.write(f"**Fisher's Exact Test p-value:** {p:.4f}")
                            except Exception as e:
                                st.error(f"❌ **Error:** {e}")
                                st.info("Please ensure you have entered valid values for the contingency table.")
        
                    ## --- More than Two Samples ---
                    elif group_selection == "More than Two Samples":
                        if paired == "Paired":
                            st.subheader("🧪 **Paired Test: Cochran's Q Test**")
  
                            try:
                                import numpy as np
                                data = pd.DataFrame(additional_params['paired_data']).T
                                result = cochrans_q(data)
                                stat = result.statistic
                                p = result.pvalue
                                st.write(f"**Cochran's Q Test Statistic:** {stat:.4f}")
                                st.write(f"**p-value:** {p:.4f}")
                            except Exception as e:
                                st.error(f"❌ **Error:** {e}")
                                st.info("Please ensure you have entered valid values for the contingency table.")
  
                        else:
                            st.subheader("🧪 **Unpaired Test: Chi-Square Test**")

                            try:
                                import numpy as np
                                table = np.array(additional_params['chi_table'])
                                statistic, p_value, dof, expected_freq = chi2_contingency(table)
                                stat = statistic
                                p = p_value
                                expected = expected_freq
                                st.write(f"**Chi-Square Test Statistic:** {stat:.4f}")
                                st.write(f"**p-value:** {p:.4f}")
                                st.write(f"**Degrees of Freedom:** {dof}")
                                st.write("**Expected Frequencies:**")
                                st.write(pd.DataFrame(expected))
                            except Exception as e:
                                st.error(f"❌ **Error:** {e}")
                                st.info("Please ensure you have entered valid values for the contingency table.")

                ## --- Final Result Message ---
                if p < 0.05:
                    st.success("✅ **Statistically Significant Result:** Reject Null Hypothesis")
                else:
                    st.warning("❌ **Not Statistically Significant:** Fail to Reject Null Hypothesis")

            except Exception as e:
                st.error(f"❌ **Error:** {e}")
                st.info("Please ensure you have selected the correct data type and group selection.")

# --- Reset Button ---
if st.button("🔄 Reset App"):
    st.session_state.clear()
    st.query_params.clear()
    st.query_params["tab"] = "📂 Data Input"
    st.rerun()