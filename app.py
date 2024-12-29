# app.py

import streamlit as st
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import ast

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

# --- Main Tabs ---
st.title("ADS 511: Statistical Inference Methods Web Application")


# Tab Layout
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
            st.experimental_rerun()

# --- Tab 2: Data Type Selection ---
with tab2:
    if not st.session_state['step_completed']['Data Input']:
        st.warning("⚠️ Please complete 'Data Input' first.")
    else:
        st.header("2️⃣ Data Type Selection")
        st.session_state['data_type'] = st.radio("Select Data Type:", ['Continuous', 'Discrete'])
        
        if st.session_state['data_type']:
            if st.button("Next: Assumption Check"):
                st.session_state['step_completed']['Data Type'] = True
                st.experimental_rerun()

# --- Tab 3: Assumption Check ---
with tab3:
    if not st.session_state['step_completed']['Data Type']:
        st.warning("⚠️ Please complete 'Data Type Selection' first.")
    else:
        st.header("3️⃣ Assumption Check")
        
        st.subheader("📊 Continuous Data Assumption Checks")
        
        # Normality Test (Shapiro-Wilk)
        with st.expander("🔍 Normality Test (Shapiro-Wilk)"):
            shapiro_p = stats.shapiro(st.session_state['data'].iloc[:, 0])[1]
            st.write(f"**Shapiro-Wilk p-value:** {shapiro_p:.4f}")
            normal = shapiro_p > 0.05
            st.success("✅ Data is normally distributed." if normal else "❌ Data is not normally distributed.")
        
        # Homogeneity of Variances (Levene Test)
        with st.expander("🔍 Homogeneity of Variances (Levene Test)"):
            if st.session_state['data'].shape[1] > 1:
                levene_p = stats.levene(*[st.session_state['data'][col] for col in st.session_state['data'].columns])[1]
                st.write(f"**Levene p-value:** {levene_p:.4f}")
                homogeneous = levene_p > 0.05
                st.success("✅ Variances are homogeneous." if homogeneous else "❌ Variances are not homogeneous.")
            else:
                st.info("ℹ️ At least two groups are needed to test homogeneity of variances.")
                homogeneous = True  # Defaulting to True for single-group data
        
        # Independence and Identically Distributed Samples Check
        with st.expander("🔗 Independence and Identical Distribution of Samples"):
            st.write("""
            - Samples should be collected independently.
            - Each observation should not influence another observation.
            - Samples should follow the same distribution.
            """)
            independence_check = st.checkbox("✅ Check if samples are independent and identically distributed")
            st.success("✅ Samples are independent and identically distributed." if independence_check else "❌ Samples might not be independent or identically distributed.")
        
        # Outlier Detection (Using Z-Score Method)
        with st.expander("⚠️ Absence of Significant Outliers"):
            st.write("""
            - Outliers can significantly impact statistical test results.
            - Outliers are detected using the Z-score method.
            """)
            z_scores = (st.session_state['data'] - st.session_state['data'].mean()) / st.session_state['data'].std()
            outliers = (z_scores.abs() > 3).sum().sum()
            st.write(f"**Number of Outliers Detected:** {outliers}")
            st.success("✅ No significant outliers detected." if outliers == 0 else f"⚠️ {outliers} significant outlier(s) detected in the dataset.")
        
        # Final Decision on Parametric/Non-Parametric Tests
        parametric = normal and homogeneous and independence_check and outliers == 0
        st.markdown("---")
        if parametric:
            st.success("✅ **All assumptions hold. Proceed with Parametric Tests.**")
        else:
            st.warning("❌ **Assumptions violated. Proceed with Non-Parametric Tests.**")
        
        # Proceed to Next Step
        if st.button("Next: Group Selection"):
            st.session_state['step_completed']['Assumption Check'] = True
            st.experimental_rerun()


# --- Tab 4: Group Selection ---
with tab4:
    if not st.session_state['step_completed']['Assumption Check']:
        st.warning("⚠️ Please complete 'Assumption Check' first.")
    else:
        st.header("4️⃣ Group Selection")
        st.session_state['group_selection'] = st.radio("Select Group Type:", ["One Sample", "Two Samples", "More than Two Samples"])
        
        if st.button("Next: Run Test"):
            st.session_state['step_completed']['Group Selection'] = True
            st.experimental_rerun()

# --- Tab 5: Run Test ---
with tab5:
    if not st.session_state['step_completed']['Group Selection']:
        st.warning("⚠️ Please complete 'Group Selection' first.")
    else:
        st.header("5️⃣ Run Statistical Test")
        
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

        # Additional Parameters Based on Data Type and Group Selection
        additional_params = {}
        data_type = st.session_state['data_type']
        group_selection = st.session_state['group_selection']

        st.subheader("📊 Test Parameters")
        
        ## Continuous Data Test Parameters
        if data_type == "Continuous":
            st.write("🧠 **Continuous Data Testing Parameters:**")
            if group_selection == "One Sample":
                additional_params['population_mean'] = st.number_input(
                    "Enter the Population Mean (μ₀) for comparison:",
                    min_value=-1000.0,
                    max_value=10000.0,
                    value=0.0,
                    step=0.1
                )
            elif group_selection == "Two Samples":
                paired = st.radio("Are the groups Paired or Unpaired?", ["Paired", "Unpaired"])
            elif group_selection == "More than Two Samples":
                paired = st.radio("Are the groups Paired or Unpaired?", ["Paired", "Unpaired"])

        ## Discrete Data Test Parameters
        if data_type == "Discrete":
            st.write("🧠 **Discrete Data Testing Parameters:**")
            if group_selection == "One Sample":
                additional_params['success'] = st.number_input("Enter number of successes:", min_value=0, value=1)
                additional_params['trials'] = st.number_input("Enter number of trials:", min_value=1, value=10)
            elif group_selection == "Two Samples":
                table = [
                    [st.number_input('Cell 1,1'), st.number_input('Cell 1,2')],
                    [st.number_input('Cell 2,1'), st.number_input('Cell 2,2')]
                ]
                additional_params['table'] = table
            elif group_selection == "More than Two Samples":
                additional_params['data'] = st.text_area("Enter the data for each group (e.g., [1,2,3], [4,5,6])")

        # Run Test Button
        if st.button("Run Test"):
            try:
                data = st.session_state['data']
                
                ## --- Continuous Data Workflow ---
                if data_type == 'Continuous':
                    if st.session_state['parametric']:
                        # Parametric Tests
                        if group_selection == "One Sample":
                            st.write("🧪 **One Sample Test: One Sample t-test**")
                            population_mean = additional_params['population_mean']
                            stat, p = stats.ttest_1samp(data.iloc[:, 0], population_mean, alternative=alternative)
                            st.success("✅ Test Completed Successfully!")
                            st.write(f"**Population Mean (μ₀):** {population_mean:.4f}")
                            st.write(f"**One Sample t-test Statistic:** {stat:.4f},**p-value:** {p:.4f}")

                        elif group_selection == "Two Samples":
                            if paired == "Paired":
                                st.write("🧪 **Two Sample Paired Test: Paired t-test**")
                                stat, p = stats.ttest_rel(data.iloc[:, 0], data.iloc[:, 1], alternative=alternative)
                                st.write(f"**Paired t-test Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                            else:
                                st.write("🧪 **Two Sample Unpaired Test: Independent t-test**")
                                stat, p = stats.ttest_ind(data.iloc[:, 0], data.iloc[:, 1], alternative=alternative)
                                st.write(f"**Independent t-test Statistic:** {stat:.4f}, **p-value:** {p:.4f}")

                        elif group_selection == "More than Two Samples":
                            if paired == "Paired":
                                # Repeated Measures ANOVA
                                st.write("🧪 **Repeated Measures ANOVA: One-Way ANOVA**")
                                stat, p = stats.f_oneway(*[data[col] for col in data.columns])
                                st.write(f"**One-Way ANOVA F-Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                                if p < 0.05:
                                    import scikit_posthocs as sp
                                    st.success("✅ Significant Differences Found! Performing Pairwise T-tests...")
                                    posthoc_df = sp.posthoc_ttest(*[data[col] for col in data.columns], equal_var=True, p_adjust='bonferroni')
                                    group_names = list(data.columns)
                                    posthoc_df.index = group_names
                                    posthoc_df.columns = group_names
                                    st.write("🔍 **Pairwise T-Test Results (Bonferroni Corrected):**")
                                    st.write(posthoc_df)

                            else:
                                # One-Way ANOVA
                                st.write("🧪 **One-Way ANOVA: One-Way ANOVA**")
                                stat, p = stats.f_oneway(*[data[col] for col in data.columns])
                                st.write(f"**One-Way ANOVA F-Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                                if p < 0.05:
                                    import scikit_posthocs as sp
                                    st.success("✅ Significant Differences Found! Performing Pairwise T-tests...")
                                    posthoc_df = sp.posthoc_ttest(*[data[col] for col in data.columns], equal_var=True, p_adjust='bonferroni')
                                    group_names = list(data.columns)
                                    posthoc_df.index = group_names
                                    posthoc_df.columns = group_names
                                    st.write("🔍 **Pairwise T-Test Results (Bonferroni Corrected):**")
                                    st.write(posthoc_df)

                    else:
                        # Non-Parametric Tests
                        if group_selection == "One Sample":
                            st.write("🧪 **Non-Parametric One Sample Test: Wilcoxon Signed-Rank Test**")
                            population_mean = additional_params['population_mean']
                            diff = data.iloc[:, 0] - population_mean
                            stat, p = stats.wilcoxon(diff, alternative=alternative)
                            st.write(f"**Wilcoxon Signed-Rank Test Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                        elif group_selection == "Two Samples":
                            paired = st.radio("Are the groups Paired or Unpaired?", ["Paired", "Unpaired"])
                            if paired == "Paired":
                                st.write("🧪 **Non-Parametric Two-Sample Paired Test: Wilcoxon Signed-Rank Test**")
                                stat, p = stats.wilcoxon(data.iloc[:, 0], data.iloc[:, 1], alternative=alternative)
                                st.write(f"**Wilcoxon Signed-Rank Test Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                            else:
                                st.write("🧪 **Non-Parametric Two-Sample Unpaired Test: Mann-Whitney U Test**")
                                stat, p = stats.mannwhitneyu(data.iloc[:, 0], data.iloc[:, 1], alternative=alternative)
                                st.write(f"**Mann-Whitney U Test Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                        elif group_selection == "More than Two Samples":
                            paired = st.radio("Are the groups Paired or Unpaired?", ["Paired", "Unpaired"])
                            if paired == "Paired":
                                st.write("🧪 **Non-Parametric Repeated Measures Test: Friedman Test**")
                                stat, p = stats.friedmanchisquare(*[data[col] for col in data.columns])
                                st.write(f"**Friedman Test Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                                if p < 0.05:
                                    import scikit_posthocs as sp
                                    st.success("✅ Significant Differences Found! Performing Pairwise T-tests...")
                                    posthoc_df = sp.posthoc_ttest(*[data[col] for col in data.columns], equal_var=True, p_adjust='bonferroni')
                                    group_names = list(data.columns)
                                    posthoc_df.index = group_names
                                    posthoc_df.columns = group_names
                                    st.write("🔍 **Pairwise T-Test Results (Bonferroni Corrected):**")
                                    st.write(posthoc_df)
                            else:
                                st.write("🧪 **Non-Parametric One-Way Test: Kruskal-Wallis H Test**")
                                stat, p = stats.kruskal(*[data[col] for col in data.columns])
                                st.write(f"**Kruskal-Wallis H Test Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                                if p < 0.05:
                                    import scikit_posthocs as sp
                                    st.success("✅ Significant Differences Found! Performing Pairwise T-tests...")
                                    posthoc_df = sp.posthoc_ttest(*[data[col] for col in data.columns], equal_var=True, p_adjust='bonferroni')
                                    group_names = list(data.columns)
                                    posthoc_df.index = group_names
                                    posthoc_df.columns = group_names
                                    st.write("🔍 **Pairwise T-Test Results (Bonferroni Corrected):**")
                                    st.write(posthoc_df)   

                ## --- Discrete Data Workflow ---
                if data_type == 'Discrete':
                    if group_selection == "One Sample":
                        success = additional_params['success']
                        trials = additional_params['trials']
                        p = stats.binom_test(success, trials, alternative=alternative)
                        st.write(f"**Binomial Test p-value:** {p:.4f}")
                    elif group_selection == "Two Samples":
                        table = additional_params['table']
                        stat, p = stats.fisher_exact(table)
                        st.write(f"**Fisher's Exact Test p-value:** {p:.4f}")
                    elif group_selection == "More than Two Samples":
                        stat, p = stats.chisquare(*[data[col] for col in data.columns])
                        st.write(f"**Chi-Square Test p-value:** {p:.4f}")

                ## --- Final Result Message ---
                if p < 0.05:
                    st.success("✅ **Statistically Significant Result:** Reject Null Hypothesis")
                else:
                    st.warning("❌ **Not Statistically Significant:** Fail to Reject Null Hypothesis")

            except Exception as e:
                st.error(f"❌ **Error:** {e}")
                st.info("Please ensure you have selected the correct data type and group selection.")





























# --- Tab 5: Run Test ---
with tab5:
    if 'data' in locals():
        st.header("5️⃣ Run Test")
        alternative = st.selectbox(
                    "Choose Hypothesis Type:",
                    ("two-sided", "greater", "less"),
                    index=0,
                    help="Choose 'two-sided' for general differences, 'greater' if mean/median is expected to be higher, 'less' if lower."
                    )
        

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
                                                                      max_value=10000.0,
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
                            stat, p = stats.ttest_1samp(data.iloc[:, 0], population_mean, alternative=alternative)
                            if alternative == "greater" and stat < 0:
                                p /= 2
                            elif alternative == "less" and stat > 0:
                                p /= 2
                            # Display the results of the one-sample t-test
                            st.toast("✅ Test Completed Successfully!", icon="🎯")
                            st.write(f"**Population Mean (μ₀):** {population_mean:.4f}")
                            st.write(f"**Test Statistic:** {stat:.4f}")
                            st.write(f"**One-Sample t-test p-value:** {p:.4f}")

                        elif group_selection == "Two Samples":
                            if paired == "Paired":
                                stat, p = stats.ttest_rel(data.iloc[:, 0], data.iloc[:, 1], alternative=alternative)
                                if alternative == "greater" and stat < 0:
                                    p /= 2
                                elif alternative == "less" and stat > 0:
                                    p /= 2
                                st.write(f"**Paired t-test Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                            else:
                                stat, p = stats.ttest_ind(data.iloc[:, 0], data.iloc[:, 1],alternative=alternative)
                                if alternative == "greater" and stat < 0:
                                    p /= 2  
                                elif alternative == "less" and stat > 0:
                                    p /= 2
                                st.write(f"**Independent t-test Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                        elif group_selection == "More than Two Samples":
                            if paired == "Paired":
                                # Repeated Measures ANOVA
                                stat, p = stats.f_oneway(*[data[col] for col in data.columns])
                                st.write(f"**Repeated Measures ANOVA results:**")
                                st.write(f"**F-Statistic:** {stat:.4f}, **p-value:** {p:.4f}")

                                # Post-hoc Pairwise T-Tests
                                if p < 0.05:
                                    import scikit_posthocs as sp
                                    st.success("✅ Significant Differences Found! Performing Pairwise T-tests...")
            
                                    # Pairwise T-Tests with Bonferroni Correction
                                    posthoc_df = sp.posthoc_ttest(*[data[col] for col in data.columns], 
                                                                  equal_var=True,
                                                                  p_adjust='bonferroni')
                                    
                                    # Add group names for clarity
                                    group_names = list(data.columns)
                                    posthoc_df.index = group_names
                                    posthoc_df.columns = group_names

                                    # Display the pairwise t-test results
                                    st.write("🔍 **Pairwise T-Test Results (Bonferroni Corrected):**")
                                    st.write(posthoc_df.style.map(
                                        lambda x: 'background-color: lightblue' if x < 0.05 else 'background-color: white '
                                    ))
                                else:
                                    st.warning("❌ No significant difference detected across groups.")
                                    
                            else: 
                                # One-Way ANOVA
                                stat, p = stats.f_oneway(*[data[col] for col in data.columns])
                                st.write(f"**One-Way ANOVA results:**")
                                st.write(f"**F-Statistic:** {stat:.4f}, **p-value:** {p:.4f}")

                                # Post-hoc Pairwise T-Tests
                                if p < 0.05:
                                    import scikit_posthocs as sp
                                    st.success("✅ Significant Differences Found! Performing Pairwise T-tests...")
            
                                    # Pairwise T-Tests with Bonferroni Correction
                                    posthoc_df = sp.posthoc_ttest(*[data[col] for col in data.columns], 
                                                                  equal_var=True,
                                                                  p_adjust='bonferroni')
                                    
                                    # Add group names for clarity
                                    group_names = list(data.columns)
                                    posthoc_df.index = group_names
                                    posthoc_df.columns = group_names

                                    # Display the pairwise t-test results
                                    st.write("🔍 **Pairwise T-Test Results (Bonferroni Corrected):**")
                                    st.write(posthoc_df.style.map(
                                        lambda x: 'background-color: lightblue' if x < 0.05 else 'background-color: white '
                                    ))
                                else:
                                    st.warning("❌ No significant difference detected across groups.")                                
                    
                    else:  # Non-Parametric Tests
                        if group_selection == "One Sample":
                            population_mean = additional_params['population_mean']
                            diff = data.iloc[:, 0] - population_mean
                            stat, p = stats.wilcoxon(diff,alternative=alternative)
                            st.write(f"**One-Sample Wilcoxon Signed-Rank Test Statistic:** {stat:.4f}, **p-value**: {p:.4f}")
                            
                        elif group_selection == "Two Samples":
                            if paired == "Paired":
                                stat, p = stats.wilcoxon(data.iloc[:, 0], data.iloc[:, 1], alternative=alternative)
                                st.write(f"**Wilcoxon Signed-Rank Test Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
                            else:
                                stat, p = stats.mannwhitneyu(data.iloc[:, 0], data.iloc[:, 1], alternative=alternative)
                                st.write(f"**Mann-Whitney U Test Statistic:** {stat:.4f}, **p-value:** {p:.4f}")
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