import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

st.title("Two-Way ANOVA App")

st.write("Upload a CSV file with one numeric dependent variable and two categorical factors.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.write("### Dataset Preview")
    st.dataframe(df)

    columns = df.columns.tolist()

    dependent_var = st.selectbox("Select Dependent Variable (numeric)", columns)
    factor1 = st.selectbox("Select First Factor (categorical)", columns)
    factor2 = st.selectbox("Select Second Factor (categorical)", columns)

    if st.button("Run Two-Way ANOVA"):
        try:
            # Build formula
            formula = f"{dependent_var} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"
            
            # Fit model
            model = ols(formula, data=df).fit()
            
            # ANOVA table
            anova_table = sm.stats.anova_lm(model, typ=2)

            st.write("### ANOVA Results")
            st.dataframe(anova_table)

            st.write("### Interpretation Guide")
            st.write("""
            - **PR(>F)** < 0.05 → Significant effect
            - Check:
              - Factor 1 effect
              - Factor 2 effect
              - Interaction effect
            """)

        except Exception as e:
            st.error(f"Error: {e}")