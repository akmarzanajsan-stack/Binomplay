import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import pandas as pd

st.set_page_config(page_title="BinomPlay", page_icon="🎲", layout="wide")

st.title("🎲 BinomPlay — Binomial Distribution")
st.markdown("Enter **n**, **p**, **k** to calculate probabilities")

# Sidebar
st.sidebar.header("⚙️ Parameters")
n = st.sidebar.number_input("Number of trials (n):", 1, 200, 10)
p = st.sidebar.slider("Success probability (p):", 0.0, 1.0, 0.6, 0.01)
k = st.sidebar.number_input("Number of successes (k):", 0, n, min(6, n))
mode = st.sidebar.radio(
    "Calculation type:",
    ["P(X = k)", "P(X >= k)", "P(X <= k)"],
    index=1
)

dist = binom(n, p)

if mode == "P(X = k)":
    result = dist.pmf(k)
    label = f"P(X = {k})"
elif mode == "P(X >= k)":
    result = dist.sf(k - 1)
    label = f"P(X >= {k})"
else:
    result = dist.cdf(k)
    label = f"P(X <= {k})"

# Result
st.markdown(f"### 📊 {label} = **{result*100:.2f}%**")

col1, col2, col3 = st.columns(3)
col1.metric("E[X] — Mean", f"{n*p:.2f}")
col2.metric("Var(X) — Variance", f"{n*p*(1-p):.2f}")
col3.metric("Sigma — Std Dev", f"{np.sqrt(n*p*(1-p)):.2f}")

# Chart
st.markdown("### 📈 Distribution Chart")
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(0, n + 1)
y = dist.pmf(x) * 100

colors = []
for i in x:
    if mode == "P(X = k)" and i == k:
        colors.append('#764ba2')
    elif mode == "P(X >= k)" and i >= k:
        colors.append('#764ba2')
    elif mode == "P(X <= k)" and i <= k:
        colors.append('#764ba2')
    else:
        colors.append('#c5cae9')

ax.bar(x, y, color=colors, edgecolor='white')
ax.set_xlabel('k — Number of successes')
ax.set_ylabel('Probability (%)')
ax.set_title(f'Binomial Distribution (n={n}, p={p})')
ax.grid(axis='y', alpha=0.3, linestyle='--')
st.pyplot(fig)

# Table
st.markdown("### 📋 Full Table")
table_data = []
for i in range(n + 1):
    table_data.append({
        "k": i,
        "P(X = k)": f"{dist.pmf(i)*100:.2f}%",
        "P(X <= k)": f"{dist.cdf(i)*100:.2f}%",
        "P(X >= k)": f"{dist.sf(i-1)*100:.2f}%"
    })
st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
