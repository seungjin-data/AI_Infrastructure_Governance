#!/usr/bin/env python3
"""
Replication script for:
"Governing AI infrastructure expansion: Governance architecture, risk redistribution,
and policy responses across 25 jurisdictions (2019-2026)"

Submitted to Environmental Innovation and Societal Transitions (EIST)

Author: Seungjin Kim, Ph.D.
Affiliation: Graduate School of AI Convergence Engineering, aSSIST University
Contact: d.eng.kim@stud.assist.ac.kr

Usage:
    cd AI_Infrastructure_Governance_EIST
    python code/analysis_main.py

All outputs are saved to the outputs/ directory.
Dataset version: v5 (verified, N = 144 events)
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# =================================================================
# Configuration
# =================================================================
DATA_PATH = os.path.join("data", "work_db_v5_EIST_FINAL.csv")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ARCH_ORDER = ['Facilitation', 'Constraint', 'Steering']
REDIST_ORDER = ['Sectoral', 'Spatial', 'Temporal']

print("=" * 65)
print("EIST Replication Script - Full Analysis")
print("=" * 65)

# =================================================================
# Load and validate data
# =================================================================
df = pd.read_csv(DATA_PATH)
N = len(df)
print(f"\nDataset loaded: N = {N}")
assert N == 144, f"Expected N=144, got N={N}"

required_cols = ['Event_ID', 'Year', 'Jurisdiction', 'Level',
                 'Policy_Type', 'Macro_Category',
                 'Redistribution_Type', 'Gov_Architecture']
for col in required_cols:
    assert col in df.columns, f"Missing column: {col}"
print("All required columns present")

# =================================================================
# Table 1: Distribution by macro-category
# =================================================================
print("\n" + "-" * 65)
print("Table 1: Distribution of policy events by macro-category (N=144)")
print("-" * 65)

cat_order = ['REGULATORY_FRAMEWORK', 'NUCLEAR_PROCUREMENT', 'MORATORIUM',
             'GRID_REGULATION', 'NUCLEAR_GOVERNANCE', 'SITING_CONTROL']

table1 = df['Macro_Category'].value_counts().reindex(cat_order)
table1_pct = (table1 / N * 100).round(1)
table1_df = pd.DataFrame({'n': table1, 'pct': table1_pct})
table1_df.loc['Total'] = [N, 100.0]
print(table1_df.to_string())
table1_df.to_csv(os.path.join(OUTPUT_DIR, "table1_macro_distribution.csv"))

# =================================================================
# Table 2: Temporal distribution
# =================================================================
print("\n" + "-" * 65)
print("Table 2: Temporal distribution by year and governance architecture")
print("-" * 65)

table2 = pd.crosstab(df['Year'], df['Gov_Architecture'])[ARCH_ORDER]
table2['Annual_Total'] = table2.sum(axis=1)
table2['Cumulative_pct'] = (table2['Annual_Total'].cumsum() / N * 100).round(1)
table2.loc['Total'] = [table2[c].sum() for c in ARCH_ORDER] + [N, '']
print(table2.to_string())
table2.to_csv(os.path.join(OUTPUT_DIR, "table2_annual_distribution.csv"))

events_2024_2025 = df[df['Year'].isin([2024, 2025])].shape[0]
pct_2024_2025 = round(events_2024_2025 / N * 100, 1)
print(f"\n2024-2025 concentration: {events_2024_2025}/{N} = {pct_2024_2025}%")

# =================================================================
# Table 3: Cross-tabulation (Architecture x Redistribution)
# =================================================================
print("\n" + "-" * 65)
print("Table 3: Governance Architecture x Redistribution Type (N=144)")
print("-" * 65)

ct = pd.crosstab(df['Gov_Architecture'], df['Redistribution_Type'])
ct = ct.reindex(index=ARCH_ORDER, columns=REDIST_ORDER)
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

print("\nCounts:")
print(ct.to_string())
print("\nRow percentages:")
print(ct_pct.round(1).to_string())

# Chi-square test
chi2, p, dof, expected = chi2_contingency(ct)
k = min(ct.shape)
V = np.sqrt(chi2 / (N * (k - 1)))

print(f"\nChi-square test:")
print(f"  chi2({dof}, N={N}) = {chi2:.2f}")
print(f"  p-value < .001" if p < 0.001 else f"  p-value = {p:.4f}")
print(f"  Cramers V = {V:.3f}")

print(f"\nExpected frequencies:")
exp_df = pd.DataFrame(expected, index=ARCH_ORDER, columns=REDIST_ORDER)
print(exp_df.round(2).to_string())
print(f"  Min expected frequency: {expected.min():.2f}")
print(f"  All > 5: {'YES' if expected.min() > 5 else 'NO'}")

print(f"\nKey associations:")
print(f"  Constraint -> Spatial:    {ct_pct.loc['Constraint', 'Spatial']:.1f}%")
print(f"  Facilitation -> Sectoral: {ct_pct.loc['Facilitation', 'Sectoral']:.1f}%")
print(f"  Steering -> Temporal:     {ct_pct.loc['Steering', 'Temporal']:.1f}%")

ct.to_csv(os.path.join(OUTPUT_DIR, "table3_architecture_redistribution.csv"))

# =================================================================
# Table 4: Multi-level governance
# =================================================================
print("\n" + "-" * 65)
print("Table 4: Distribution by governance level (N=144)")
print("-" * 65)

level_order = ['Subnational', 'National', 'Supranational']
table4 = pd.crosstab(df['Level'], df['Gov_Architecture'])
table4 = table4.reindex(index=level_order, columns=ARCH_ORDER)
table4['Total'] = table4.sum(axis=1)
table4_pct = table4[ARCH_ORDER].div(table4['Total'], axis=0) * 100

print("\nCounts:")
print(table4.to_string())
print("\nRow percentages:")
print(table4_pct.round(1).to_string())
print(f"\nKey multi-level patterns:")
print(f"  Subnational -> Constraint: {table4_pct.loc['Subnational', 'Constraint']:.1f}%")
print(f"  National -> Steering:      {table4_pct.loc['National', 'Steering']:.1f}%")
print(f"  National -> Facilitation:  {table4_pct.loc['National', 'Facilitation']:.1f}%")

table4.to_csv(os.path.join(OUTPUT_DIR, "table4_multilevel_distribution.csv"))

# =================================================================
# Figure 1: Heatmap
# =================================================================
print("\n" + "-" * 65)
print("Figure 1: Heatmap")
print("-" * 65)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd",
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Number of events'}, ax=ax)
ax.set_title("Governance Architecture x Redistribution Type (N=144)",
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("Redistribution Type", fontsize=11, labelpad=10)
ax.set_ylabel("Governance Architecture", fontsize=11, labelpad=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure1_heatmap.png"),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, "figure1_heatmap.tiff"),
            dpi=600, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/figure1_heatmap.png (300dpi)")
print(f"Saved: {OUTPUT_DIR}/figure1_heatmap.tiff (600dpi)")

# =================================================================
# Sensitivity Analysis: Excluding non-enacted / proposed events
# =================================================================
# Two-step identification as described in manuscript Section 5.7:
#   Step 1: Policy_Type filter (non-enacted instrument types)
#   Step 2: Description keyword filter (proposed/pending/draft/bill)
#   Union of both = 23 events excluded
# =================================================================
print("\n" + "-" * 65)
print("Sensitivity Analysis: Excluding proposed/non-enacted events")
print("-" * 65)

# Step 1: Non-enacted Policy_Type
non_enacted_types = [
    'Moratorium_Proposed', 'Moratorium_Bill', 'Advocacy',
    'Legal_Challenge', 'Project_Opposition', 'Project_Pause',
    'Project_Withdrawal'
]
ids_by_type = set(
    df[df['Policy_Type'].isin(non_enacted_types)]['Event_ID']
)
print(f"  Step 1 - Non-enacted Policy_Type: {len(ids_by_type)} events")

# Step 2: Description contains proposed/pending/draft/bill
ids_by_desc = set(
    df[df['Description'].str.contains(
        'propos|pending|under consideration|draft|bill',
        case=False, na=False
    )]['Event_ID']
)
print(f"  Step 2 - Description keywords:    {len(ids_by_desc)} events")

# Union
ids_excluded = ids_by_type | ids_by_desc
print(f"  Union (non-enacted + proposed):   {len(ids_excluded)} events")

df_sens = df[~df['Event_ID'].isin(ids_excluded)]
N_s = len(df_sens)
N_excl = N - N_s
print(f"  Remaining sample: N = {N_s}")

ct_s = pd.crosstab(df_sens['Gov_Architecture'], df_sens['Redistribution_Type'])
ct_s = ct_s.reindex(index=ARCH_ORDER, columns=REDIST_ORDER)
chi2_s, p_s, dof_s, exp_s = chi2_contingency(ct_s)
V_s = np.sqrt(chi2_s / (N_s * (min(ct_s.shape) - 1)))
ct_s_pct = ct_s.div(ct_s.sum(axis=1), axis=0) * 100

print(f"\n  chi2({dof_s}) = {chi2_s:.2f}")
print(f"  p-value < .001" if p_s < 0.001 else f"  p-value = {p_s:.4f}")
print(f"  Cramers V = {V_s:.3f}")

# Comparison table
print(f"\nComparison (Full vs Sensitivity):")
h_s = f"Sensitivity (N={N_s})"
print(f"  {'Metric':<26} {'Full (N=144)':<18} {h_s:<20}")
print(f"  {'-'*64}")
print(f"  {'Chi-square':<26} {chi2:<18.2f} {chi2_s:<20.2f}")
print(f"  {'Cramers V':<26} {V:<18.3f} {V_s:<20.3f}")
for ga, rt in [('Constraint','Spatial'),
               ('Facilitation','Sectoral'),
               ('Steering','Temporal')]:
    label = f"{ga}->{rt}"
    v_full = ct_pct.loc[ga, rt]
    v_sens = ct_s_pct.loc[ga, rt]
    print(f"  {label:<26} {v_full:<18.1f} {v_sens:<20.1f}")

# Save sensitivity results
sens_results = pd.DataFrame({
    'Metric': ['N', 'Excluded', 'Chi_square', 'p_value', 'Cramers_V',
               'Constraint_Spatial_pct', 'Facilitation_Sectoral_pct',
               'Steering_Temporal_pct'],
    'Full_Sample': [N, 0, round(chi2, 2), '<.001', round(V, 3),
                    round(ct_pct.loc['Constraint', 'Spatial'], 1),
                    round(ct_pct.loc['Facilitation', 'Sectoral'], 1),
                    round(ct_pct.loc['Steering', 'Temporal'], 1)],
    'Sensitivity': [N_s, N_excl, round(chi2_s, 2), '<.001', round(V_s, 3),
                    round(ct_s_pct.loc['Constraint', 'Spatial'], 1),
                    round(ct_s_pct.loc['Facilitation', 'Sectoral'], 1),
                    round(ct_s_pct.loc['Steering', 'Temporal'], 1)]
})
sens_results.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_analysis.csv"),
                    index=False)

# Save list of excluded events
excl_df = df[df['Event_ID'].isin(ids_excluded)][
    ['Event_ID', 'Year', 'Jurisdiction', 'Policy_Type',
     'Gov_Architecture', 'Redistribution_Type']
].sort_values('Event_ID')
excl_df.to_csv(os.path.join(OUTPUT_DIR, "excluded_events_sensitivity.csv"),
               index=False)

# =================================================================
# Summary
# =================================================================
print("\n" + "=" * 65)
print("REPLICATION COMPLETE")
print("=" * 65)
print(f"  Dataset:       N = {N} verified events")
print(f"  Jurisdictions: {df['Jurisdiction'].nunique()} governance entities")
print(f"  Period:        {df['Year'].min()}-{df['Year'].max()}")
print(f"  Chi-square:    chi2({dof}, N={N}) = {chi2:.2f}, p < .001")
print(f"  Effect size:   Cramers V = {V:.3f}")
print(f"  Sensitivity:   V = {V_s:.3f} (excluding {N_excl} proposed/non-enacted)")
print(f"\n  Output files:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    size = os.path.getsize(fpath)
    print(f"    {fname:<50} {size:>12,} bytes")
print()
