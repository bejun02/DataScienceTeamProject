# -*- coding: utf-8 -*-
"""
Statistical Significance Test Visualization
t-test & Chi-square test results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Data preprocessing function
def preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    columns_to_drop = [
        'SALE TYPE', 'STATE OR PROVINCE', 'STATUS',
        'NEXT OPEN HOUSE START TIME', 'NEXT OPEN HOUSE END TIME',
        'SOURCE', 'FAVORITE', 'INTERESTED', 'SOLD DATE', 'MLS#',
        'ZIP OR POSTAL CODE', 'HOA/MONTH', 'ADDRESS', 'LOCATION',
        'LOT SIZE', 'DAYS ON MARKET',
        'URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    df = df.dropna()
    types_to_remove = ['Mobile/Manufactured Home', 'Multi-Family (2-4 Unit)', 'Vacant Land']
    df = df[~df['PROPERTY TYPE'].isin(types_to_remove)]
    df = df[(df['PRICE'] != 7050000) & (df['PRICE'] != 4500000)]
    return df

# Load data
print("Loading data...")
king_df = preprocess_data('King_County_Sold.csv')
pierce_df = preprocess_data('Pierce_County_Sold.csv')
print(f"King County: {len(king_df)} records, Pierce County: {len(pierce_df)} records")

# ============================================================================
# Figure: Statistical Test Results Comprehensive Visualization
# ============================================================================

fig = plt.figure(figsize=(16, 12))
fig.suptitle('Statistical Significance Test: "This Difference is NOT Random"', fontsize=18, fontweight='bold', y=0.98)

# 2x2 layout
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

# ============================================================================
# 1. Top Left: t-test Boxplot + Mean Difference Visualization
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

# Boxplot
bp = ax1.boxplot([king_df['PRICE']/1e6, pierce_df['PRICE']/1e6], 
                 positions=[1, 2], widths=0.6, patch_artist=True)
bp['boxes'][0].set_facecolor('#2E86AB')
bp['boxes'][1].set_facecolor('#A23B72')
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_alpha(0.7)

# Mean markers
king_mean = king_df['PRICE'].mean()/1e6
pierce_mean = pierce_df['PRICE'].mean()/1e6
ax1.scatter([1, 2], [king_mean, pierce_mean], color='red', s=150, zorder=5, 
            marker='D', edgecolor='white', linewidth=2, label='Mean')

# Mean connection line + difference display
ax1.plot([1, 2], [king_mean, pierce_mean], 'r--', linewidth=2, alpha=0.7)
ax1.annotate('', xy=(2.3, pierce_mean), xytext=(2.3, king_mean),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax1.text(2.45, (king_mean + pierce_mean)/2, f'Diff\n$280K\n(+43.9%)', 
         fontsize=10, ha='left', va='center', color='red', fontweight='bold')

ax1.set_xticks([1, 2])
ax1.set_xticklabels(['King County\n(n=347)', 'Pierce County\n(n=320)'], fontsize=11)
ax1.set_ylabel('Price (Million $)', fontsize=12)
ax1.set_title('[Welch\'s t-test]\nMean Price Comparison', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_xlim(0.3, 3)

# ============================================================================
# 2. Top Right: t-test Result Table + Interpretation
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')

# t-test result table
table_data = [
    ['Test Item', 'Value', 'Interpretation'],
    ['t-statistic', '9.30', 'Mean diff = 9.3x std error'],
    ['p-value', '< 0.001 *', 'Random chance < 0.1%'],
    ['Significance (a)', '0.05', 'Threshold'],
    ['Mean Difference', '$280,476', 'King is $280K higher'],
    ['95% CI', '$221K ~ $340K', 'True diff range'],
    ['Conclusion', 'Reject H0 [O]', 'Price diff is REAL']
]

table = ax2.table(cellText=table_data, loc='center', cellLoc='center',
                   colWidths=[0.3, 0.3, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.2)

# Header style
for j in range(3):
    table[(0, j)].set_facecolor('#2E86AB')
    table[(0, j)].set_text_props(color='white', fontweight='bold')
# p-value row highlight
for j in range(3):
    table[(2, j)].set_facecolor('#fff3cd')
# Conclusion row highlight
for j in range(3):
    table[(6, j)].set_facecolor('#d4edda')
    table[(6, j)].set_text_props(fontweight='bold')

ax2.set_title('[t-test Result Interpretation]', fontsize=13, fontweight='bold', pad=20)

# ============================================================================
# 3. Bottom Left: Chi-square Test - Bar Chart
# ============================================================================
ax3 = fig.add_subplot(gs[1, 0])

# Chi-square values
categories = ['CITY\n(King)', 'CITY\n(Pierce)', 'PROPERTY\nTYPE (King)']
chi_values = [113.42, 131.53, 78.10]
colors = ['#2E86AB', '#A23B72', '#F39C12']

bars = ax3.bar(categories, chi_values, color=colors, edgecolor='white', linewidth=2)

# Value display
for bar, val in zip(bars, chi_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
             f'X2 = {val}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Critical value line
ax3.axhline(y=15, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Critical Value')
ax3.text(2.5, 18, 'Above Critical = Significant', fontsize=10, color='red', ha='right')

ax3.set_ylabel('Chi-square Statistic', fontsize=12)
ax3.set_title('[Chi-square Test]\nCategorical Variables vs Price', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, 160)

# ============================================================================
# 4. Bottom Right: Chi-square Result Table + Meaning
# ============================================================================
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

# Chi-square result table
chi_table_data = [
    ['Variable', 'X2 Stat', 'p-value', 'Conclusion'],
    ['CITY (King)', '113.42', '< 0.001', 'City affects price [O]'],
    ['CITY (Pierce)', '131.53', '< 0.001', 'City affects price [O]'],
    ['PROPERTY TYPE', '78.10', '< 0.001', 'Type affects price [O]']
]

chi_table = ax4.table(cellText=chi_table_data, loc='upper center', cellLoc='center',
                       colWidths=[0.28, 0.2, 0.17, 0.35])
chi_table.auto_set_font_size(False)
chi_table.set_fontsize(10)
chi_table.scale(1.2, 2.0)

# Header style
for j in range(4):
    chi_table[(0, j)].set_facecolor('#A23B72')
    chi_table[(0, j)].set_text_props(color='white', fontweight='bold')
# Data row colors
for i in range(1, 4):
    chi_table[(i, 3)].set_facecolor('#d4edda')

ax4.set_title('[Chi-square Test Results]', fontsize=13, fontweight='bold', pad=20)

# Bottom interpretation text
interpretation = """
[Key Point] What Chi-square test tells us:
   - "If city and price were unrelated" -> prices should be evenly distributed
   - But in reality: Bellevue = high price, Tacoma = low price
   - Large Chi-square = Big gap between expected & observed = Related!
   
[Conclusion] Location (City) is a KEY factor determining price"""

ax4.text(0.5, 0.15, interpretation, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6', alpha=0.9))

plt.savefig('visual_statistical_test_results.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("\nSaved: visual_statistical_test_results.png")

# ============================================================================
# Additional: p-value Meaning Explanation Graph
# ============================================================================
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('What p-value Means: "Probability of Random Occurrence"', fontsize=16, fontweight='bold')

# Left: t-distribution and t-statistic position
ax = axes[0]
x = np.linspace(-12, 12, 1000)
y = stats.t.pdf(x, df=600)
ax.plot(x, y, 'b-', linewidth=2, label='t-distribution (under H0)')
ax.fill_between(x, y, where=(x <= -9.3) | (x >= 9.3), color='red', alpha=0.4, label='p-value area')
ax.axvline(x=9.30, color='red', linestyle='--', linewidth=2)
ax.axvline(x=-9.30, color='red', linestyle='--', linewidth=2)
ax.text(9.5, 0.15, 't = 9.30\n(Our Result)', fontsize=11, color='red', fontweight='bold')
ax.set_xlabel('t-statistic', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('t-test: What is the probability of t=9.30?', fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax.set_xlim(-12, 14)

# Red area explanation
ax.annotate('This area = p-value\n= Less than 0.001 (Very small!)', 
            xy=(10, 0.02), xytext=(6, 0.2),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='red'))

# Right: p-value interpretation guide
ax = axes[1]
ax.axis('off')

# p-value interpretation visualization
p_levels = [0.5, 0.1, 0.05, 0.01, 0.001]
colors = ['#d4edda', '#c3e6cb', '#ffc107', '#fd7e14', '#dc3545']
labels = ['50%\nNo Evidence', '10%\nWeak', '5%\nThreshold', '1%\nStrong', '0.1%\nVery Strong']

for i, (p, c, l) in enumerate(zip(p_levels, colors, labels)):
    rect = mpatches.FancyBboxPatch((i*0.18 + 0.05, 0.4), 0.15, 0.35, 
                                    boxstyle="round,pad=0.02", 
                                    facecolor=c, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(i*0.18 + 0.125, 0.57, l, ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(i*0.18 + 0.125, 0.45, f'p = {p}', ha='center', va='center', fontsize=10)

# Arrow to show our result
ax.annotate('Our Result\np < 0.001', xy=(0.86, 0.75), xytext=(0.75, 0.9),
            fontsize=12, ha='center', fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax.text(0.5, 0.2, '-> Smaller p-value = Stronger evidence against "random"\n-> Our result (p < 0.001) is the STRONGEST evidence!', 
        ha='center', va='center', fontsize=12, 
        bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('p-value Interpretation Guide', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('visual_pvalue_interpretation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: visual_pvalue_interpretation.png")
