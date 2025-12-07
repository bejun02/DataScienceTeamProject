# -*- coding: utf-8 -*-
"""
통합 회귀 모델 탄력성 분석
각 변수가 1% 변할 때 가격이 몇 % 변하는지 비교
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 데이터 전처리
# ============================================================================
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

# 데이터 로드 및 통합
king_df = preprocess_data('King_County_Sold.csv')
pierce_df = preprocess_data('Pierce_County_Sold.csv')

combined_df = pd.concat([
    king_df.assign(COUNTY=1),  # King = 1
    pierce_df.assign(COUNTY=0)  # Pierce = 0 (기준)
], ignore_index=True)

# 0 이하 값 제거 (로그 변환 위해)
combined_df = combined_df[
    (combined_df['BEDS'] > 0) & 
    (combined_df['BATHS'] > 0) & 
    (combined_df['SQUARE FEET'] > 0) &
    (combined_df['YEAR BUILT'] > 0)
]

print("="*70)
print("통합 회귀 모델: 탄력성 분석")
print("각 변수 1% 변화 시 가격 변화율 비교")
print("="*70)
print(f"\n통합 데이터: {len(combined_df)}건 (King + Pierce)")

# ============================================================================
# Log-Log 회귀 모델 (탄력성 추정)
# ============================================================================
# ln(PRICE) = β₀ + β₁·ln(SQFT) + β₂·ln(BEDS) + β₃·ln(BATHS) + β₄·ln(YEAR) + β₅·COUNTY
# → β₁, β₂, β₃, β₄ = 각 변수의 탄력성 (1% 증가 시 가격 몇 % 증가)

print("\n" + "="*70)
print("【 Log-Log 통합 회귀 모델 】")
print("="*70)
print("\nln(PRICE) = β₀ + β₁·ln(SQFT) + β₂·ln(BEDS) + β₃·ln(BATHS) + β₄·ln(YEAR) + β₅·COUNTY")
print("→ β = 해당 변수 1% 증가 시 가격 몇 % 변화 (탄력성)")

# 로그 변환
y = np.log(combined_df['PRICE'])
X = pd.DataFrame({
    'ln_SQFT': np.log(combined_df['SQUARE FEET']),
    'ln_BEDS': np.log(combined_df['BEDS']),
    'ln_BATHS': np.log(combined_df['BATHS']),
    'ln_YEAR': np.log(combined_df['YEAR BUILT']),
    'COUNTY_King': combined_df['COUNTY']  # 더미 변수 (로그 변환 X)
})

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()

print("\n" + "-"*70)
print(f"{'변수':<20} {'탄력성(β)':<15} {'해석':<40} {'p-value':<10}")
print("-"*70)

results = []
for var in ['ln_SQFT', 'ln_BEDS', 'ln_BATHS', 'ln_YEAR']:
    coef = model.params[var]
    pval = model.pvalues[var]
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    
    var_name = var.replace('ln_', '')
    if coef >= 0:
        interpretation = f"{var_name} 1% 증가 → 가격 {coef:.3f}% 증가"
    else:
        interpretation = f"{var_name} 1% 증가 → 가격 {abs(coef):.3f}% 감소"
    
    print(f"{var_name:<20} {coef:>+.4f} {sig:<6} {interpretation:<40} {pval:.4f}")
    results.append({'변수': var_name, '탄력성': coef, 'p-value': pval, '유의': pval < 0.05})

# COUNTY 더미 (탄력성이 아닌 % 프리미엄)
county_coef = model.params['COUNTY_King']
county_pval = model.pvalues['COUNTY_King']
county_premium = (np.exp(county_coef) - 1) * 100  # % 변환
print(f"{'COUNTY_King':<20} {county_coef:>+.4f} ***   King County 프리미엄: +{county_premium:.1f}%      {county_pval:.4f}")

print("-"*70)
print(f"\nR² = {model.rsquared:.4f}")
print(f"Adj. R² = {model.rsquared_adj:.4f}")

# ============================================================================
# 탄력성 순위 정리
# ============================================================================
print("\n\n" + "="*70)
print("【 탄력성 순위: 어떤 변수가 가격에 가장 민감한가? 】")
print("="*70)

results_df = pd.DataFrame(results)
results_df['절대값'] = results_df['탄력성'].abs()
results_df = results_df.sort_values('절대값', ascending=False)

print(f"\n{'순위':<5} {'변수':<15} {'탄력성':<12} {'해석':<45} {'유의미':<8}")
print("-"*85)

for rank, (_, row) in enumerate(results_df.iterrows(), 1):
    var = row['변수']
    elast = row['탄력성']
    sig = "O" if row['유의'] else "X"
    
    if elast >= 0:
        interp = f"{var} 1% ↑ → 가격 {elast:.2f}% ↑"
    else:
        interp = f"{var} 1% ↑ → 가격 {abs(elast):.2f}% ↓ (역효과)"
    
    print(f"{rank:<5} {var:<15} {elast:>+.4f}      {interp:<45} {sig:<8}")

# ============================================================================
# 시각화
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Elasticity Analysis: Price Change (%) per 1% Variable Change', 
             fontsize=14, fontweight='bold')

# 1. 탄력성 막대 그래프
ax = axes[0]
vars_list = results_df['변수'].tolist()
elasticities = results_df['탄력성'].tolist()
significant = results_df['유의'].tolist()

colors = ['#2E86AB' if s else '#CCCCCC' for s in significant]
bars = ax.barh(vars_list, elasticities, color=colors, edgecolor='white', height=0.6)

for bar, val, sig in zip(bars, elasticities, significant):
    label = f'{val:+.3f}' + (' ***' if sig else ' (n.s.)')
    ax.text(val + 0.02 if val > 0 else val - 0.02, bar.get_y() + bar.get_height()/2,
            label, ha='left' if val > 0 else 'right', va='center', fontsize=11)

ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Elasticity (1% increase → X% price change)', fontsize=11)
ax.set_title('Variable Elasticity Ranking\n(Combined Model)', fontsize=12, fontweight='bold')
ax.set_xlim(-0.3, 1.0)
ax.grid(axis='x', alpha=0.3)

# 2. 해석 요약 테이블
ax = axes[1]
ax.axis('off')

summary_text = f"""
【 Elasticity Interpretation 】

Variable 1% Increase → Price Change:

  SQFT     +1%  →  Price +{results_df[results_df['변수']=='SQFT']['탄력성'].values[0]:.2f}%
  BATHS    +1%  →  Price +{results_df[results_df['변수']=='BATHS']['탄력성'].values[0]:.2f}%
  BEDS     +1%  →  Price {results_df[results_df['변수']=='BEDS']['탄력성'].values[0]:+.2f}%
  YEAR     +1%  →  Price +{results_df[results_df['변수']=='YEAR']['탄력성'].values[0]:.2f}%

──────────────────────────────────────

【 Key Finding 】

  1st: SQFT (Area) - Most elastic
       → 10% larger → {results_df[results_df['변수']=='SQFT']['탄력성'].values[0]*10:.1f}% higher price
  
  2nd: BATHS - High elasticity
       → +1 bath ≈ +33% → +{results_df[results_df['변수']=='BATHS']['탄력성'].values[0]*33:.1f}% price
  
  3rd: BEDS - Negative elasticity
       → More rooms (same area) = smaller rooms
       → Price slightly decreases

Model R² = {model.rsquared:.3f}
"""

ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='Consolas',
        bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

plt.tight_layout()
plt.savefig('elasticity_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("\nSaved: elasticity_analysis.png")

# ============================================================================
# 최종 요약
# ============================================================================
print("\n\n" + "="*70)
print("【 최종 요약 】")
print("="*70)

sqft_e = results_df[results_df['변수']=='SQFT']['탄력성'].values[0]
baths_e = results_df[results_df['변수']=='BATHS']['탄력성'].values[0]
beds_e = results_df[results_df['변수']=='BEDS']['탄력성'].values[0]
year_e = results_df[results_df['변수']=='YEAR']['탄력성'].values[0]

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  통합 모델 탄력성 (변수 1% 증가 시 가격 변화율)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1위: SQFT (면적)      {sqft_e:+.3f}   → 1% 크면 {sqft_e:.2f}% 비쌈         │
│  2위: BATHS (욕실)     {baths_e:+.3f}   → 1% 증가 시 {baths_e:.2f}% 비쌈      │
│  3위: YEAR (건축연도)  {year_e:+.3f}   → 1% 신축 시 {year_e:.2f}% 비쌈       │
│  4위: BEDS (침실)      {beds_e:+.3f}   → 침실 많으면 오히려 가격 하락     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  해석 예시:                                                          │
│  • 면적 10% 큰 집 → 가격 약 {sqft_e*10:.1f}% 높음                         │
│  • 면적 50% 큰 집 → 가격 약 {sqft_e*50:.1f}% 높음                         │
│  • SQFT가 BEDS보다 {abs(sqft_e/beds_e) if beds_e != 0 else 0:.1f}배 더 가격에 민감                       │
│                                                                     │
│  King County Premium: +{county_premium:.1f}%                                │
└─────────────────────────────────────────────────────────────────────┘
""")
