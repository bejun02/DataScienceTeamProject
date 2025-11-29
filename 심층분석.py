import pandas as pd
import numpy as np
from scipy.stats import shapiro, ttest_ind, chi2_contingency, spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from 데이터전처리_1 import preprocess_data

# ============================================================================
# 데이터 로드
# ============================================================================
print("="*120)
print("【 King County vs Pierce County 심층 분석 】")
print("="*120)

king = preprocess_data('King_County_Sold.csv', 'King County')
pierce = preprocess_data('Pierce_County_Sold.csv', 'Pierce County')

# ============================================================================
# 1. 기본 통계 비교
# ============================================================================
print("\n\n【 1. 기본 통계 비교 】")
print("="*120)

print(f"\n{'지표':<25} {'King County':<20} {'Pierce County':<20} {'차이':<15}")
print("-"*80)
print(f"{'데이터 수':<25} {len(king):<20} {len(pierce):<20} {len(king)-len(pierce):<15}")
print(f"{'평균 가격':<25} ${king['PRICE'].mean():>15,.0f} ${pierce['PRICE'].mean():>15,.0f} ${king['PRICE'].mean()-pierce['PRICE'].mean():>+10,.0f}")
print(f"{'중앙값':<25} ${king['PRICE'].median():>15,.0f} ${pierce['PRICE'].median():>15,.0f} ${king['PRICE'].median()-pierce['PRICE'].median():>+10,.0f}")
print(f"{'표준편차':<25} ${king['PRICE'].std():>15,.0f} ${pierce['PRICE'].std():>15,.0f} ${king['PRICE'].std()-pierce['PRICE'].std():>+10,.0f}")
print(f"{'최소값':<25} ${king['PRICE'].min():>15,.0f} ${pierce['PRICE'].min():>15,.0f} ${king['PRICE'].min()-pierce['PRICE'].min():>+10,.0f}")
print(f"{'최대값':<25} ${king['PRICE'].max():>15,.0f} ${pierce['PRICE'].max():>15,.0f} ${king['PRICE'].max()-pierce['PRICE'].max():>+10,.0f}")

price_premium = ((king['PRICE'].mean() - pierce['PRICE'].mean()) / pierce['PRICE'].mean()) * 100
print(f"\n▶ King County 가격 프리미엄: {price_premium:+.1f}%")

# ============================================================================
# 2. 정규분포 검정 (Shapiro-Wilk)
# ============================================================================
print("\n\n【 2. 정규분포 검정 (Shapiro-Wilk Test) 】")
print("="*120)

# PRICE 정규성 검정
king_shapiro = shapiro(king['PRICE'].sample(min(50, len(king))))
pierce_shapiro = shapiro(pierce['PRICE'].sample(min(50, len(pierce))))

print(f"\n[PRICE 정규분포 검정]")
print(f"  King County:   통계량={king_shapiro[0]:.4f}, p-value={king_shapiro[1]:.6f}")
print(f"  Pierce County: 통계량={pierce_shapiro[0]:.4f}, p-value={pierce_shapiro[1]:.6f}")

if king_shapiro[1] < 0.05:
    print(f"  -> King County PRICE: 정규분포 아님 (p < 0.05)")
else:
    print(f"  -> King County PRICE: 정규분포 따름 (p >= 0.05)")
    
if pierce_shapiro[1] < 0.05:
    print(f"  -> Pierce County PRICE: 정규분포 아님 (p < 0.05)")
else:
    print(f"  -> Pierce County PRICE: 정규분포 따름 (p >= 0.05)")

# 왜도/첨도
print(f"\n[왜도 (Skewness) - 0에 가까울수록 대칭]")
print(f"  King County:   {king['PRICE'].skew():.4f}")
print(f"  Pierce County: {pierce['PRICE'].skew():.4f}")

print(f"\n[첨도 (Kurtosis) - 0에 가까울수록 정규분포]")
print(f"  King County:   {king['PRICE'].kurtosis():.4f}")
print(f"  Pierce County: {pierce['PRICE'].kurtosis():.4f}")

# ============================================================================
# 3. 두 카운티 평균 가격 비교 (t-검정)
# ============================================================================
print("\n\n【 3. 두 카운티 평균 가격 비교 (독립 t-검정) 】")
print("="*120)

t_stat, p_value = ttest_ind(king['PRICE'], pierce['PRICE'])
print(f"\n  King County 평균:   ${king['PRICE'].mean():,.0f}")
print(f"  Pierce County 평균: ${pierce['PRICE'].mean():,.0f}")
print(f"  차이:               ${king['PRICE'].mean() - pierce['PRICE'].mean():,.0f}")
print(f"\n  t-통계량: {t_stat:.4f}")
print(f"  p-value:  {p_value:.10f}")

if p_value < 0.05:
    print(f"\n  결론: [O] 두 카운티의 평균 가격이 **통계적으로 유의미하게 다름** (p < 0.05)")
else:
    print(f"\n  결론: [X] 두 카운티의 평균 가격 차이가 **통계적으로 유의미하지 않음** (p >= 0.05)")

# ============================================================================
# 4. 상관계수 분석 (Pearson & Spearman)
# ============================================================================
print("\n\n【 4. 상관계수 분석 (Pearson & Spearman) 】")
print("="*120)

numeric_cols = ['BEDS', 'BATHS', 'SQUARE FEET', 'YEAR BUILT', '$/SQUARE FEET', 'LATITUDE', 'LONGITUDE']

print(f"\n[King County - PRICE와의 상관계수]")
print(f"  {'변수':<20} {'Pearson':<15} {'Spearman':<15} {'영향도':<10}")
print(f"  {'-'*60}")
for col in numeric_cols:
    pearson_corr = king['PRICE'].corr(king[col])
    spearman_corr, _ = spearmanr(king['PRICE'], king[col])
    if abs(pearson_corr) > 0.6:
        strength = "매우 강함"
    elif abs(pearson_corr) > 0.4:
        strength = "강함"
    elif abs(pearson_corr) > 0.2:
        strength = "중간"
    else:
        strength = "약함"
    print(f"  {col:<20} {pearson_corr:<15.4f} {spearman_corr:<15.4f} {strength:<10}")

print(f"\n[Pierce County - PRICE와의 상관계수]")
print(f"  {'변수':<20} {'Pearson':<15} {'Spearman':<15} {'영향도':<10}")
print(f"  {'-'*60}")
for col in numeric_cols:
    pearson_corr = pierce['PRICE'].corr(pierce[col])
    spearman_corr, _ = spearmanr(pierce['PRICE'], pierce[col])
    if abs(pearson_corr) > 0.6:
        strength = "매우 강함"
    elif abs(pearson_corr) > 0.4:
        strength = "강함"
    elif abs(pearson_corr) > 0.2:
        strength = "중간"
    else:
        strength = "약함"
    print(f"  {col:<20} {pearson_corr:<15.4f} {spearman_corr:<15.4f} {strength:<10}")

# ============================================================================
# 5. 경제 기능 차이 분석 (일자리 중심지 vs 위성 주거지)
# ============================================================================
print("\n\n【 5. 경제 기능 차이 분석: 일자리 중심지 vs 위성 주거지 】")
print("="*120)

print(f"\n[주택 규모 비교 - 도시 vs 교외 생활 방식]")
print(f"  {'지표':<25} {'King (도시)':<20} {'Pierce (교외)':<20} {'해석':<30}")
print(f"  {'-'*95}")

king_sqft = king['SQUARE FEET'].mean()
pierce_sqft = pierce['SQUARE FEET'].mean()
sqft_diff = ((pierce_sqft - king_sqft) / king_sqft) * 100
print(f"  {'건물 면적 (sqft)':<25} {king_sqft:>15,.0f} {pierce_sqft:>15,.0f} Pierce가 {sqft_diff:+.1f}% 넓음")

king_beds = king['BEDS'].mean()
pierce_beds = pierce['BEDS'].mean()
beds_diff = ((pierce_beds - king_beds) / king_beds) * 100
print(f"  {'침실 수':<25} {king_beds:>15.2f} {pierce_beds:>15.2f} Pierce가 {beds_diff:+.1f}% 많음")

king_baths = king['BATHS'].mean()
pierce_baths = pierce['BATHS'].mean()
baths_diff = ((pierce_baths - king_baths) / king_baths) * 100
print(f"  {'욕실 수':<25} {king_baths:>15.2f} {pierce_baths:>15.2f} Pierce가 {baths_diff:+.1f}% 많음")

king_year = king['YEAR BUILT'].mean()
pierce_year = pierce['YEAR BUILT'].mean()
print(f"  {'평균 건축년도':<25} {king_year:>15.0f} {pierce_year:>15.0f} Pierce가 {pierce_year-king_year:.0f}년 신축")

king_ppsf = king['$/SQUARE FEET'].mean()
pierce_ppsf = pierce['$/SQUARE FEET'].mean()
ppsf_ratio = king_ppsf / pierce_ppsf
print(f"  {'$/SQUARE FEET':<25} ${king_ppsf:>14.2f} ${pierce_ppsf:>14.2f} King가 {ppsf_ratio:.2f}배 비쌈")

print(f"\n[종합 분석]")
print(f"  [O] King County: 가격 높음 + 규모 작음 = 직장 접근성 프리미엄, 도시형 소형 주택")
print(f"  [O] Pierce County: 가격 낮음 + 규모 큼 = 저렴한 대형 주택, 가족 친화적")

# ============================================================================
# 6. 도시별 시장 분석
# ============================================================================
print("\n\n【 6. 도시별 시장 분석 】")
print("="*120)

print(f"\n[King County - 상위 10개 도시]")
king_cities = king.groupby('CITY')['PRICE'].agg(['mean', 'count', 'std']).sort_values('mean', ascending=False)
print(f"  {'도시':<25} {'평균 가격':<18} {'거래 수':<10} {'표준편차':<15}")
print(f"  {'-'*68}")
for city, row in king_cities.head(10).iterrows():
    print(f"  {city:<25} ${row['mean']:>12,.0f} {int(row['count']):>8} ${row['std']:>10,.0f}")

king_top5_pct = king_cities.head(5)['count'].sum() / len(king) * 100
print(f"\n  -> 상위 5개 도시 집중도: {king_top5_pct:.1f}%")

print(f"\n[Pierce County - 상위 10개 도시]")
pierce_cities = pierce.groupby('CITY')['PRICE'].agg(['mean', 'count', 'std']).sort_values('mean', ascending=False)
print(f"  {'도시':<25} {'평균 가격':<18} {'거래 수':<10} {'표준편차':<15}")
print(f"  {'-'*68}")
for city, row in pierce_cities.head(10).iterrows():
    print(f"  {city:<25} ${row['mean']:>12,.0f} {int(row['count']):>8} ${row['std']:>10,.0f}")

pierce_top5_pct = pierce_cities.head(5)['count'].sum() / len(pierce) * 100
print(f"\n  -> 상위 5개 도시 집중도: {pierce_top5_pct:.1f}%")

# ============================================================================
# 7. 부동산 유형별 분석
# ============================================================================
print("\n\n【 7. 부동산 유형별 분석 】")
print("="*120)

print(f"\n[King County - PROPERTY TYPE별]")
king_prop = king.groupby('PROPERTY TYPE')['PRICE'].agg(['mean', 'count', 'std']).sort_values('mean', ascending=False)
print(f"  {'유형':<35} {'평균 가격':<18} {'거래 수':<10} {'비율':<10}")
print(f"  {'-'*73}")
for prop, row in king_prop.iterrows():
    pct = row['count'] / len(king) * 100
    print(f"  {prop:<35} ${row['mean']:>12,.0f} {int(row['count']):>8} {pct:>6.1f}%")

print(f"\n[Pierce County - PROPERTY TYPE별]")
pierce_prop = pierce.groupby('PROPERTY TYPE')['PRICE'].agg(['mean', 'count', 'std']).sort_values('mean', ascending=False)
print(f"  {'유형':<35} {'평균 가격':<18} {'거래 수':<10} {'비율':<10}")
print(f"  {'-'*73}")
for prop, row in pierce_prop.iterrows():
    pct = row['count'] / len(pierce) * 100
    print(f"  {prop:<35} ${row['mean']:>12,.0f} {int(row['count']):>8} {pct:>6.1f}%")

# ============================================================================
# 8. 가격대별 시장 분절 분석
# ============================================================================
print("\n\n【 8. 가격대별 시장 분절 분석 】")
print("="*120)

price_bins = [0, 400000, 600000, 800000, 1000000, 1500000, 2000000, 10000000]
price_labels = ['~400K', '400-600K', '600-800K', '800K-1M', '1-1.5M', '1.5-2M', '2M~']

king['price_segment'] = pd.cut(king['PRICE'], bins=price_bins, labels=price_labels)
pierce['price_segment'] = pd.cut(pierce['PRICE'], bins=price_bins, labels=price_labels)

print(f"\n[가격대별 분포 비교]")
print(f"  {'가격대':<15} {'King County':<20} {'Pierce County':<20}")
print(f"  {'-'*55}")

king_seg = king['price_segment'].value_counts().sort_index()
pierce_seg = pierce['price_segment'].value_counts().sort_index()

for label in price_labels:
    king_count = king_seg.get(label, 0)
    pierce_count = pierce_seg.get(label, 0)
    king_pct = king_count / len(king) * 100
    pierce_pct = pierce_count / len(pierce) * 100
    print(f"  {label:<15} {king_count:>5} ({king_pct:>5.1f}%) {pierce_count:>12} ({pierce_pct:>5.1f}%)")

# 지니 계수 계산
def gini_coefficient(prices):
    sorted_prices = np.sort(prices)
    n = len(sorted_prices)
    cumulative = np.cumsum(sorted_prices)
    return (2 * np.sum((np.arange(1, n+1) * sorted_prices)) / (n * np.sum(sorted_prices))) - (n + 1) / n

king_gini = gini_coefficient(king['PRICE'].values)
pierce_gini = gini_coefficient(pierce['PRICE'].values)

print(f"\n[시장 불평등도 (Gini 계수: 0=완전평등, 1=극도불평등)]")
print(f"  King County:   {king_gini:.4f}")
print(f"  Pierce County: {pierce_gini:.4f}")
if king_gini > pierce_gini:
    print(f"  -> King County가 더 불평등한 시장 (고가/저가 양극화)")
else:
    print(f"  -> Pierce County가 더 불평등한 시장")

# ============================================================================
# 9. 카테고리형 변수 유의성 검정 (Chi-square)
# ============================================================================
print("\n\n【 9. 카테고리형 변수 유의성 검정 (Chi-square) 】")
print("="*120)

def chi_square_test(df, col, county_name):
    # 가격을 3분위로 분류
    df['price_level'] = pd.qcut(df['PRICE'], q=3, labels=['하', '중', '상'], duplicates='drop')
    contingency = pd.crosstab(df[col], df['price_level'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    return chi2, p_value, dof

print(f"\n[King County - 카테고리형 변수 검정]")
for col in ['PROPERTY TYPE', 'CITY']:
    chi2, p_val, dof = chi_square_test(king.copy(), col, 'King')
    sig = "[O] 유의미함" if p_val < 0.05 else "[X] 유의미하지 않음"
    print(f"  {col:<20}: X2={chi2:>10.4f}, p-value={p_val:.6f} -> {sig}")

print(f"\n[Pierce County - 카테고리형 변수 검정]")
for col in ['PROPERTY TYPE', 'CITY']:
    chi2, p_val, dof = chi_square_test(pierce.copy(), col, 'Pierce')
    sig = "[O] 유의미함" if p_val < 0.05 else "[X] 유의미하지 않음"
    print(f"  {col:<20}: X2={chi2:>10.4f}, p-value={p_val:.6f} -> {sig}")

# ============================================================================
# 10. 예측 모델링 (Linear Regression vs Random Forest)
# ============================================================================
print("\n\n【 10. 예측 모델링: Linear Regression vs Random Forest 】")
print("="*120)

def build_models(df, county_name):
    # 특성 준비
    features = ['BEDS', 'BATHS', 'SQUARE FEET', 'YEAR BUILT']
    X = df[features].copy()
    y = df['PRICE'].copy()
    
    # 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_mae = mean_absolute_error(y_test, lr_pred)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)
    
    # Feature Importance
    importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    
    return {
        'lr_r2': lr_r2, 'lr_rmse': lr_rmse, 'lr_mae': lr_mae,
        'rf_r2': rf_r2, 'rf_rmse': rf_rmse, 'rf_mae': rf_mae,
        'importance': importance
    }

king_model = build_models(king, 'King County')
pierce_model = build_models(pierce, 'Pierce County')

print(f"\n[King County 모델 성능]")
print(f"  {'모델':<25} {'R²':<15} {'RMSE':<20} {'MAE':<15}")
print(f"  {'-'*75}")
print(f"  {'Linear Regression':<25} {king_model['lr_r2']:<15.4f} ${king_model['lr_rmse']:>15,.0f} ${king_model['lr_mae']:>10,.0f}")
print(f"  {'Random Forest':<25} {king_model['rf_r2']:<15.4f} ${king_model['rf_rmse']:>15,.0f} ${king_model['rf_mae']:>10,.0f}")

print(f"\n  [특성 중요도 (Random Forest)]")
for feat, imp in king_model['importance'].items():
    print(f"    {feat:<20}: {imp*100:.1f}%")

print(f"\n[Pierce County 모델 성능]")
print(f"  {'모델':<25} {'R²':<15} {'RMSE':<20} {'MAE':<15}")
print(f"  {'-'*75}")
print(f"  {'Linear Regression':<25} {pierce_model['lr_r2']:<15.4f} ${pierce_model['lr_rmse']:>15,.0f} ${pierce_model['lr_mae']:>10,.0f}")
print(f"  {'Random Forest':<25} {pierce_model['rf_r2']:<15.4f} ${pierce_model['rf_rmse']:>15,.0f} ${pierce_model['rf_mae']:>10,.0f}")

print(f"\n  [특성 중요도 (Random Forest)]")
for feat, imp in pierce_model['importance'].items():
    print(f"    {feat:<20}: {imp*100:.1f}%")

# ============================================================================
# 11. 종합 결론
# ============================================================================
print("\n\n【 11. 종합 결론 】")
print("="*120)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        King County vs Pierce County 비교                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│  지표                    │  King County          │  Pierce County              │
├─────────────────────────────────────────────────────────────────────────────────┤
│  역할                    │  일자리 중심지 (Seattle) │  위성 주거지 (통근지)        │
│  평균 가격               │  ${king['PRICE'].mean():>15,.0f}    │  ${pierce['PRICE'].mean():>15,.0f}           │
│  가격 프리미엄           │  +{price_premium:.1f}%               │  기준                        │
│  평균 건물 면적          │  {king_sqft:>15,.0f} sqft  │  {pierce_sqft:>15,.0f} sqft         │
│  평균 침실 수            │  {king_beds:>15.2f}개       │  {pierce_beds:>15.2f}개              │
│  $/SQUARE FEET           │  ${king_ppsf:>15.2f}      │  ${pierce_ppsf:>15.2f}             │
│  시장 불평등도 (Gini)    │  {king_gini:>15.4f}        │  {pierce_gini:>15.4f}               │
│  예측 정확도 (RF R²)     │  {king_model['rf_r2']:>15.4f}        │  {pierce_model['rf_r2']:>15.4f}               │
│  상위 5개 도시 집중도    │  {king_top5_pct:>15.1f}%       │  {pierce_top5_pct:>15.1f}%              │
└─────────────────────────────────────────────────────────────────────────────────┘
""")

print(f"\n[** King County 특징 **]")
print(f"  1. 일자리 중심지로서 가격 프리미엄 {price_premium:.1f}% 존재")
print(f"  2. 건물 면적 대비 높은 가격 ($/sqft {ppsf_ratio:.2f}배)")
print(f"  3. 도시형 소형 주택 중심")
print(f"  4. SQUARE FEET 상관계수: {king['PRICE'].corr(king['SQUARE FEET']):.4f}")

print(f"\n[** Pierce County 특징 **]")
print(f"  1. 위성 주거지로서 저렴한 대형 주택 제공")
print(f"  2. 가족 친화적 (더 넓은 면적, 더 많은 침실)")
print(f"  3. 예측 정확도가 더 높음 (시장 구조가 단순)")
print(f"  4. SQUARE FEET 상관계수: {pierce['PRICE'].corr(pierce['SQUARE FEET']):.4f}")

print(f"\n[** 투자 시사점 **]")
print(f"  • King County: 위치 중심 투자, 도시 프리미엄 활용")
print(f"  • Pierce County: 건물 규모 중심 투자, 신축/대형 주택 개발")

print("\n" + "="*120)
print("【 심층 분석 완료 】")
print("="*120)
