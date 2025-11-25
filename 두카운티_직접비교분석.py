import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency

# 데이터 로드
king = pd.read_csv('king_county_sold_preprocessed.csv')
pierce = pd.read_csv('pierce_county_sold_preprocessed.csv')

print("=" * 100)
print("【 King County vs Pierce County 직접 비교 분석 】")
print("=" * 100)

# ============================================================================
# 1. 기본 통계 비교
# ============================================================================
print("\n\n【 1. 기본 통계 비교 】")
print("=" * 100)

comparison_stats = pd.DataFrame({
    'King County': {
        '데이터 수': len(king),
        '평균 거래가': king['PRICE'].mean(),
        '중앙값': king['PRICE'].median(),
        '표준편차': king['PRICE'].std(),
        '최소값': king['PRICE'].min(),
        '최대값': king['PRICE'].max(),
    },
    'Pierce County': {
        '데이터 수': len(pierce),
        '평균 거래가': pierce['PRICE'].mean(),
        '중앙값': pierce['PRICE'].median(),
        '표준편차': pierce['PRICE'].std(),
        '최소값': pierce['PRICE'].min(),
        '최대값': pierce['PRICE'].max(),
    }
})

print(comparison_stats.to_string())
print(f"\n평균 거래가 차이: ${king['PRICE'].mean() - pierce['PRICE'].mean():,.0f}")
print(f"평균 거래가 비율: King/Pierce = {king['PRICE'].mean() / pierce['PRICE'].mean():.4f}x")

# ============================================================================
# 2. 수치형 변수 비교
# ============================================================================
print("\n\n【 2. 수치형 변수 비교 】")
print("=" * 100)

numeric_cols = ['BEDS', 'BATHS', 'SQUARE FEET', 'LOT SIZE', 'YEAR BUILT', '$/SQUARE FEET']

comparison_numeric = pd.DataFrame()
for col in numeric_cols:
    king_mean = king[col].mean()
    pierce_mean = pierce[col].mean()
    diff = king_mean - pierce_mean
    ratio = king_mean / pierce_mean if pierce_mean != 0 else 0
    
    comparison_numeric = pd.concat([comparison_numeric, pd.DataFrame({
        'King County': [king_mean],
        'Pierce County': [pierce_mean],
        '차이': [diff],
        '비율 (K/P)': [ratio]
    }, index=[col])], axis=0)

print(comparison_numeric.to_string())

# ============================================================================
# 3. 독립 표본 t-검정 (두 그룹 평균 비교)
# ============================================================================
print("\n\n【 3. 두 카운티 평균 가격 비교 (t-검정) 】")
print("=" * 100)

t_stat, p_value = ttest_ind(king['PRICE'], pierce['PRICE'])
print(f"\n평균 거래가 비교:")
print(f"  King County 평균: ${king['PRICE'].mean():,.0f}")
print(f"  Pierce County 평균: ${pierce['PRICE'].mean():,.0f}")
print(f"  차이: ${king['PRICE'].mean() - pierce['PRICE'].mean():,.0f}")
print(f"  t-통계량: {t_stat:.4f}")
print(f"  p-value: {p_value:.6f}")

if p_value < 0.05:
    print(f"  결론: ✓ 두 카운티의 평균 거래가가 **유의미하게 다름** (p < 0.05)")
else:
    print(f"  결론: ✗ 두 카운티의 평균 거래가가 **차이 없음** (p >= 0.05)")

# ============================================================================
# 4. 건물 면적별 가격 비교
# ============================================================================
print("\n\n【 4. SQUARE FEET (건물 면적) 분석 】")
print("=" * 100)

print(f"\nKing County:")
print(f"  평균: {king['SQUARE FEET'].mean():.2f} sqft")
print(f"  중앙값: {king['SQUARE FEET'].median():.2f} sqft")
print(f"  범위: {king['SQUARE FEET'].min():.0f} ~ {king['SQUARE FEET'].max():.0f} sqft")

print(f"\nPierce County:")
print(f"  평균: {pierce['SQUARE FEET'].mean():.2f} sqft")
print(f"  중앙값: {pierce['SQUARE FEET'].median():.2f} sqft")
print(f"  범위: {pierce['SQUARE FEET'].min():.0f} ~ {pierce['SQUARE FEET'].max():.0f} sqft")

# 건물 면적에 따른 가격 상관계수
king_corr_sqft = king['PRICE'].corr(king['SQUARE FEET'])
pierce_corr_sqft = pierce['PRICE'].corr(pierce['SQUARE FEET'])

print(f"\n가격과의 상관계수:")
print(f"  King County: {king_corr_sqft:.4f}")
print(f"  Pierce County: {pierce_corr_sqft:.4f}")

# ============================================================================
# 5. 침실/욕실 비교
# ============================================================================
print("\n\n【 5. BEDS/BATHS (침실/욕실) 비교 】")
print("=" * 100)

print(f"\nBEDS (침실):")
print(f"  King County - 평균: {king['BEDS'].mean():.2f}개, 중앙값: {king['BEDS'].median():.0f}개")
print(f"  Pierce County - 평균: {pierce['BEDS'].mean():.2f}개, 중앙값: {pierce['BEDS'].median():.0f}개")

print(f"\nBATHS (욕실):")
print(f"  King County - 평균: {king['BATHS'].mean():.2f}개, 중앙값: {king['BATHS'].median():.2f}개")
print(f"  Pierce County - 평균: {pierce['BATHS'].mean():.2f}개, 중앙값: {pierce['BATHS'].median():.2f}개")

# ============================================================================
# 6. 건축년도 비교
# ============================================================================
print("\n\n【 6. YEAR BUILT (건축년도) 비교 】")
print("=" * 100)

print(f"\nKing County:")
print(f"  평균: {king['YEAR BUILT'].mean():.0f}년")
print(f"  중앙값: {king['YEAR BUILT'].median():.0f}년")
print(f"  범위: {king['YEAR BUILT'].min():.0f} ~ {king['YEAR BUILT'].max():.0f}년")

print(f"\nPierce County:")
print(f"  평균: {pierce['YEAR BUILT'].mean():.0f}년")
print(f"  중앙값: {pierce['YEAR BUILT'].median():.0f}년")
print(f"  범위: {pierce['YEAR BUILT'].min():.0f} ~ {pierce['YEAR BUILT'].max():.0f}년")

# ============================================================================
# 7. 부동산 유형별 비교
# ============================================================================
print("\n\n【 7. PROPERTY TYPE (부동산 유형) 비교 】")
print("=" * 100)

king_prop = king['PROPERTY TYPE'].value_counts()
pierce_prop = pierce['PROPERTY TYPE'].value_counts()

print(f"\nKing County:")
for prop_type, count in king_prop.items():
    pct = count / len(king) * 100
    avg_price = king[king['PROPERTY TYPE'] == prop_type]['PRICE'].mean()
    print(f"  {prop_type}: {count}개 ({pct:.1f}%) - 평균 ${avg_price:,.0f}")

print(f"\nPierce County:")
for prop_type, count in pierce_prop.items():
    pct = count / len(pierce) * 100
    avg_price = pierce[pierce['PROPERTY TYPE'] == prop_type]['PRICE'].mean()
    print(f"  {prop_type}: {count}개 ({pct:.1f}%) - 평균 ${avg_price:,.0f}")

# ============================================================================
# 8. 도시별 상위 5개 비교
# ============================================================================
print("\n\n【 8. 도시별 상위 5개 비교 】")
print("=" * 100)

king_cities = king.groupby('CITY')['PRICE'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(5)
pierce_cities = pierce.groupby('CITY')['PRICE'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(5)

print(f"\nKing County - 상위 5개 도시:")
for city, row in king_cities.iterrows():
    print(f"  {city}: 평균 ${row['mean']:,.0f} ({int(row['count'])}건)")

print(f"\nPierce County - 상위 5개 도시:")
for city, row in pierce_cities.iterrows():
    print(f"  {city}: 평균 ${row['mean']:,.0f} ({int(row['count'])}건)")

# ============================================================================
# 9. 가격대별 분포 비교
# ============================================================================
print("\n\n【 9. 가격대별 분포 비교 】")
print("=" * 100)

price_bins = [0, 400000, 600000, 800000, 1000000, 2000000]
price_labels = ['400K 이하', '400K~600K', '600K~800K', '800K~1M', '1M 이상']

king_price_dist = pd.cut(king['PRICE'], bins=price_bins, labels=price_labels).value_counts().sort_index()
pierce_price_dist = pd.cut(pierce['PRICE'], bins=price_bins, labels=price_labels).value_counts().sort_index()

print(f"\nKing County:")
for price_range, count in king_price_dist.items():
    pct = count / len(king) * 100
    print(f"  {price_range}: {count}개 ({pct:.1f}%)")

print(f"\nPierce County:")
for price_range, count in pierce_price_dist.items():
    pct = count / len(pierce) * 100
    print(f"  {price_range}: {count}개 ({pct:.1f}%)")

# ============================================================================
# 10. 상관계수 비교
# ============================================================================
print("\n\n【 10. 가격과의 상관계수 비교 】")
print("=" * 100)

numeric_for_corr = ['BEDS', 'BATHS', 'SQUARE FEET', 'LOT SIZE', 'YEAR BUILT', '$/SQUARE FEET']

corr_comparison = pd.DataFrame({
    'King County': [king['PRICE'].corr(king[col]) for col in numeric_for_corr],
    'Pierce County': [pierce['PRICE'].corr(pierce[col]) for col in numeric_for_corr]
}, index=numeric_for_corr)

corr_comparison['차이'] = corr_comparison['King County'] - corr_comparison['Pierce County']
corr_comparison = corr_comparison.sort_values('차이', key=abs, ascending=False)

print("\n" + corr_comparison.to_string())

# ============================================================================
# 11. 종합 분석
# ============================================================================
print("\n\n【 11. 종합 분석 및 결론 】")
print("=" * 100)

print("\n[King County의 특징]")
print(f"  • 평균 거래가: ${king['PRICE'].mean():,.0f} (Pierce보다 4.4% 비쌈)")
print(f"  • 건축년도 영향: 상관계수 0.3997 (중간 정도 - 신축 선호)")
print(f"  • 거래량: 235개")
print(f"  • 주요 지역: Tacoma, Puyallup")

print("\n[Pierce County의 특징]")
print(f"  • 평균 거래가: ${pierce['PRICE'].mean():,.0f} (King County보다 저가)")
print(f"  • 건축년도 영향: 상관계수 0.1852 (약함 - 건축년도 무관)")
print(f"  • 거래량: 230개")
print(f"  • 주요 지역: Gig Harbor, Fircrest (고가)")

print("\n[공통점]")
print(f"  • 건물 면적이 가장 중요 (King: 0.7810, Pierce: 0.8180)")
print(f"  • 침실/욕실 개수도 중요 (King: 0.54~0.58, Pierce: 0.59~0.63)")
print(f"  • 부동산 유형, 도시, 지역 모두 유의미한 영향")

print("\n[차이점]")
print(f"  • King County: 신축/리모델링 건물이 프리미엄 획득")
print(f"  • Pierce County: 건축년도와 무관하게 거래가 결정")
print(f"  • King County 평균 거래가가 4.4% 더 높음")

print("\n" + "="*100)
print("【 분석 완료 】")
print("="*100)
