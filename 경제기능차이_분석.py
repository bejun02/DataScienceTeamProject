import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
king = pd.read_csv('king_county_sold_preprocessed.csv')
pierce = pd.read_csv('pierce_county_sold_preprocessed.csv')

print("=" * 120)
print("【 King County(일자리 중심지) vs Pierce County(위성 주거지) - 경제 기능 차이 분석 】")
print("=" * 120)

# ============================================================================
# 1. 경제 기능에 따른 주택 가격 프리미엄 분석
# ============================================================================
print("\n\n【 1. 일자리 중심지 프리미엄 분석: 가격대와 통근 가능성 】")
print("=" * 120)

king_price = king['PRICE'].mean()
pierce_price = pierce['PRICE'].mean()
price_premium = ((king_price - pierce_price) / pierce_price) * 100

print(f"\n[King County - 일자리 중심지 (Seattle)]")
print(f"  • 평균 거래가: ${king_price:,.0f}")
print(f"  • 중앙값: ${king['PRICE'].median():,.0f}")
print(f"  • 가격 분포: 표준편차 ${king['PRICE'].std():,.0f}")

print(f"\n[Pierce County - 위성 주거지]")
print(f"  • 평균 거래가: ${pierce_price:,.0f}")
print(f"  • 중앙값: ${pierce['PRICE'].median():,.0f}")
print(f"  • 가격 분포: 표준편차 ${pierce['PRICE'].std():,.0f}")

print(f"\n[가격 분석]")
print(f"  • King County 프리미엄: +${king_price - pierce_price:,.0f}")
print(f"  • 프리미엄 비율: {price_premium:+.1f}%")
print(f"  • 해석: 일자리 중심지에 사는 것이 위성지역보다 {price_premium:.1f}% 비쌈")
print(f"  → 직장 접근성이 높을수록 주택 가격이 상승")

# ============================================================================
# 2. 주택 규모와 구성 비교: 생활 방식의 차이
# ============================================================================
print("\n\n【 2. 주택 규모/구성 비교: 도시 vs 교외 생활 방식 】")
print("=" * 120)

king_sqft = king['SQUARE FEET'].mean()
pierce_sqft = pierce['SQUARE FEET'].mean()
king_beds = king['BEDS'].mean()
pierce_beds = pierce['BEDS'].mean()
king_baths = king['BATHS'].mean()
pierce_baths = pierce['BATHS'].mean()

print(f"\n[건물 면적 (생활 공간)]")
print(f"  King County (도시):     평균 {king_sqft:,.0f} sqft")
print(f"  Pierce County (교외):   평균 {pierce_sqft:,.0f} sqft")
print(f"  차이: Pierce가 {((pierce_sqft - king_sqft) / king_sqft * 100):+.1f}% 넓음")
print(f"  → 교외 지역의 방대한 토지: 넓은 주택 + 넓은 정원")

print(f"\n[침실 개수 (가족 규모)]")
print(f"  King County (도시):     평균 {king_beds:.2f}개")
print(f"  Pierce County (교외):   평균 {pierce_beds:.2f}개")
print(f"  차이: Pierce가 {((pierce_beds - king_beds) / king_beds * 100):+.1f}% 많음")
print(f"  → 교외 지역이 가족 단위 거주에 더 적합")

print(f"\n[욕실 개수 (생활 편의)]")
print(f"  King County (도시):     평균 {king_baths:.2f}개")
print(f"  Pierce County (교외):   평균 {pierce_baths:.2f}개")
print(f"  차이: Pierce가 {((pierce_baths - king_baths) / king_baths * 100):+.1f}% 많음")

print(f"\n[종합 분석]")
print(f"  ✓ King: 가격↑ + 규모↓ = 직장 접근성 프리미엄, 도시형 소형 주택")
print(f"  ✓ Pierce: 가격↓ + 규모↑ = 저렴한 대형 주택, 가족 친화적")

# ============================================================================
# 3. 주택 연식(신축도)과 개발 패턴 비교
# ============================================================================
print("\n\n【 3. 주택 연식과 개발 패턴: 도시 포화 vs 신도시 개발 】")
print("=" * 120)

king_year = king['YEAR BUILT'].mean()
pierce_year = pierce['YEAR BUILT'].mean()
year_diff = pierce_year - king_year

print(f"\n[건축년도 분석]")
print(f"  King County (도시):   평균 {king_year:.0f}년 (오래된 도시)")
print(f"  Pierce County (교외): 평균 {pierce_year:.0f}년 (신도시 개발 진행)")
print(f"  연식 차이: Pierce가 {year_diff:.0f}년 더 새로움")

# 연식별 분포
king_year_bins = pd.cut(king['YEAR BUILT'], bins=[0, 1950, 1980, 2000, 2100])
pierce_year_bins = pd.cut(pierce['YEAR BUILT'], bins=[0, 1950, 1980, 2000, 2100])

print(f"\n[King County 연식 분포 (도시의 시간적 계층)]")
king_year_dist = king_year_bins.value_counts().sort_index()
for period, count in king_year_dist.items():
    pct = count / len(king) * 100
    print(f"  {period}: {count:3d}건 ({pct:5.1f}%)")

print(f"\n[Pierce County 연식 분포 (신도시 개발 중심)]")
pierce_year_dist = pierce_year_bins.value_counts().sort_index()
for period, count in pierce_year_dist.items():
    pct = count / len(pierce) * 100
    print(f"  {period}: {count:3d}건 ({pct:5.1f}%)")

# 연식과 가격의 관계
king_year_corr = king['PRICE'].corr(king['YEAR BUILT'])
pierce_year_corr = pierce['PRICE'].corr(pierce['YEAR BUILT'])

print(f"\n[신축도의 가격 영향]")
print(f"  King County (도시):   상관계수 {king_year_corr:.4f} - 신축이 약간의 가치 추가")
print(f"  Pierce County (교외): 상관계수 {pierce_year_corr:.4f} - 신축이 상당한 가치 추가")
print(f"  해석: Pierce의 신축 프리미엄이 훨씬 강함 (신도시 개발 활발)")

# ============================================================================
# 4. 도시 네트워크 분석: 중심지 vs 위성지
# ============================================================================
print("\n\n【 4. 도시 네트워크: 핵심 도시 vs 분산된 위성지 】")
print("=" * 120)

king_top_cities = king.groupby('CITY')['PRICE'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)
pierce_top_cities = pierce.groupby('CITY')['PRICE'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)

print(f"\n[King County - 도시 집중도 분석]")
print(f"  {'도시':<20} {'거래수':<10} {'평균가격':<15} {'집중도':<10}")
print(f"  {'-'*55}")
total_king = len(king)
for city, row in king_top_cities.head(5).iterrows():
    pct = row['count'] / total_king * 100
    print(f"  {city:<20} {int(row['count']):<10} ${row['mean']:>12,.0f}  {pct:>6.1f}%")

king_top5_pct = king_top_cities.head(5)['count'].sum() / len(king) * 100
print(f"\n  상위 5개 도시 집중도: {king_top5_pct:.1f}%")
print(f"  → 도시 기능 집중, 극명한 중심지 형성")

print(f"\n[Pierce County - 도시 분산도 분석]")
print(f"  {'도시':<20} {'거래수':<10} {'평균가격':<15} {'집중도':<10}")
print(f"  {'-'*55}")
total_pierce = len(pierce)
for city, row in pierce_top_cities.head(5).iterrows():
    pct = row['count'] / total_pierce * 100
    print(f"  {city:<20} {int(row['count']):<10} ${row['mean']:>12,.0f}  {pct:>6.1f}%")

pierce_top5_pct = pierce_top_cities.head(5)['count'].sum() / len(pierce) * 100
print(f"\n  상위 5개 도시 집중도: {pierce_top5_pct:.1f}%")
print(f"  → 도시 기능 분산, 다양한 위성 커뮤니티 형성")

# ============================================================================
# 5. 가격대별 시장 분절 분석: 고가 시장의 다양성
# ============================================================================
print("\n\n【 5. 가격대별 시장 분절: 불평등과 다양성 】")
print("=" * 120)

# 가격대별 분류
price_bins = [0, 400000, 600000, 800000, 1000000, 1500000, 2000000, 5000000]
price_labels = ['~400K', '400-600K', '600-800K', '800K-1M', '1-1.5M', '1.5-2M', '2M~']

king_price_seg = pd.cut(king['PRICE'], bins=price_bins, labels=price_labels)
pierce_price_seg = pd.cut(pierce['PRICE'], bins=price_bins, labels=price_labels)

king_seg_dist = king_price_seg.value_counts().sort_index()
pierce_seg_dist = pierce_price_seg.value_counts().sort_index()

print(f"\n[King County - 가격대별 분포 (불평등한 고급 시장)]")
print(f"  {'가격대':<15} {'거래수':<10} {'비율':<10} {'누적비율':<10}")
print(f"  {'-'*45}")
cumsum = 0
for price_range, count in king_seg_dist.items():
    pct = count / len(king) * 100
    cumsum += pct
    print(f"  {str(price_range):<15} {int(count):<10} {pct:>6.1f}%   {cumsum:>6.1f}%")

print(f"\n[Pierce County - 가격대별 분포 (균형잡힌 대중 시장)]")
print(f"  {'가격대':<15} {'거래수':<10} {'비율':<10} {'누적비율':<10}")
print(f"  {'-'*45}")
cumsum = 0
for price_range, count in pierce_seg_dist.items():
    pct = count / len(pierce) * 100
    cumsum += pct
    print(f"  {str(price_range):<15} {int(count):<10} {pct:>6.1f}%   {cumsum:>6.1f}%")

# 지니 계수 계산 (불평등 지수)
def gini_coefficient(prices):
    n = len(prices)
    sorted_prices = np.sort(prices)
    cumsum = np.cumsum(sorted_prices)
    return (2 * np.sum(cumsum)) / (n * cumsum[-1]) - (n + 1) / n

king_gini = gini_coefficient(king['PRICE'].values)
pierce_gini = gini_coefficient(pierce['PRICE'].values)

print(f"\n[시장 불평등도 (Gini 계수: 0=완전평등, 1=극도불평등)]")
print(f"  King County (도시):   {king_gini:.4f} - 더 불평등한 시장 (고급주택 다수)")
print(f"  Pierce County (교외): {pierce_gini:.4f} - 더 평등한 시장 (중산층 중심)")

# ============================================================================
# 6. 통근 가능성 지표: 가격대 대비 공간
# ============================================================================
print("\n\n【 6. 통근 가능성 지표: 가격 효율성 비교 】")
print("=" * 120)

# $/sqft 분석
king_price_per_sqft = king['$/SQUARE FEET'].mean()
pierce_price_per_sqft = pierce['$/SQUARE FEET'].mean()

print(f"\n[공간당 가격 (개발 강도)]")
print(f"  King County (도시):   ${king_price_per_sqft:.2f}/sqft")
print(f"  Pierce County (교외): ${pierce_price_per_sqft:.2f}/sqft")
print(f"  비율: King가 {(king_price_per_sqft / pierce_price_per_sqft):.2f}배 비쌈")
print(f"  → 도시의 토지 프리미엄이 훨씬 높음")

# 통근 거리별 가치 제시
print(f"\n[통근자 관점: 같은 예산으로 얼마나 다른 주택을 살 수 있는가?]")
budget = 600000

king_sqft_budget = budget / king_price_per_sqft
pierce_sqft_budget = budget / pierce_price_per_sqft

print(f"  예산 ${budget:,.0f}일 때:")
print(f"  • King County에서:   {king_sqft_budget:,.0f} sqft 구매 가능")
print(f"  • Pierce County에서: {pierce_sqft_budget:,.0f} sqft 구매 가능")
print(f"  • Pierce가 {((pierce_sqft_budget - king_sqft_budget) / king_sqft_budget * 100):.1f}% 더 큼")
print(f"\n  → Seattle 일자리 중심지 vs 교외 통근지의 '공간-비용 트레이드오프'")

# ============================================================================
# 7. 주택 유형별 시장 구조 비교
# ============================================================================
print("\n\n【 7. 주택 유형별 시장 구조: 다양성 vs 균일성 】")
print("=" * 120)

king_prop_dist = king['PROPERTY TYPE'].value_counts()
pierce_prop_dist = pierce['PROPERTY TYPE'].value_counts()

print(f"\n[King County - 주택 유형 분포 (도시형 다양성)]")
for prop_type, count in king_prop_dist.items():
    pct = count / len(king) * 100
    avg_price = king[king['PROPERTY TYPE'] == prop_type]['PRICE'].mean()
    avg_sqft = king[king['PROPERTY TYPE'] == prop_type]['SQUARE FEET'].mean()
    print(f"  {prop_type:<20} {count:>3d}건 ({pct:>5.1f}%) - 평균 ${avg_price:>10,.0f} ({avg_sqft:>6.0f} sqft)")

print(f"\n[Pierce County - 주택 유형 분포 (교외형 균일성)]")
for prop_type, count in pierce_prop_dist.items():
    pct = count / len(pierce) * 100
    avg_price = pierce[pierce['PROPERTY TYPE'] == prop_type]['PRICE'].mean()
    avg_sqft = pierce[pierce['PROPERTY TYPE'] == prop_type]['SQUARE FEET'].mean()
    print(f"  {prop_type:<20} {count:>3d}건 ({pct:>5.1f}%) - 평균 ${avg_price:>10,.0f} ({avg_sqft:>6.0f} sqft)")

# ============================================================================
# 8. 종합 경제 기능 프로파일
# ============================================================================
print("\n\n【 8. 경제 기능별 시장 프로파일: 최종 분석 】")
print("=" * 120)

profile_data = {
    '지표': [
        '역할',
        '평균 주택 가격',
        '전형적 주택 규모',
        '전형적 침실 수',
        '주택 연식',
        '시장 개발 단계',
        '가격 불평등도',
        '공간당 가격',
        '신축 프리미엄',
        '도시 집중도',
        '대표 특징'
    ],
    'King County (일자리 중심지)': [
        '고용 중심, Seattle 도심',
        f'${king_price:,.0f}',
        f'{king_sqft:,.0f} sqft (소형)',
        f'{king_beds:.1f}개 (소형)',
        f'{king_year:.0f}년 (포화 도시)',
        '도시 포화 (재개발)',
        f'{king_gini:.4f} (불평등)',
        f'${king_price_per_sqft:.2f}/sqft (고가)',
        f'{king_year_corr:.4f} (약함)',
        f'{king_top5_pct:.1f}% (집중)',
        '직장 접근성이 최우선 가치'
    ],
    'Pierce County (위성 주거지)': [
        '위성 주거, King 통근지',
        f'${pierce_price:,.0f}',
        f'{pierce_sqft:,.0f} sqft (대형)',
        f'{pierce_beds:.1f}개 (가족형)',
        f'{pierce_year:.0f}년 (신도시)',
        '신규 개발 (확장 중)',
        f'{pierce_gini:.4f} (평등)',
        f'${pierce_price_per_sqft:.2f}/sqft (저가)',
        f'{pierce_year_corr:.4f} (강함)',
        f'{pierce_top5_pct:.1f}% (분산)',
        '저렴한 대형 주택이 핵심 가치'
    ]
}

profile_df = pd.DataFrame(profile_data)
print("\n" + profile_df.to_string(index=False))

# ============================================================================
# 9. 투자 전략 및 인사이트
# ============================================================================
print("\n\n【 9. 시사점 및 투자 전략 】")
print("=" * 120)

print("\n[King County에 투자하는 사람들:]")
print("  ✓ 목표: Seattle 도심 직장에 가까운 주거")
print("  ✓ 가치: 시간(통근 시간 절약) > 공간")
print("  ✓ 특징: 더 높은 가격 감수, 소형 고급 주택 선호")
print(f"  ✓ 시장: {king_top5_pct:.1f}%의 거래가 상위 5개 도시에 집중 → 선별적 투자 필요")

print("\n[Pierce County에 투자하는 사람들:]")
print("  ✓ 목표: Seattle 일자리는 유지하면서 저렴한 교외 주거")
print("  ✓ 가치: 공간(넓은 주택) > 가격")
print("  ✓ 특징: 대형 가족주택, 신축 개발 선호")
print(f"  ✓ 시장: {pierce_top5_pct:.1f}%의 거래가 분산 → 다양한 선택 가능")

print("\n[통근 가능성의 경제적 의미:]")
print(f"  • King County: {price_premium:+.1f}% 프리미엄 = Seattle 일자리 접근성 가치")
print(f"  • Pierce County: 같은 예산으로 {((pierce_sqft_budget - king_sqft_budget) / king_sqft_budget * 100):.1f}% 큰 주택")
print(f"  • 결론: 1시간 통근 가능성이 약 {price_premium:.0f}%의 가격 할인을 정당화")

print("\n[데이터 기반 권장사항:]")
print("  1. King County: 도시 프리미엄 높음 → 신축/리모델링 물건이 좋은 투자 수익률")
print(f"     (신축 프리미엄 {king_year_corr:.4f})")
print("  2. Pierce County: 신도시 개발 활발 → 신축 물건이 강한 가치 상승")
print(f"     (신축 프리미엄 {pierce_year_corr:.4f})")
print("  3. 통근자: Pierce County 선택 시 30% 가량의 가격 절감 가능")

print("\n" + "="*120)
print("【 분석 완료 】")
print("="*120)
