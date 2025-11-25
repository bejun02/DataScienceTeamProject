import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def analyze_data(csv_file, county_name):
    """
    전처리된 부동산 데이터를 분석하는 함수
    - 상관계수 분석
    - 카테고리별 가격 분석
    - 카테고리형 변수의 유의성 검정
    """
    print(f"\n{'='*80}")
    print(f"【 {county_name} 데이터 분석 】")
    print(f"{'='*80}")
    
    # 데이터 불러오기
    df = pd.read_csv(csv_file)
    print(f"\n▶ 데이터 로드")
    print(f"   shape: {df.shape}")
    
    # 숫자 칼럼과 가격의 상관계수
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print(f"\n【 1. 숫자 칼럼과 PRICE의 상관계수 】")
    correlations = df[numeric_cols].corr()['PRICE'].sort_values(ascending=False)
    print(correlations)
    print("\n" + "="*50)
    
    # 가장 강한 상관관계를 가진 칼럼 분석
    print(f"\n【 2. 가격 변동과 가장 밀접한 관계의 칼럼 분석 】")
    top_corr_cols = correlations[correlations.index != 'PRICE'].head(5)
    print("\n상위 5개 상관계수:")
    for col, corr_value in top_corr_cols.items():
        print(f"  {col}: {corr_value:.4f}")
    
    # 각 숫자 칼럼별 상세 통계
    print(f"\n【 3. 상위 관련성 칼럼의 상세 통계 】")
    for col in top_corr_cols.index:
        print(f"\n▶ {col}")
        print(f"   상관계수: {correlations[col]:.4f}")
        print(f"   평균: {df[col].mean():.2f}")
        print(f"   표준편차: {df[col].std():.2f}")
        print(f"   최소값: {df[col].min():.2f}")
        print(f"   최대값: {df[col].max():.2f}")
        print(f"   중앙값: {df[col].median():.2f}")
    
    # 객체 칼럼별 분석 (ADDRESS 제외)
    object_cols = df.select_dtypes(include=['object']).columns
    object_cols = [col for col in object_cols if col != 'ADDRESS']  # ADDRESS 제외
    
    print(f"\n\n【 4. 객체(카테고리) 칼럼별 가격 분석 】")
    print(f"분석 대상: {list(object_cols)}\n")
    
    for col in object_cols:
        print(f"\n▶ {col}:")
        grouped = df.groupby(col)['PRICE'].agg(['mean', 'std', 'count', 'min', 'max'])
        print(grouped.to_string())
        
        price_range = grouped['max'].max() - grouped['min'].min()
        mean_diff = grouped['mean'].max() - grouped['mean'].min()
        print(f"   → 카테고리별 평균 가격 차이: ${mean_diff:,.0f}")
        print(f"   → 전체 가격 범위: ${price_range:,.0f}")
        print(f"   → 카테고리 개수: {len(grouped)}")
        print("-" * 70)
    
    # 카테고리형 변수의 유의성 검정 (카이제곱)
    print(f"\n\n【 5. 카테고리형 변수의 가격 연관성 검정 】")
    print(f"\n(가격을 상/중/하 3개 등급으로 분류 후 카이제곱 검정 수행)\n")
    
    for col in object_cols:
        # 가격을 상/중/하로 분류
        price_level = pd.cut(df['PRICE'], 
                            bins=[df['PRICE'].quantile(0), 
                                  df['PRICE'].quantile(0.33), 
                                  df['PRICE'].quantile(0.67), 
                                  df['PRICE'].max()],
                            labels=['하', '중', '상'],
                            include_lowest=True)
        
        # 교차표 생성
        contingency_table = pd.crosstab(df[col], price_level)
        
        print(f"\n▶ {col} vs 가격 등급 교차 분석")
        print(contingency_table.to_string())
        
        # 카이제곱 검정
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\n   카이제곱 통계량: {chi2:.4f}")
        print(f"   p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print(f"   결론: ✓ 유의미함 (p < 0.05)")
        else:
            print(f"   결론: ✗ 유의미하지 않음 (p >= 0.05)")
        print("-" * 70)
    
    # 종합 평가
    print(f"\n\n【 6. 가격 변동과의 관련성 종합 평가 】\n")
    
    print("수치형 특성 영향도 (상관계수):")
    for col in correlations[correlations.index != 'PRICE'].head(7).index:
        corr_val = correlations[col]
        if corr_val > 0.6:
            strength = "(매우 강함)"
        elif corr_val > 0.4:
            strength = "(강함)"
        elif corr_val > 0.2:
            strength = "(중간)"
        else:
            strength = "(약함)"
        print(f"  {col:25s}: {corr_val:7.4f} {strength}")
    
    print("\n카테고리형 특성 영향도 (카이제곱 검정):")
    for col in object_cols:
        price_level = pd.cut(df['PRICE'], 
                            bins=[df['PRICE'].quantile(0), 
                                  df['PRICE'].quantile(0.33), 
                                  df['PRICE'].quantile(0.67), 
                                  df['PRICE'].max()],
                            labels=['하', '중', '상'],
                            include_lowest=True)
        contingency_table = pd.crosstab(df[col], price_level)
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        if p_value < 0.05:
            sig_text = "(유의미함)"
        else:
            sig_text = "(유의미하지 않음)"
        print(f"  {col:25s}: p-value = {p_value:.6f} {sig_text}")

# ============================================================================
# 메인 실행
# ============================================================================

print("\n" + "="*80)
print("【 부동산 데이터 분석 시작 】")
print("="*80)

# King County Sold Data
analyze_data('king_county_sold_preprocessed.csv', 'King County Sold')

# Pierce County Sold Data
analyze_data('pierce_county_sold_preprocessed.csv', 'Pierce County Sold')

print("\n" + "="*80)
print("【 분석 완료! 】")
print("="*80)
