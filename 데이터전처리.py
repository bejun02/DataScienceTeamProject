import pandas as pd
import numpy as np

def preprocess_data(csv_file, county_name):
    """
    부동산 데이터를 전처리하는 함수
    - CSV 파일 불러오기
    - 불필요한 칼럼 제거
    - 결측치 제거
    - 이상치 제거
    """
    print(f"\n{'='*80}")
    print(f"【 {county_name} 데이터 전처리 시작 】")
    print(f"{'='*80}")
    
    # 1. 데이터 불러오기
    df = pd.read_csv(csv_file)
    print(f"\n▶ 원본 데이터 로드")
    print(f"   초기 shape: {df.shape}")
    
    # 2. 불필요한 칼럼 제거
    columns_to_drop = [
        'SALE TYPE',
        'STATE OR PROVINCE',
        'STATUS',
        'NEXT OPEN HOUSE START TIME',
        'NEXT OPEN HOUSE END TIME',
        'SOURCE',
        'FAVORITE',
        'URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)',
        'LATITUDE',
        'LONGITUDE',
        'INTERESTED',
        'SOLD DATE',
        'MLS#',
        'ZIP OR POSTAL CODE',
        'HOA/MONTH'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    print(f"\n▶ 칼럼 제거 완료")
    print(f"   남은 칼럼: {list(df.columns)}")
    print(f"   shape: {df.shape}")
    
    # 3. 결측치 제거
    df = df.dropna()
    print(f"\n▶ 결측치 제거 완료")
    print(f"   shape: {df.shape}")
    
    # 4. 이상치 처리 (IQR 방식) - 모든 숫자형 칼럼
    print(f"\n▶ 이상치 처리 (IQR 방식)")
    numeric_cols_for_outlier = df.select_dtypes(include=['int64', 'float64']).columns
    
    removed_indices = set()
    for col in numeric_cols_for_outlier:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        removed_indices.update(outliers.index)
        
        if len(outliers) > 0:
            print(f"   {col}: {len(outliers)}개 이상치 발견")
    
    print(f"   총 이상치 제거 행 수: {len(removed_indices)}")
    df = df.drop(removed_indices)
    print(f"   최종 shape: {df.shape}")
    
    return df

# ============================================================================
# 메인 실행
# ============================================================================

print("\n" + "="*80)
print("【 부동산 데이터 전처리 시작 】")
print("="*80)

# King County 전처리
king_df = preprocess_data('KingCounty.csv', 'King County')

# Pierce County 전처리
pierce_df = preprocess_data('PierceCounty.csv', 'Pierce County')

# 전처리된 데이터 저장
print(f"\n{'='*80}")
print("【 전처리 완료 및 저장 】")
print(f"{'='*80}")

king_df.to_csv('king_county_preprocessed.csv', index=False)
print(f"\n✓ King County 전처리 데이터 저장: king_county_preprocessed.csv")
print(f"  ({king_df.shape[0]} rows, {king_df.shape[1]} columns)")

pierce_df.to_csv('pierce_county_preprocessed.csv', index=False)
print(f"✓ Pierce County 전처리 데이터 저장: pierce_county_preprocessed.csv")
print(f"  ({pierce_df.shape[0]} rows, {pierce_df.shape[1]} columns)")

print(f"\n{'='*80}")
print("【 전처리 완료! 】")
print(f"{'='*80}\n")
