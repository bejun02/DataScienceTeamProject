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
    '''print(f"\n{'='*80}")
    print(f"【 {county_name} 데이터 전처리 시작 】")
    print(f"{'='*80}")'''
    
    # 1. 데이터 불러오기
    df = pd.read_csv(csv_file)
    '''print(f"\n▶ 원본 데이터 로드")
    print(f"   초기 shape: {df.shape}")'''
    
    # 2. 불필요한 칼럼 제거 sold date, address, location, lot size(보류), HOA/month,
    # status
    columns_to_drop = [
        'SALE TYPE',
        'STATE OR PROVINCE',
        'STATUS',
        'NEXT OPEN HOUSE START TIME',
        'NEXT OPEN HOUSE END TIME',
        'SOURCE',
        'FAVORITE',
        'URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)',
        'INTERESTED',
        'SOLD DATE',
        'MLS#',
        'ZIP OR POSTAL CODE',
        'HOA/MONTH',
        'SOLD DATE',
        'ADDRESS',
        'LOCATION',
        'LOT SIZE',  # 보류 가능
        'DAYS ON MARKET'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    print(f"\n▶ 칼럼 제거 완료")
    print(f"   남은 칼럼: {list(df.columns)}")
    print(f"   shape: {df.shape}")

    # 3. 결측치 확인 및 제거
    '''print(f"\n▶ 결측치 현황:")
    print(df.isnull().sum())'''
    
    before_drop = len(df)
    df = df.dropna()
    after_drop = len(df)
    
    '''print(f"\n▶ 결측치 제거 완료")
    print(f"   제거된 행: {before_drop - after_drop}개")
    print(f"   shape: {df.shape}")'''
    
    # 4. 특정 PROPERTY TYPE 제거
    types_to_remove = ['Mobile/Manufactured Home', 'Multi-Family (2-4 Unit)', 'Vacant Land']
    before_filter = len(df)
    df = df[~df['PROPERTY TYPE'].isin(types_to_remove)]
    after_filter = len(df)
    '''print(f"\n▶ PROPERTY TYPE 필터링 완료")
    print(f"   제거된 타입: {types_to_remove}")
    print(f"   제거된 행: {before_filter - after_filter}개")
    print(f"   shape: {df.shape}")
    print(f"\n   남은 PROPERTY TYPE:")
    print(df['PROPERTY TYPE'].value_counts())'''
    
    # 5. 이상치 제거 (PRICE = 7050000, 4500000)
    before_outlier = len(df)
    df = df[(df['PRICE'] != 7050000) & (df['PRICE'] != 4500000)]
    after_outlier = len(df)
    '''print(f"\n▶ 이상치 제거 완료 (PRICE = 7,050,000, 4,500,000)")
    print(f"   제거된 행: {before_outlier - after_outlier}개")
    print(f"   shape: {df.shape}")'''
    
    return df

# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    # King County 전처리 실행
    king_df = preprocess_data('King_County_Sold.csv', 'King County')

if __name__ == "__main__":
    # King County 전처리 실행
    king_df = preprocess_data('Pierce_County_Sold.csv', 'Pierce County')