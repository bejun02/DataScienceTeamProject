import pandas as pd
import numpy as np

"""
================================================================================
데이터 수집 정보
================================================================================
- 데이터 출처: Redfin (https://www.redfin.com)
- 수집 날짜: 2024년 11월
- 데이터 기간: 2024년 5월 ~ 2024년 10월 (최근 6개월간 판매 완료 매물)
- 지역: 미국 워싱턴 주 King County, Pierce County
- 데이터 유형: 판매 완료(Sold) 부동산 거래 데이터
================================================================================
"""

def preprocess_data(csv_file, county_name, verbose=False):
    """
    부동산 데이터를 전처리하는 함수
    
    전처리 과정:
    1. CSV 파일 불러오기
    2. 불필요한 칼럼 제거 (27개 → 10개)
    3. 결측치 제거 (dropna)
    4. 주거용 부동산 필터링 (PROPERTY TYPE)
    5. 이상치 제거 (직접 데이터 확인 후 극단값 2개만 제거)
    
    Parameters:
    -----------
    csv_file : str
        CSV 파일 경로
    county_name : str
        카운티 이름 (출력용)
    verbose : bool
        상세 출력 여부
        
    Returns:
    --------
    df : DataFrame
        전처리된 데이터프레임
    """
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"【 {county_name} 데이터 전처리 시작 】")
        print(f"{'='*80}")
    
    # 1. 데이터 불러오기
    df = pd.read_csv(csv_file)
    if verbose:
        print(f"\n▶ 원본 데이터 로드")
        print(f"   초기 shape: {df.shape}")
    
    # 2. 불필요한 칼럼 제거
    # 제거 이유:
    # - 거래 메타데이터 (SALE TYPE, SOLD DATE, STATUS)
    # - 중복 위치 정보 (ADDRESS, ZIP, LOCATION - CITY와 위도/경도로 대체)
    # - 결측치 비율 높음 (LOT SIZE, HOA/MONTH)
    # - 시스템/사용자 정보 (URL, SOURCE, MLS#, FAVORITE, INTERESTED)
    # - 판매 완료 데이터에 불필요 (NEXT OPEN HOUSE 관련)
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
        'ADDRESS',
        'LOCATION',
        'LOT SIZE',
        'DAYS ON MARKET'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    if verbose:
        print(f"\n▶ 칼럼 제거 완료")
        print(f"   남은 칼럼: {list(df.columns)}")
        print(f"   shape: {df.shape}")

    # 3. 결측치 확인 및 제거
    # 방법: dropna() - 결측치가 있는 행 전체 제거
    # 이유: 주요 변수(PRICE, SQUARE FEET, 위도/경도)의 결측치는 분석 불가
    #       평균값 대체는 분포 왜곡 우려로 제외
    if verbose:
        print(f"\n▶ 결측치 현황:")
        print(df.isnull().sum())
    
    before_drop = len(df)
    df = df.dropna()
    after_drop = len(df)
    
    if verbose:
        print(f"\n▶ 결측치 제거 완료")
        print(f"   제거된 행: {before_drop - after_drop}개")
        print(f"   shape: {df.shape}")
    
    # 4. 주거용 부동산 필터링 (PROPERTY TYPE)
    # 제거 대상:
    # - Mobile/Manufactured Home: 이동식 주택, 일반 주거용과 가격 구조 다름
    # - Multi-Family (2-4 Unit): 다세대 투자용 부동산, 단위당 가격 비교 어려움
    # - Vacant Land: 건물 없는 토지, 건물 속성(SQFT, BEDS 등) 분석 불가
    # 유지 대상: Single Family Residential, Townhouse, Condo/Co-op
    types_to_remove = ['Mobile/Manufactured Home', 'Multi-Family (2-4 Unit)', 'Vacant Land']
    before_filter = len(df)
    df = df[~df['PROPERTY TYPE'].isin(types_to_remove)]
    after_filter = len(df)
    
    if verbose:
        print(f"\n▶ PROPERTY TYPE 필터링 완료")
        print(f"   제거된 타입: {types_to_remove}")
        print(f"   제거된 행: {before_filter - after_filter}개")
        print(f"   shape: {df.shape}")
        print(f"\n   남은 PROPERTY TYPE:")
        print(df['PROPERTY TYPE'].value_counts())
    
    # 5. 이상치 제거
    # ※ 이상치 제거 기준 설명:
    #    IQR 방식이 아닌 "직접 데이터 확인" 후 극단값만 제거
    #    - 전체 데이터를 가격순 정렬하여 육안 검토
    #    - $7,050,000: King County 최고가, 2위($2,895,000) 대비 2.4배 이상 (단일 극단값)
    #    - $4,500,000: 3위 가격, 일반 분포에서 크게 이탈
    #    - 이 2건은 일반 주거용이 아닌 특수 매물(대저택/상업겸용 등)로 추정
    #    - 나머지 데이터는 연속적인 분포를 보여 유지
    #    → 총 2건만 제거하여 데이터 손실 최소화
    before_outlier = len(df)
    outlier_prices = [7050000, 4500000]  # 직접 확인한 극단값 2개
    df = df[~df['PRICE'].isin(outlier_prices)]
    after_outlier = len(df)
    
    if verbose:
        print(f"\n▶ 이상치 제거 완료")
        print(f"   제거 기준: 직접 데이터 확인 후 극단값 2개만 제거")
        print(f"   제거된 가격: ${outlier_prices[0]:,}, ${outlier_prices[1]:,}")
        print(f"   제거된 행: {before_outlier - after_outlier}개")
        print(f"   shape: {df.shape}")
    
    return df


def create_property_type_dummies(df):
    """
    PROPERTY TYPE에 대한 더미 변수 생성
    
    회귀 모델에서 범주형 변수를 사용하기 위해 One-Hot Encoding 적용
    기준 범주(Reference): Single Family Residential (가장 빈도 높음)
    
    Parameters:
    -----------
    df : DataFrame
        전처리된 데이터프레임
        
    Returns:
    --------
    df : DataFrame
        더미 변수가 추가된 데이터프레임
    """
    # One-Hot Encoding (기준 범주 제외)
    dummies = pd.get_dummies(df['PROPERTY TYPE'], prefix='TYPE', drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    
    return df


def get_preprocessing_summary():
    """
    전처리 과정 요약 정보 반환
    """
    summary = """
    ================================================================================
    데이터 전처리 요약
    ================================================================================
    
    【 데이터 수집 정보 】
    - 출처: Redfin (https://www.redfin.com)
    - 수집일: 2024년 11월
    - 기간: 2024년 5월 ~ 2024년 10월 (6개월)
    - 지역: 워싱턴 주 King County, Pierce County
    
    【 전처리 단계 】
    Step 1. 칼럼 제거: 27개 → 10개
            (거래 메타, 중복 위치, 결측 많은 칼럼 제거)
            
    Step 2. 결측치 제거: dropna()
            (주요 변수 결측은 분석 불가, 평균 대체 시 분포 왜곡)
            
    Step 3. PROPERTY TYPE 필터링
            - 유지: Single Family, Townhouse, Condo/Co-op
            - 제거: Mobile Home, Multi-Family, Vacant Land
            
    Step 4. 이상치 제거
            ※ IQR 방식이 아닌 "직접 데이터 확인" 방식 사용
            - 가격순 정렬 후 육안 검토
            - 극단값 2건만 제거: $7,050,000, $4,500,000
            - 사유: 2위 대비 2.4배 이상 이탈, 특수 매물 추정
            - 나머지 데이터는 연속 분포 → 유지
    
    【 최종 데이터 】
    - King County: 347건
    - Pierce County: 320건
    - 칼럼: PROPERTY TYPE, CITY, PRICE, BEDS, BATHS, SQUARE FEET,
            YEAR BUILT, $/SQUARE FEET, LATITUDE, LONGITUDE
    ================================================================================
    """
    return summary


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    print(get_preprocessing_summary())
    
    # King County 전처리 실행
    print("\n" + "="*80)
    king_df = preprocess_data('King_County_Sold.csv', 'King County', verbose=True)
    
    # Pierce County 전처리 실행
    print("\n" + "="*80)
    pierce_df = preprocess_data('Pierce_County_Sold.csv', 'Pierce County', verbose=True)
    
    # 더미 변수 생성 예시
    print("\n" + "="*80)
    print("【 PROPERTY TYPE 더미 변수 생성 】")
    king_df_with_dummies = create_property_type_dummies(king_df)
    print(f"더미 변수 추가 후 칼럼: {list(king_df_with_dummies.columns)}")
    print(f"\n더미 변수 분포:")
    for col in king_df_with_dummies.columns:
        if col.startswith('TYPE_'):
            print(f"  {col}: {king_df_with_dummies[col].sum()}건")