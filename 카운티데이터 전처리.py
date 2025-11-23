import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# KingCounty 데이터 불러오기
king_df = pd.read_csv('KingCounty.csv')

# 데이터 확인
'''print(king_df.head())
print(king_df.info())
print(king_df.columns)'''

# 각 열의 상세 정보 확인
'''print("\n=== 각 열의 데이터 타입 및 결측치 ===")
print(king_df.dtypes)
print(f"\n결측치 개수:\n{king_df.isnull().sum()}")

print("\n=== 각 열의 통계 정보 ===")
print(king_df.describe(include='all'))

print("\n=== 각 열별 샘플 데이터 ===")
for col in king_df.columns:
    print(f"\n{col}: {king_df[col].unique()[:5]}")'''

# 쓸모없는 칼럼 제거
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
    'ZIP OR POSTAL CODE'
]
king_df = king_df.drop(columns=columns_to_drop, errors='ignore')
print("\n=== 칼럼 제거 완료 ===")
print(f"남은 칼럼: {list(king_df.columns)}")
print(f"데이터 shape: {king_df.shape}")

# 객체 타입 열만 필터링
'''print("\n=== 객체(Object) 타입 열 분석 ===")
object_cols = king_df.select_dtypes(include=['object']).columns
print(f"객체 타입 열: {list(object_cols)}\n")'''
# 각 객체 열의 값과 개수 확인
'''for col in object_cols:
    print(f"\n【 {col} 】")
    print(f"고유값 개수: {king_df[col].nunique()}")
    print(f"값의 분포:")
    print(king_df[col].value_counts())
    print("-" * 50)'''

# 숫자 타입 열만 필터링
'''print("\n=== 숫자(Numeric) 타입 열 분석 ===")
numeric_cols = king_df.select_dtypes(include=['int64', 'float64']).columns
print(f"숫자 타입 열: {list(numeric_cols)}\n")
# 각 숫자 열의 값과 개수 확인
for col in numeric_cols:
    print(f"\n【 {col} 】")
    print(f"고유값 개수: {king_df[col].nunique()}")
    print(f"값의 분포:")
    print(king_df[col].value_counts().sort_index())
    print("-" * 50)'''

print("\n=== 가격과의 관련성 분석 ===")

# 숫자 칼럼과 가격의 상관계수
numeric_cols = king_df.select_dtypes(include=['int64', 'float64']).columns
print("\n【 숫자 칼럼과 PRICE의 상관계수 】")
correlations = king_df[numeric_cols].corr()['PRICE'].sort_values(ascending=False)
print(correlations)

# 객체 칼럼별 평균 가격 비교
object_cols = king_df.select_dtypes(include=['object']).columns
print("\n【 객체 칼럼별 평균 가격 】")
for col in object_cols:
    print(f"\n{col}:")
    avg_price = king_df.groupby(col)['PRICE'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    print(avg_price)
    print("-" * 50)


