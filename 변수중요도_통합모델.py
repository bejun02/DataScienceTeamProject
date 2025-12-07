import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 데이터 로드
print("Loading data...")
king_df = pd.read_csv('King_County_Sold.csv', skiprows=[1])
pierce_df = pd.read_csv('Pierce_County_Sold.csv', skiprows=[1])

king_df['COUNTY'] = 'King'
pierce_df['COUNTY'] = 'Pierce'

df = pd.concat([king_df, pierce_df], ignore_index=True)

# 필요한 컬럼 선택
df = df[['PRICE', 'SQUARE FEET', 'BEDS', 'BATHS', 'YEAR BUILT', 
         'PROPERTY TYPE', 'COUNTY', 'LATITUDE', 'LONGITUDE']].dropna()

print(f"Total data: {len(df)}")

# 거리 계산 함수
from math import radians, sin, cos, sqrt, asin

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (km)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# 시애틀, 벨뷰 좌표
SEATTLE_COORDS = (47.6062, -122.3321)
BELLEVUE_COORDS = (47.6101, -122.2015)

print("Calculating distances...")
df['dist_seattle'] = df.apply(
    lambda row: haversine(row['LATITUDE'], row['LONGITUDE'], 
                          SEATTLE_COORDS[0], SEATTLE_COORDS[1]), axis=1)
df['dist_bellevue'] = df.apply(
    lambda row: haversine(row['LATITUDE'], row['LONGITUDE'], 
                          BELLEVUE_COORDS[0], BELLEVUE_COORDS[1]), axis=1)

# 더미 변수 생성
df['COUNTY_King'] = (df['COUNTY'] == 'King').astype(int)

# PROPERTY TYPE 더미 변수 (Single Family가 기준, Townhouse/Condo만 더미)
df['TYPE_Townhouse'] = (df['PROPERTY TYPE'] == 'Townhouse').astype(int)
df['TYPE_Condo'] = (df['PROPERTY TYPE'] == 'Condo/Co-op').astype(int)

# 특성 준비 (통합 모델: 9변수)
feature_cols = ['SQUARE FEET', 'BEDS', 'BATHS', 'YEAR BUILT', 
                'COUNTY_King', 'TYPE_Townhouse', 'TYPE_Condo',
                'dist_seattle', 'dist_bellevue']

X = df[feature_cols]
y = df['PRICE']

print(f"\nModel features: {len(feature_cols)}")
print(f"  Basic: SQFT, BEDS, BATHS, YEAR_BUILT")
print(f"  County: COUNTY_King")
print(f"  Type: TYPE_Townhouse, TYPE_Condo")
print(f"  Location: dist_seattle, dist_bellevue")

# Train-Test 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest 학습
print("\n🌲 Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# 성능 평가
train_score = rf_model.score(X_train, y_train)
test_score = rf_model.score(X_test, y_test)

print(f"  Train R²: {train_score:.4f}")
print(f"  Test R²: {test_score:.4f}")

# Feature Importance 추출 (내림차순 정렬)
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=True)  # 오름차순으로 정렬 (barh에서 위가 작은 값)

print("\n📊 Feature Importance (MDI):")
# 출력은 내림차순으로
for idx, row in feature_importance_df.sort_values('Importance', ascending=False).iterrows():
    print(f"  {row['Feature']}: {row['Importance']*100:.1f}%")

# 한글 라벨 매핑
label_mapping = {
    'SQUARE FEET': 'SQFT',
    'BEDS': 'BEDS',
    'BATHS': 'BATHS',
    'YEAR BUILT': 'YEAR_BUILT',
    'COUNTY_King': 'COUNTY_King',
    'TYPE_Townhouse': 'TYPE_Townhouse',
    'TYPE_Condo': 'TYPE_Condo',
    'dist_seattle': 'dist_seattle',
    'dist_bellevue': 'dist_bellevue'
}

feature_importance_df['Feature_Label'] = feature_importance_df['Feature'].map(label_mapping)

# 시각화
fig, ax = plt.subplots(figsize=(12, 8))

# 색상 지정 (카테고리별)
colors = []
for feat in feature_importance_df['Feature']:
    if feat == 'SQUARE FEET':
        colors.append('#FF8C00')  # 주황 - 면적
    elif feat in ['dist_bellevue', 'dist_seattle']:
        colors.append('#2E86AB')  # 파랑 - 위치
    elif feat == 'COUNTY_King':
        colors.append('#9370DB')  # 보라 - 카운티
    elif 'TYPE' in feat:
        colors.append('#32CD32')  # 초록 - 유형
    else:
        colors.append('#708090')  # 회색 - 기타

# 내림차순으로 정렬 (중요도 높은 것이 위로)
bars = ax.barh(range(len(feature_importance_df)), 
               feature_importance_df['Importance'],
               color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)

# Y축 라벨 설정 (내림차순)
ax.set_yticks(range(len(feature_importance_df)))
ax.set_yticklabels(feature_importance_df['Feature_Label'])

# 값 라벨 추가
for i, (idx, row) in enumerate(feature_importance_df.iterrows()):
    ax.text(row['Importance'] + 0.01, i, 
            f"{row['Importance']*100:.1f}%", 
            va='center', fontsize=11, fontweight='bold')

ax.set_xlabel('Feature Importance', fontsize=13, fontweight='bold')
ax.set_ylabel('')
ax.set_title('Random Forest Feature Importance Analysis (Unified Model)\n"SQFT accounts for over 50% of total importance"', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xlim(0, max(feature_importance_df['Importance']) * 1.15)

# 그리드 추가
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# 범례 추가
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF8C00', label='Area (SQFT)', edgecolor='black'),
    Patch(facecolor='#2E86AB', label='Location (Distance)', edgecolor='black'),
    Patch(facecolor='#9370DB', label='County', edgecolor='black'),
    Patch(facecolor='#32CD32', label='Property Type', edgecolor='black'),
    Patch(facecolor='#708090', label='Others', edgecolor='black')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9)

# 상위 75% 차지하는 변수 표시
cumsum = feature_importance_df.sort_values('Importance', ascending=False)['Importance'].cumsum()
top_75_count = (cumsum <= 0.75).sum()
if top_75_count == 0:
    top_75_count = 1

ax.axhline(y=len(feature_importance_df) - top_75_count - 0.5, 
           color='red', linestyle='--', linewidth=2, alpha=0.6)
ax.text(0.45, len(feature_importance_df) - top_75_count - 0.3, 
        f'← Top {top_75_count} variables account for 75%+', 
        fontsize=11, color='red', fontweight='bold', va='bottom')

plt.tight_layout()
plt.savefig('변수중요도_통합모델.png', dpi=300, bbox_inches='tight')
print("\n✅ Graph saved: 변수중요도_통합모델.png")

# 상위 변수 누적 비율
print("\n📈 Cumulative Feature Importance:")
cumsum_pct = 0
for i, (idx, row) in enumerate(feature_importance_df.sort_values('Importance', ascending=False).iterrows()):
    cumsum_pct += row['Importance']
    print(f"  Top {i+1}: {cumsum_pct*100:.1f}%")
    if cumsum_pct >= 0.75:
        print(f"  → Top {i+1} variables account for 75%")
        break

plt.show()
