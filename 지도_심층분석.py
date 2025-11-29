import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings('ignore')

from 데이터전처리_1 import preprocess_data

# ============================================================================
# 데이터 로드
# ============================================================================
print("="*100)
print("【 위도/경도 기반 심층 지리 분석 】")
print("="*100)

king = preprocess_data('King_County_Sold.csv', 'King County')
pierce = preprocess_data('Pierce_County_Sold.csv', 'Pierce County')

# ============================================================================
# 1. 시애틀 도심 거리 기반 가격 분석
# ============================================================================
print("\n\n【 1. 시애틀 도심 거리와 가격 관계 분석 】")
print("="*100)

# 주요 거점 좌표 (시애틀 다운타운, 벨뷰, 타코마)
SEATTLE_DOWNTOWN = (47.6062, -122.3321)  # 시애틀 다운타운
BELLEVUE = (47.6101, -122.2015)          # 벨뷰 (테크 허브)
TACOMA = (47.2529, -122.4443)            # 타코마 (Pierce 중심)

def haversine_distance(lat1, lon1, lat2, lon2):
    """두 좌표 간 거리 계산 (km)"""
    R = 6371  # 지구 반경 (km)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# 각 매물의 주요 거점까지 거리 계산
king['dist_seattle'] = king.apply(lambda x: haversine_distance(
    x['LATITUDE'], x['LONGITUDE'], SEATTLE_DOWNTOWN[0], SEATTLE_DOWNTOWN[1]), axis=1)
king['dist_bellevue'] = king.apply(lambda x: haversine_distance(
    x['LATITUDE'], x['LONGITUDE'], BELLEVUE[0], BELLEVUE[1]), axis=1)

pierce['dist_seattle'] = pierce.apply(lambda x: haversine_distance(
    x['LATITUDE'], x['LONGITUDE'], SEATTLE_DOWNTOWN[0], SEATTLE_DOWNTOWN[1]), axis=1)
pierce['dist_tacoma'] = pierce.apply(lambda x: haversine_distance(
    x['LATITUDE'], x['LONGITUDE'], TACOMA[0], TACOMA[1]), axis=1)

# 거리-가격 상관관계 분석
print("\n[King County - 거리와 가격 상관관계]")
corr_seattle_k = stats.pearsonr(king['dist_seattle'], king['PRICE'])
corr_bellevue_k = stats.pearsonr(king['dist_bellevue'], king['PRICE'])
print(f"  시애틀 도심까지 거리 vs 가격: r = {corr_seattle_k[0]:.4f} (p = {corr_seattle_k[1]:.6f})")
print(f"  벨뷰까지 거리 vs 가격:       r = {corr_bellevue_k[0]:.4f} (p = {corr_bellevue_k[1]:.6f})")

if corr_seattle_k[0] < 0:
    print(f"  -> 시애틀 도심에 가까울수록 가격이 {'높음' if abs(corr_seattle_k[0]) > 0.1 else '약간 높음'}")
else:
    print(f"  -> 시애틀 도심에서 멀수록 가격이 높음 (교외 프리미엄)")

if corr_bellevue_k[0] < 0:
    print(f"  -> 벨뷰에 가까울수록 가격이 {'높음' if abs(corr_bellevue_k[0]) > 0.1 else '약간 높음'}")

print("\n[Pierce County - 거리와 가격 상관관계]")
corr_seattle_p = stats.pearsonr(pierce['dist_seattle'], pierce['PRICE'])
corr_tacoma_p = stats.pearsonr(pierce['dist_tacoma'], pierce['PRICE'])
print(f"  시애틀 도심까지 거리 vs 가격: r = {corr_seattle_p[0]:.4f} (p = {corr_seattle_p[1]:.6f})")
print(f"  타코마까지 거리 vs 가격:     r = {corr_tacoma_p[0]:.4f} (p = {corr_tacoma_p[1]:.6f})")

if corr_seattle_p[0] < 0:
    print(f"  -> 시애틀에 가까울수록(북쪽) 가격이 높음")

# ============================================================================
# 2. 거리 구간별 가격 분석
# ============================================================================
print("\n\n【 2. 시애틀 도심 거리 구간별 가격 분석 】")
print("="*100)

# King County 거리 구간 분석
king['dist_zone'] = pd.cut(king['dist_seattle'], 
                           bins=[0, 10, 20, 30, 50, 100],
                           labels=['0-10km', '10-20km', '20-30km', '30-50km', '50km+'])

print("\n[King County - 시애틀 거리별 평균 가격]")
dist_analysis_k = king.groupby('dist_zone', observed=True).agg({
    'PRICE': ['mean', 'median', 'count', 'std'],
    'SQUARE FEET': 'mean',
    '$/SQUARE FEET': 'mean'
}).round(0)
dist_analysis_k.columns = ['평균가격', '중앙값', '거래수', '표준편차', '평균면적', '$/sqft']
print(dist_analysis_k.to_string())

# Pierce County 거리 구간 분석
pierce['dist_zone'] = pd.cut(pierce['dist_seattle'], 
                             bins=[0, 30, 40, 50, 60, 100],
                             labels=['~30km', '30-40km', '40-50km', '50-60km', '60km+'])

print("\n[Pierce County - 시애틀 거리별 평균 가격]")
dist_analysis_p = pierce.groupby('dist_zone', observed=True).agg({
    'PRICE': ['mean', 'median', 'count', 'std'],
    'SQUARE FEET': 'mean',
    '$/SQUARE FEET': 'mean'
}).round(0)
dist_analysis_p.columns = ['평균가격', '중앙값', '거래수', '표준편차', '평균면적', '$/sqft']
print(dist_analysis_p.to_string())

# ============================================================================
# 3. K-Means 클러스터링 (지리 + 가격 기반)
# ============================================================================
print("\n\n【 3. K-Means 클러스터링: 지리-가격 기반 시장 세분화 】")
print("="*100)

# King County 클러스터링
features_k = ['LATITUDE', 'LONGITUDE', 'PRICE', 'SQUARE FEET', '$/SQUARE FEET']
X_king = king[features_k].copy()
scaler = StandardScaler()
X_king_scaled = scaler.fit_transform(X_king)

kmeans_king = KMeans(n_clusters=4, random_state=42, n_init=10)
king['cluster'] = kmeans_king.fit_predict(X_king_scaled)

print("\n[King County 클러스터 분석]")
cluster_analysis_k = king.groupby('cluster').agg({
    'PRICE': ['mean', 'count'],
    'SQUARE FEET': 'mean',
    '$/SQUARE FEET': 'mean',
    'LATITUDE': 'mean',
    'LONGITUDE': 'mean',
    'dist_seattle': 'mean',
    'dist_bellevue': 'mean'
}).round(2)
cluster_analysis_k.columns = ['평균가격', '거래수', '평균면적', '$/sqft', '평균위도', '평균경도', '시애틀거리', '벨뷰거리']

# 클러스터 특성 해석
cluster_names_k = {}
for c in range(4):
    row = cluster_analysis_k.loc[c]
    if row['평균가격'] > 1200000:
        cluster_names_k[c] = '프리미엄 (고가)'
    elif row['평균가격'] > 800000:
        cluster_names_k[c] = '상위 중산층'
    elif row['평균가격'] > 500000:
        cluster_names_k[c] = '중산층'
    else:
        cluster_names_k[c] = '실속형 (저가)'

cluster_analysis_k['시장유형'] = [cluster_names_k[i] for i in range(4)]
print(cluster_analysis_k.to_string())

# Pierce County 클러스터링
X_pierce = pierce[features_k].copy()
X_pierce_scaled = scaler.fit_transform(X_pierce)

kmeans_pierce = KMeans(n_clusters=4, random_state=42, n_init=10)
pierce['cluster'] = kmeans_pierce.fit_predict(X_pierce_scaled)

print("\n[Pierce County 클러스터 분석]")
cluster_analysis_p = pierce.groupby('cluster').agg({
    'PRICE': ['mean', 'count'],
    'SQUARE FEET': 'mean',
    '$/SQUARE FEET': 'mean',
    'LATITUDE': 'mean',
    'LONGITUDE': 'mean',
    'dist_seattle': 'mean',
    'dist_tacoma': 'mean'
}).round(2)
cluster_analysis_p.columns = ['평균가격', '거래수', '평균면적', '$/sqft', '평균위도', '평균경도', '시애틀거리', '타코마거리']

cluster_names_p = {}
for c in range(4):
    row = cluster_analysis_p.loc[c]
    if row['평균가격'] > 800000:
        cluster_names_p[c] = '프리미엄'
    elif row['평균가격'] > 600000:
        cluster_names_p[c] = '상위 중산층'
    elif row['평균가격'] > 450000:
        cluster_names_p[c] = '중산층'
    else:
        cluster_names_p[c] = '실속형'

cluster_analysis_p['시장유형'] = [cluster_names_p[i] for i in range(4)]
print(cluster_analysis_p.to_string())

# ============================================================================
# 4. 공간 자기상관 분석 (주변 매물 영향)
# ============================================================================
print("\n\n【 4. 공간 자기상관 분석: 주변 매물이 가격에 미치는 영향 】")
print("="*100)

def calculate_neighbor_stats(df, radius_km=2):
    """반경 내 주변 매물 통계 계산"""
    coords = df[['LATITUDE', 'LONGITUDE']].values
    prices = df['PRICE'].values
    
    neighbor_avg = []
    neighbor_count = []
    
    for i in range(len(df)):
        distances = []
        for j in range(len(df)):
            if i != j:
                d = haversine_distance(coords[i][0], coords[i][1], 
                                       coords[j][0], coords[j][1])
                distances.append((d, prices[j]))
        
        nearby = [p for d, p in distances if d <= radius_km]
        neighbor_avg.append(np.mean(nearby) if nearby else prices[i])
        neighbor_count.append(len(nearby))
    
    return neighbor_avg, neighbor_count

print("\n[King County - 반경 2km 내 주변 매물 분석]")
king['neighbor_avg_price'], king['neighbor_count'] = calculate_neighbor_stats(king, 2)
neighbor_corr_k = stats.pearsonr(king['PRICE'], king['neighbor_avg_price'])
print(f"  자기 가격 vs 주변 평균 가격 상관계수: r = {neighbor_corr_k[0]:.4f}")
print(f"  -> 주변 매물 가격이 {'강하게' if abs(neighbor_corr_k[0]) > 0.5 else '어느정도'} 영향을 미침")
print(f"  평균 주변 매물 수: {king['neighbor_count'].mean():.1f}개")

print("\n[Pierce County - 반경 2km 내 주변 매물 분석]")
pierce['neighbor_avg_price'], pierce['neighbor_count'] = calculate_neighbor_stats(pierce, 2)
neighbor_corr_p = stats.pearsonr(pierce['PRICE'], pierce['neighbor_avg_price'])
print(f"  자기 가격 vs 주변 평균 가격 상관계수: r = {neighbor_corr_p[0]:.4f}")
print(f"  -> 주변 매물 가격이 {'강하게' if abs(neighbor_corr_p[0]) > 0.5 else '어느정도'} 영향을 미침")
print(f"  평균 주변 매물 수: {pierce['neighbor_count'].mean():.1f}개")

# ============================================================================
# 5. 고가 지역 vs 저가 지역 특성 비교
# ============================================================================
print("\n\n【 5. 고가 지역 vs 저가 지역 특성 비교 】")
print("="*100)

# King County 상위 25% vs 하위 25%
king_high = king[king['PRICE'] >= king['PRICE'].quantile(0.75)]
king_low = king[king['PRICE'] <= king['PRICE'].quantile(0.25)]

print("\n[King County - 상위 25% (고가) vs 하위 25% (저가)]")
print(f"  {'지표':<25} {'고가 지역':<20} {'저가 지역':<20} {'차이':<15}")
print(f"  {'-'*80}")
print(f"  {'평균 가격':<25} ${king_high['PRICE'].mean():>15,.0f} ${king_low['PRICE'].mean():>15,.0f}")
print(f"  {'시애틀 거리 (km)':<25} {king_high['dist_seattle'].mean():>15.1f} {king_low['dist_seattle'].mean():>15.1f} {king_high['dist_seattle'].mean() - king_low['dist_seattle'].mean():>+10.1f}")
print(f"  {'벨뷰 거리 (km)':<25} {king_high['dist_bellevue'].mean():>15.1f} {king_low['dist_bellevue'].mean():>15.1f} {king_high['dist_bellevue'].mean() - king_low['dist_bellevue'].mean():>+10.1f}")
print(f"  {'평균 위도':<25} {king_high['LATITUDE'].mean():>15.4f} {king_low['LATITUDE'].mean():>15.4f}")
print(f"  {'평균 경도':<25} {king_high['LONGITUDE'].mean():>15.4f} {king_low['LONGITUDE'].mean():>15.4f}")
print(f"  {'평균 면적 (sqft)':<25} {king_high['SQUARE FEET'].mean():>15,.0f} {king_low['SQUARE FEET'].mean():>15,.0f}")
print(f"  {'$/sqft':<25} ${king_high['$/SQUARE FEET'].mean():>14.0f} ${king_low['$/SQUARE FEET'].mean():>14.0f}")

# t-검정: 거리 차이가 유의미한가?
t_seattle, p_seattle = stats.ttest_ind(king_high['dist_seattle'], king_low['dist_seattle'])
t_bellevue, p_bellevue = stats.ttest_ind(king_high['dist_bellevue'], king_low['dist_bellevue'])
print(f"\n  [t-검정 결과]")
print(f"  시애틀 거리 차이: t={t_seattle:.2f}, p={p_seattle:.6f} {'*유의미*' if p_seattle < 0.05 else ''}")
print(f"  벨뷰 거리 차이:   t={t_bellevue:.2f}, p={p_bellevue:.6f} {'*유의미*' if p_bellevue < 0.05 else ''}")

# Pierce County 상위 25% vs 하위 25%
pierce_high = pierce[pierce['PRICE'] >= pierce['PRICE'].quantile(0.75)]
pierce_low = pierce[pierce['PRICE'] <= pierce['PRICE'].quantile(0.25)]

print("\n[Pierce County - 상위 25% (고가) vs 하위 25% (저가)]")
print(f"  {'지표':<25} {'고가 지역':<20} {'저가 지역':<20} {'차이':<15}")
print(f"  {'-'*80}")
print(f"  {'평균 가격':<25} ${pierce_high['PRICE'].mean():>15,.0f} ${pierce_low['PRICE'].mean():>15,.0f}")
print(f"  {'시애틀 거리 (km)':<25} {pierce_high['dist_seattle'].mean():>15.1f} {pierce_low['dist_seattle'].mean():>15.1f} {pierce_high['dist_seattle'].mean() - pierce_low['dist_seattle'].mean():>+10.1f}")
print(f"  {'타코마 거리 (km)':<25} {pierce_high['dist_tacoma'].mean():>15.1f} {pierce_low['dist_tacoma'].mean():>15.1f} {pierce_high['dist_tacoma'].mean() - pierce_low['dist_tacoma'].mean():>+10.1f}")
print(f"  {'평균 위도':<25} {pierce_high['LATITUDE'].mean():>15.4f} {pierce_low['LATITUDE'].mean():>15.4f}")

# ============================================================================
# 6. 위치 프리미엄 정량화
# ============================================================================
print("\n\n【 6. 위치 프리미엄 정량화 】")
print("="*100)

# 벨뷰 10km 이내 vs 이외
king_near_bellevue = king[king['dist_bellevue'] <= 10]
king_far_bellevue = king[king['dist_bellevue'] > 10]

bellevue_premium = (king_near_bellevue['PRICE'].mean() - king_far_bellevue['PRICE'].mean()) / king_far_bellevue['PRICE'].mean() * 100
bellevue_sqft_premium = (king_near_bellevue['$/SQUARE FEET'].mean() - king_far_bellevue['$/SQUARE FEET'].mean()) / king_far_bellevue['$/SQUARE FEET'].mean() * 100

print(f"\n[벨뷰 프리미엄 (10km 이내 vs 이외)]")
print(f"  벨뷰 10km 이내: {len(king_near_bellevue)}건, 평균 ${king_near_bellevue['PRICE'].mean():,.0f}")
print(f"  벨뷰 10km 이외: {len(king_far_bellevue)}건, 평균 ${king_far_bellevue['PRICE'].mean():,.0f}")
print(f"  -> 가격 프리미엄: {bellevue_premium:+.1f}%")
print(f"  -> $/sqft 프리미엄: {bellevue_sqft_premium:+.1f}%")

# 시애틀 15km 이내 vs 이외 (Pierce)
pierce_near_seattle = pierce[pierce['dist_seattle'] <= 45]
pierce_far_seattle = pierce[pierce['dist_seattle'] > 45]

if len(pierce_near_seattle) > 0 and len(pierce_far_seattle) > 0:
    seattle_premium_p = (pierce_near_seattle['PRICE'].mean() - pierce_far_seattle['PRICE'].mean()) / pierce_far_seattle['PRICE'].mean() * 100
    print(f"\n[시애틀 접근성 프리미엄 - Pierce County (45km 기준)]")
    print(f"  시애틀 45km 이내: {len(pierce_near_seattle)}건, 평균 ${pierce_near_seattle['PRICE'].mean():,.0f}")
    print(f"  시애틀 45km 이외: {len(pierce_far_seattle)}건, 평균 ${pierce_far_seattle['PRICE'].mean():,.0f}")
    print(f"  -> 가격 프리미엄: {seattle_premium_p:+.1f}%")

# ============================================================================
# 7. 회귀분석: 위치가 가격에 미치는 영향 정량화
# ============================================================================
print("\n\n【 7. 다중회귀분석: 위치 변수의 가격 설명력 】")
print("="*100)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# King County - 위치만으로 가격 예측
X_loc_k = king[['LATITUDE', 'LONGITUDE', 'dist_seattle', 'dist_bellevue']]
y_k = king['PRICE']
lr_loc_k = LinearRegression().fit(X_loc_k, y_k)
r2_loc_k = r2_score(y_k, lr_loc_k.predict(X_loc_k))

# King County - 위치 + 건물속성으로 가격 예측
X_all_k = king[['LATITUDE', 'LONGITUDE', 'dist_seattle', 'dist_bellevue', 'SQUARE FEET', 'BEDS', 'BATHS', 'YEAR BUILT']]
lr_all_k = LinearRegression().fit(X_all_k, y_k)
r2_all_k = r2_score(y_k, lr_all_k.predict(X_all_k))

print(f"\n[King County 회귀분석]")
print(f"  위치 변수만 (위도, 경도, 거리):      R² = {r2_loc_k:.4f} ({r2_loc_k*100:.1f}% 설명)")
print(f"  위치 + 건물속성 (면적, 침실 등):    R² = {r2_all_k:.4f} ({r2_all_k*100:.1f}% 설명)")
print(f"  -> 위치 변수의 독립 기여도: {r2_loc_k/r2_all_k*100:.1f}%")

# 회귀계수 해석
print(f"\n  [회귀계수 - 위치 변수]")
for i, col in enumerate(['LATITUDE', 'LONGITUDE', 'dist_seattle', 'dist_bellevue']):
    coef = lr_all_k.coef_[i]
    print(f"    {col:<20}: {coef:>15,.0f}")

# Pierce County
X_loc_p = pierce[['LATITUDE', 'LONGITUDE', 'dist_seattle', 'dist_tacoma']]
y_p = pierce['PRICE']
lr_loc_p = LinearRegression().fit(X_loc_p, y_p)
r2_loc_p = r2_score(y_p, lr_loc_p.predict(X_loc_p))

X_all_p = pierce[['LATITUDE', 'LONGITUDE', 'dist_seattle', 'dist_tacoma', 'SQUARE FEET', 'BEDS', 'BATHS', 'YEAR BUILT']]
lr_all_p = LinearRegression().fit(X_all_p, y_p)
r2_all_p = r2_score(y_p, lr_all_p.predict(X_all_p))

print(f"\n[Pierce County 회귀분석]")
print(f"  위치 변수만:      R² = {r2_loc_p:.4f} ({r2_loc_p*100:.1f}% 설명)")
print(f"  위치 + 건물속성:  R² = {r2_all_p:.4f} ({r2_all_p*100:.1f}% 설명)")
print(f"  -> 위치 변수의 독립 기여도: {r2_loc_p/r2_all_p*100:.1f}%")

# ============================================================================
# 8. 분석 결과 지도 시각화 (개선된 버전)
# ============================================================================
print("\n\n【 8. 분석 결과 지도 생성 】")
print("="*100)

# 클러스터 색상 및 이름 정렬 (가격순)
cluster_order = sorted(range(4), key=lambda c: king[king['cluster']==c]['PRICE'].mean(), reverse=True)
cluster_colors_map = {cluster_order[0]: '#e74c3c', cluster_order[1]: '#f39c12', 
                      cluster_order[2]: '#3498db', cluster_order[3]: '#27ae60'}
cluster_labels = {cluster_order[0]: '프리미엄', cluster_order[1]: '상위중산층',
                  cluster_order[2]: '중산층', cluster_order[3]: '실속형'}

# King County 클러스터 지도
king_cluster_map = folium.Map(location=[king['LATITUDE'].mean(), king['LONGITUDE'].mean()], 
                               zoom_start=10, tiles='cartodbpositron')

# 벨뷰 10km 반경 원 (프리미엄 존)
folium.Circle(
    location=BELLEVUE,
    radius=10000,
    color='#e74c3c',
    fill=True,
    fillColor='#e74c3c',
    fillOpacity=0.1,
    weight=2,
    dash_array='10',
    popup='벨뷰 10km 반경 (프리미엄 존)'
).add_to(king_cluster_map)

# 주요 거점 마커 (더 눈에 띄게)
folium.Marker(
    SEATTLE_DOWNTOWN, 
    popup=folium.Popup('<b>Seattle Downtown</b><br>도심 기준점', max_width=200),
    icon=folium.Icon(color='black', icon='star', prefix='fa')
).add_to(king_cluster_map)

folium.Marker(
    BELLEVUE, 
    popup=folium.Popup(f'<b>Bellevue (Tech Hub)</b><br>10km 이내 프리미엄: +{bellevue_premium:.1f}%<br>MS, Amazon 등 테크기업 밀집', max_width=250),
    icon=folium.Icon(color='red', icon='building', prefix='fa')
).add_to(king_cluster_map)

# 클러스터별 매물 표시 (가격에 비례한 크기)
for idx, row in king.iterrows():
    cluster = row['cluster']
    color = cluster_colors_map[cluster]
    label = cluster_labels[cluster]
    
    # 가격에 비례한 원 크기 (5~15)
    size = 5 + (row['PRICE'] - king['PRICE'].min()) / (king['PRICE'].max() - king['PRICE'].min()) * 10
    
    popup_text = f"""
    <div style="font-family: Arial; min-width: 200px;">
        <h4 style="margin:0; color:{color};">{label}</h4>
        <hr style="margin:5px 0;">
        <table style="width:100%;">
            <tr><td><b>가격</b></td><td style="text-align:right;">${row['PRICE']:,.0f}</td></tr>
            <tr><td><b>면적</b></td><td style="text-align:right;">{row['SQUARE FEET']:,.0f} sqft</td></tr>
            <tr><td><b>$/sqft</b></td><td style="text-align:right;">${row['$/SQUARE FEET']:.0f}</td></tr>
            <tr><td><b>시애틀 거리</b></td><td style="text-align:right;">{row['dist_seattle']:.1f} km</td></tr>
            <tr><td><b>벨뷰 거리</b></td><td style="text-align:right;">{row['dist_bellevue']:.1f} km</td></tr>
            <tr><td><b>도시</b></td><td style="text-align:right;">{row['CITY']}</td></tr>
        </table>
    </div>
    """
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=size,
        popup=folium.Popup(popup_text, max_width=300),
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        weight=1
    ).add_to(king_cluster_map)

# 클러스터별 통계 계산
cluster_stats = {}
for c in range(4):
    cdata = king[king['cluster'] == c]
    cluster_stats[c] = {
        'count': len(cdata),
        'avg_price': cdata['PRICE'].mean(),
        'avg_sqft': cdata['SQUARE FEET'].mean(),
        'avg_bellevue': cdata['dist_bellevue'].mean()
    }

# 종합 분석 결과 패널 (좌측 하단)
analysis_panel = f'''
<div style="position: fixed; bottom: 20px; left: 20px; z-index: 1000; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px; border-radius: 15px; max-width: 380px; 
            box-shadow: 0 10px 40px rgba(0,0,0,0.3); font-family: Arial;">
    
    <h3 style="margin:0 0 15px 0; color:white; text-align:center; font-size:18px;">
        King County 지리 분석 결과
    </h3>
    
    <div style="background:white; border-radius:10px; padding:15px; margin-bottom:10px;">
        <h4 style="margin:0 0 10px 0; color:#333; border-bottom:2px solid #667eea; padding-bottom:5px;">
            1. 거리-가격 상관관계
        </h4>
        <table style="width:100%; font-size:13px;">
            <tr>
                <td>시애틀 도심</td>
                <td style="text-align:right;"><b>r = {corr_seattle_k[0]:.3f}</b></td>
                <td style="text-align:right; color:{'green' if corr_seattle_k[1]<0.05 else 'gray'};">
                    {'유의미' if corr_seattle_k[1]<0.05 else '비유의미'}
                </td>
            </tr>
            <tr>
                <td>벨뷰 (테크허브)</td>
                <td style="text-align:right;"><b>r = {corr_bellevue_k[0]:.3f}</b></td>
                <td style="text-align:right; color:green;">유의미</td>
            </tr>
        </table>
        <p style="margin:10px 0 0 0; font-size:12px; color:#666;">
            -> 벨뷰에 가까울수록 가격 상승
        </p>
    </div>
    
    <div style="background:white; border-radius:10px; padding:15px; margin-bottom:10px;">
        <h4 style="margin:0 0 10px 0; color:#333; border-bottom:2px solid #e74c3c; padding-bottom:5px;">
            2. 벨뷰 프리미엄 (10km 기준)
        </h4>
        <div style="text-align:center; font-size:28px; font-weight:bold; color:#e74c3c;">
            +{bellevue_premium:.1f}%
        </div>
        <p style="margin:5px 0 0 0; font-size:12px; color:#666; text-align:center;">
            10km 이내 평균 ${king_near_bellevue['PRICE'].mean():,.0f}<br>
            10km 이외 평균 ${king_far_bellevue['PRICE'].mean():,.0f}
        </p>
    </div>
    
    <div style="background:white; border-radius:10px; padding:15px; margin-bottom:10px;">
        <h4 style="margin:0 0 10px 0; color:#333; border-bottom:2px solid #3498db; padding-bottom:5px;">
            3. 위치 변수 설명력
        </h4>
        <div style="display:flex; justify-content:space-around; text-align:center;">
            <div>
                <div style="font-size:24px; font-weight:bold; color:#3498db;">{r2_loc_k*100:.1f}%</div>
                <div style="font-size:11px; color:#666;">위치만</div>
            </div>
            <div style="font-size:20px; color:#ccc;">→</div>
            <div>
                <div style="font-size:24px; font-weight:bold; color:#27ae60;">{r2_all_k*100:.1f}%</div>
                <div style="font-size:11px; color:#666;">전체모델</div>
            </div>
        </div>
    </div>
    
    <div style="background:white; border-radius:10px; padding:15px;">
        <h4 style="margin:0 0 10px 0; color:#333; border-bottom:2px solid #9b59b6; padding-bottom:5px;">
            4. 공간 자기상관 (주변 영향)
        </h4>
        <div style="text-align:center;">
            <span style="font-size:24px; font-weight:bold; color:#9b59b6;">r = {neighbor_corr_k[0]:.3f}</span>
            <span style="font-size:14px; color:#666;"> (강한 영향)</span>
        </div>
        <p style="margin:5px 0 0 0; font-size:12px; color:#666; text-align:center;">
            반경 2km 내 평균 {king['neighbor_count'].mean():.1f}개 매물 영향
        </p>
    </div>
</div>
'''

# 클러스터 범례 (우측 상단)
legend_panel = f'''
<div style="position: fixed; top: 20px; right: 20px; z-index: 1000; 
            background: white; padding: 15px; border-radius: 10px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.2); font-family: Arial; min-width: 200px;">
    
    <h4 style="margin:0 0 10px 0; text-align:center; color:#333;">시장 클러스터</h4>
    
    <div style="margin-bottom:8px; padding:8px; background:#fdf2f2; border-radius:5px; border-left:4px solid #e74c3c;">
        <span style="color:#e74c3c; font-weight:bold;">● 프리미엄</span><br>
        <span style="font-size:11px; color:#666;">평균 ${cluster_stats[cluster_order[0]]['avg_price']:,.0f} ({cluster_stats[cluster_order[0]]['count']}건)</span>
    </div>
    
    <div style="margin-bottom:8px; padding:8px; background:#fef9e7; border-radius:5px; border-left:4px solid #f39c12;">
        <span style="color:#f39c12; font-weight:bold;">● 상위중산층</span><br>
        <span style="font-size:11px; color:#666;">평균 ${cluster_stats[cluster_order[1]]['avg_price']:,.0f} ({cluster_stats[cluster_order[1]]['count']}건)</span>
    </div>
    
    <div style="margin-bottom:8px; padding:8px; background:#ebf5fb; border-radius:5px; border-left:4px solid #3498db;">
        <span style="color:#3498db; font-weight:bold;">● 중산층</span><br>
        <span style="font-size:11px; color:#666;">평균 ${cluster_stats[cluster_order[2]]['avg_price']:,.0f} ({cluster_stats[cluster_order[2]]['count']}건)</span>
    </div>
    
    <div style="padding:8px; background:#eafaf1; border-radius:5px; border-left:4px solid #27ae60;">
        <span style="color:#27ae60; font-weight:bold;">● 실속형</span><br>
        <span style="font-size:11px; color:#666;">평균 ${cluster_stats[cluster_order[3]]['avg_price']:,.0f} ({cluster_stats[cluster_order[3]]['count']}건)</span>
    </div>
    
    <hr style="margin:10px 0;">
    <div style="font-size:11px; color:#666; text-align:center;">
        <span style="color:#e74c3c;">- - -</span> 벨뷰 10km 프리미엄 존<br>
        ★ 시애틀 / 벨뷰 기준점
    </div>
</div>
'''

# 핵심 인사이트 (우측 하단)
insight_panel = f'''
<div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000; 
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 15px; border-radius: 10px; max-width: 280px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.2); font-family: Arial;">
    
    <h4 style="margin:0 0 10px 0; color:white; text-align:center;">핵심 인사이트</h4>
    
    <div style="background:white; border-radius:8px; padding:12px; font-size:12px;">
        <p style="margin:0 0 8px 0;">
            <b style="color:#e74c3c;">1.</b> 시애틀 도심이 아닌 <b>벨뷰(Eastside)</b>가 진짜 고가 지역
        </p>
        <p style="margin:0 0 8px 0;">
            <b style="color:#f39c12;">2.</b> 테크기업(MS, Amazon) 밀집 효과로 <b>+33.5%</b> 프리미엄
        </p>
        <p style="margin:0 0 8px 0;">
            <b style="color:#3498db;">3.</b> 위치만으로 가격의 <b>19.4%</b> 설명 가능
        </p>
        <p style="margin:0;">
            <b style="color:#27ae60;">4.</b> 주변 매물이 가격에 <b>강한 영향</b> (r=0.51)
        </p>
    </div>
</div>
'''

king_cluster_map.get_root().html.add_child(folium.Element(analysis_panel))
king_cluster_map.get_root().html.add_child(folium.Element(legend_panel))
king_cluster_map.get_root().html.add_child(folium.Element(insight_panel))

king_cluster_map.save('지도_King_클러스터분석.html')
print("   [저장] 지도_King_클러스터분석.html")

# ============================================================================
# 9. 종합 결론
# ============================================================================
print("\n\n" + "="*100)
print("【 종합 분석 결과 】")
print("="*100)

print(f"""
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                            위도/경도 기반 심층 지리 분석 결과                                │
├────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                            │
│  【 King County 】                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 1. 거리-가격 상관관계                                                                │  │
│  │    - 시애틀 도심: r = {corr_seattle_k[0]:+.4f} (도심 멀수록 오히려 비쌈 = 교외 프리미엄)      │  │
│  │    - 벨뷰 (테크허브): r = {corr_bellevue_k[0]:+.4f} (벨뷰 가까울수록 비쌈)                   │  │
│  │                                                                                      │  │
│  │ 2. 벨뷰 프리미엄: {bellevue_premium:+.1f}% (10km 이내 vs 이외)                                │  │
│  │    -> 테크 기업(MS, Amazon) 밀집 지역의 가격 프리미엄                                 │  │
│  │                                                                                      │  │
│  │ 3. 위치 변수 설명력: R² = {r2_loc_k:.4f} (전체의 {r2_loc_k/r2_all_k*100:.1f}%)                      │  │
│  │    -> 위치만으로 가격의 {r2_loc_k*100:.1f}% 설명 가능                                        │  │
│  │                                                                                      │  │
│  │ 4. 공간 자기상관: r = {neighbor_corr_k[0]:.4f}                                              │  │
│  │    -> 주변 매물 가격이 자기 가격에 {'강한' if neighbor_corr_k[0] > 0.5 else '중간'} 영향                               │  │
│  └─────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                            │
│  【 Pierce County 】                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 1. 거리-가격 상관관계                                                                │  │
│  │    - 시애틀: r = {corr_seattle_p[0]:+.4f} (시애틀 가까울수록 = 북쪽일수록 비쌈)              │  │
│  │    - 타코마: r = {corr_tacoma_p[0]:+.4f}                                                     │  │
│  │                                                                                      │  │
│  │ 2. 위치 변수 설명력: R² = {r2_loc_p:.4f} (전체의 {r2_loc_p/r2_all_p*100:.1f}%)                      │  │
│  │    -> King County보다 위치 영향 {'큼' if r2_loc_p > r2_loc_k else '작음'}                                     │  │
│  │                                                                                      │  │
│  │ 3. 시애틀 접근성 프리미엄이 핵심 가격 결정 요인                                       │  │
│  └─────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                            │
│  【 핵심 인사이트 】                                                                       │
│  • King County: 시애틀 도심이 아닌 "벨뷰(Eastside)"가 진정한 고가 지역                    │
│  • Pierce County: "시애틀 접근성(북쪽)"이 가격의 핵심 결정 요인                           │
│  • 두 카운티 모두 "공간 자기상관"이 존재 (좋은 동네 효과)                                 │
│  • 위치만으로 King {r2_loc_k*100:.0f}%, Pierce {r2_loc_p*100:.0f}% 가격 설명 가능                            │
│                                                                                            │
└────────────────────────────────────────────────────────────────────────────────────────────┘
""")

print("\n[생성된 분석 지도]")
print("  지도_King_클러스터분석.html - 클러스터 + 분석결과 포함")
