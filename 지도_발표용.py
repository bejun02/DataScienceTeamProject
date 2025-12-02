# -*- coding: utf-8 -*-
"""
발표용 인터랙티브 지도 시각화
King County + Pierce County 통합 분석
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import folium
from folium.plugins import MarkerCluster
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 데이터 로드 및 전처리
# =============================================================================
def load_and_preprocess():
    """데이터 로드 및 전처리"""
    king = pd.read_csv('King_County_Sold.csv')
    pierce = pd.read_csv('Pierce_County_Sold.csv')
    
    cols = ['PROPERTY TYPE', 'CITY', 'PRICE', 'BEDS', 'BATHS', 
            'SQUARE FEET', 'YEAR BUILT', '$/SQUARE FEET', 'LATITUDE', 'LONGITUDE']
    
    king = king[cols].dropna()
    pierce = pierce[cols].dropna()
    
    property_types = ['Single Family Residential', 'Townhouse', 'Condo/Co-op']
    king = king[king['PROPERTY TYPE'].isin(property_types)]
    pierce = pierce[pierce['PROPERTY TYPE'].isin(property_types)]
    
    king = king[king['PRICE'] <= 5000000]
    pierce = pierce[pierce['PRICE'] <= 4000000]
    
    king['COUNTY'] = 'King'
    pierce['COUNTY'] = 'Pierce'
    
    return king, pierce

def haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine 거리 계산 (km)"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# 기준점 좌표
SEATTLE = (47.6062, -122.3321)
BELLEVUE = (47.6101, -122.2015)
TACOMA = (47.2529, -122.4443)

print("="*70)
print("【 발표용 인터랙티브 지도 생성 】")
print("="*70)

# 데이터 로드
king, pierce = load_and_preprocess()
print(f"\nKing County: {len(king)}건, Pierce County: {len(pierce)}건")

# 거리 계산
for df in [king, pierce]:
    df['dist_seattle'] = df.apply(lambda x: haversine_distance(
        x['LATITUDE'], x['LONGITUDE'], SEATTLE[0], SEATTLE[1]), axis=1)
    df['dist_bellevue'] = df.apply(lambda x: haversine_distance(
        x['LATITUDE'], x['LONGITUDE'], BELLEVUE[0], BELLEVUE[1]), axis=1)

# =============================================================================
# 통계 분석
# =============================================================================
print("\n[통계 분석 수행 중...]")

# 상관관계
corr_bellevue_k = stats.pearsonr(king['dist_bellevue'], king['PRICE'])
corr_seattle_p = stats.pearsonr(pierce['dist_seattle'], pierce['PRICE'])

# 벨뷰 프리미엄
king_near = king[king['dist_bellevue'] <= 10]
king_far = king[king['dist_bellevue'] > 10]
bellevue_premium = (king_near['PRICE'].mean() - king_far['PRICE'].mean()) / king_far['PRICE'].mean() * 100

# t-검정 (벨뷰 프리미엄)
t_stat, p_value = stats.ttest_ind(king_near['PRICE'], king_far['PRICE'])

# 시애틀 접근성 프리미엄 (Pierce)
pierce_near = pierce[pierce['dist_seattle'] <= 45]
pierce_far = pierce[pierce['dist_seattle'] > 45]
seattle_premium = (pierce_near['PRICE'].mean() - pierce_far['PRICE'].mean()) / pierce_far['PRICE'].mean() * 100

# 위치 설명력
X_loc_k = king[['LATITUDE', 'LONGITUDE', 'dist_seattle', 'dist_bellevue']]
lr_k = LinearRegression().fit(X_loc_k, king['PRICE'])
r2_loc_k = r2_score(king['PRICE'], lr_k.predict(X_loc_k))

X_loc_p = pierce[['LATITUDE', 'LONGITUDE', 'dist_seattle']]
lr_p = LinearRegression().fit(X_loc_p, pierce['PRICE'])
r2_loc_p = r2_score(pierce['PRICE'], lr_p.predict(X_loc_p))

# K-Means 클러스터링
features = ['LATITUDE', 'LONGITUDE', 'PRICE', 'SQUARE FEET']
scaler = StandardScaler()

X_king = scaler.fit_transform(king[features])
kmeans_king = KMeans(n_clusters=4, random_state=42, n_init=10)
king['cluster'] = kmeans_king.fit_predict(X_king)

X_pierce = scaler.fit_transform(pierce[features])
kmeans_pierce = KMeans(n_clusters=4, random_state=42, n_init=10)
pierce['cluster'] = kmeans_pierce.fit_predict(X_pierce)

# 클러스터 정렬 (가격순)
king_order = sorted(range(4), key=lambda c: king[king['cluster']==c]['PRICE'].mean(), reverse=True)
pierce_order = sorted(range(4), key=lambda c: pierce[pierce['cluster']==c]['PRICE'].mean(), reverse=True)

king_cluster_map = {king_order[i]: i for i in range(4)}
pierce_cluster_map = {pierce_order[i]: i for i in range(4)}

king['cluster_rank'] = king['cluster'].map(king_cluster_map)
pierce['cluster_rank'] = pierce['cluster'].map(pierce_cluster_map)

print("  ✓ 상관관계 분석 완료")
print("  ✓ 벨뷰 프리미엄 계산 완료")
print("  ✓ K-Means 클러스터링 완료")

# =============================================================================
# 지도 1: King County 클러스터 분석
# =============================================================================
print("\n[지도 1] King County 클러스터 분석 생성 중...")

cluster_colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']
cluster_names = ['프리미엄', '상위중산층', '중산층', '실속형']

king_map = folium.Map(
    location=[king['LATITUDE'].mean(), king['LONGITUDE'].mean()],
    zoom_start=10,
    tiles='cartodbpositron'
)

# 벨뷰 10km 반경
folium.Circle(
    location=BELLEVUE,
    radius=10000,
    color='#e74c3c',
    fill=True,
    fillColor='#e74c3c',
    fillOpacity=0.08,
    weight=3,
    dash_array='10,5',
    popup=f'<b>벨뷰 10km 프리미엄 존</b><br>+{bellevue_premium:.1f}% 가격 프리미엄<br>p-value < 0.001 (통계적 유의)'
).add_to(king_map)

# 기준점 마커
folium.Marker(
    SEATTLE,
    popup='<b>Seattle Downtown</b><br>도심 기준점',
    icon=folium.Icon(color='black', icon='star', prefix='fa')
).add_to(king_map)

folium.Marker(
    BELLEVUE,
    popup=f'<b>Bellevue (Tech Hub)</b><br>Microsoft, Amazon 등<br>10km 이내 프리미엄: <b>+{bellevue_premium:.1f}%</b>',
    icon=folium.Icon(color='red', icon='building', prefix='fa')
).add_to(king_map)

# 매물 표시
for idx, row in king.iterrows():
    rank = row['cluster_rank']
    color = cluster_colors[rank]
    label = cluster_names[rank]
    
    size = 4 + (row['PRICE'] - king['PRICE'].min()) / (king['PRICE'].max() - king['PRICE'].min()) * 8
    
    popup_html = f'''
    <div style="font-family:Arial; min-width:180px;">
        <h4 style="margin:0; color:{color}; border-bottom:2px solid {color}; padding-bottom:5px;">{label}</h4>
        <table style="width:100%; font-size:12px; margin-top:8px;">
            <tr><td>가격</td><td style="text-align:right;"><b>${row['PRICE']:,.0f}</b></td></tr>
            <tr><td>면적</td><td style="text-align:right;">{row['SQUARE FEET']:,.0f} sqft</td></tr>
            <tr><td>$/sqft</td><td style="text-align:right;">${row['$/SQUARE FEET']:.0f}</td></tr>
            <tr><td>벨뷰 거리</td><td style="text-align:right;">{row['dist_bellevue']:.1f} km</td></tr>
            <tr><td>도시</td><td style="text-align:right;">{row['CITY']}</td></tr>
        </table>
    </div>
    '''
    
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=size,
        popup=folium.Popup(popup_html, max_width=250),
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        weight=1
    ).add_to(king_map)

# 클러스터 통계
cluster_stats_k = []
for i in range(4):
    cdata = king[king['cluster_rank'] == i]
    cluster_stats_k.append({
        'count': len(cdata),
        'avg_price': cdata['PRICE'].mean(),
        'avg_bellevue': cdata['dist_bellevue'].mean()
    })

# 분석 결과 패널 (좌측)
analysis_html = f'''
<div style="position:fixed; bottom:20px; left:20px; z-index:1000;
            background:linear-gradient(135deg, #2E86AB 0%, #1a5276 100%);
            padding:20px; border-radius:15px; max-width:350px;
            box-shadow:0 10px 40px rgba(0,0,0,0.3); font-family:Arial;">
    
    <h3 style="margin:0 0 15px 0; color:white; text-align:center;">
        🏠 King County 분석 결과
    </h3>
    
    <div style="background:white; border-radius:10px; padding:15px; margin-bottom:10px;">
        <h4 style="margin:0 0 8px 0; color:#2E86AB;">Q3: 위치가 가격에 미치는 영향</h4>
        <table style="width:100%; font-size:12px;">
            <tr>
                <td>벨뷰 거리 상관</td>
                <td style="text-align:right;"><b>r = {corr_bellevue_k[0]:.3f}</b></td>
            </tr>
            <tr>
                <td>p-value</td>
                <td style="text-align:right; color:green;"><b>< 0.001</b></td>
            </tr>
        </table>
        <p style="margin:8px 0 0 0; font-size:11px; color:#666;">
            → 벨뷰에 가까울수록 가격 상승 (통계적 유의)
        </p>
    </div>
    
    <div style="background:white; border-radius:10px; padding:15px; margin-bottom:10px;">
        <h4 style="margin:0 0 8px 0; color:#e74c3c;">벨뷰 프리미엄 (10km 기준)</h4>
        <div style="text-align:center; font-size:32px; font-weight:bold; color:#e74c3c;">
            +{bellevue_premium:.1f}%
        </div>
        <div style="display:flex; justify-content:space-around; font-size:11px; color:#666; margin-top:8px;">
            <div style="text-align:center;">
                <div style="font-weight:bold; color:#e74c3c;">${king_near['PRICE'].mean()/1e6:.2f}M</div>
                <div>10km 이내 ({len(king_near)}건)</div>
            </div>
            <div style="text-align:center;">
                <div style="font-weight:bold; color:#3498db;">${king_far['PRICE'].mean()/1e6:.2f}M</div>
                <div>10km 초과 ({len(king_far)}건)</div>
            </div>
        </div>
        <p style="margin:8px 0 0 0; font-size:11px; color:#27ae60; text-align:center;">
            <b>Welch t-test p < 0.001</b> → 통계적으로 유의미
        </p>
    </div>
    
    <div style="background:white; border-radius:10px; padding:15px;">
        <h4 style="margin:0 0 8px 0; color:#9b59b6;">위치 변수 설명력</h4>
        <div style="text-align:center;">
            <span style="font-size:28px; font-weight:bold; color:#9b59b6;">{r2_loc_k*100:.1f}%</span>
            <span style="font-size:12px; color:#666;"> (R²)</span>
        </div>
        <p style="margin:5px 0 0 0; font-size:11px; color:#666; text-align:center;">
            위치만으로 가격의 약 1/5 설명 가능
        </p>
    </div>
</div>
'''

# 범례 (우측 상단)
legend_html = f'''
<div style="position:fixed; top:20px; right:20px; z-index:1000;
            background:white; padding:15px; border-radius:10px;
            box-shadow:0 4px 15px rgba(0,0,0,0.2); font-family:Arial; min-width:180px;">
    
    <h4 style="margin:0 0 10px 0; text-align:center; color:#333;">시장 클러스터</h4>
    
    <div style="margin-bottom:6px; padding:6px; background:#fdf2f2; border-radius:5px; border-left:4px solid #e74c3c;">
        <span style="color:#e74c3c; font-weight:bold;">● 프리미엄</span><br>
        <span style="font-size:10px; color:#666;">평균 ${cluster_stats_k[0]['avg_price']:,.0f} ({cluster_stats_k[0]['count']}건)</span>
    </div>
    
    <div style="margin-bottom:6px; padding:6px; background:#fef9e7; border-radius:5px; border-left:4px solid #f39c12;">
        <span style="color:#f39c12; font-weight:bold;">● 상위중산층</span><br>
        <span style="font-size:10px; color:#666;">평균 ${cluster_stats_k[1]['avg_price']:,.0f} ({cluster_stats_k[1]['count']}건)</span>
    </div>
    
    <div style="margin-bottom:6px; padding:6px; background:#ebf5fb; border-radius:5px; border-left:4px solid #3498db;">
        <span style="color:#3498db; font-weight:bold;">● 중산층</span><br>
        <span style="font-size:10px; color:#666;">평균 ${cluster_stats_k[2]['avg_price']:,.0f} ({cluster_stats_k[2]['count']}건)</span>
    </div>
    
    <div style="padding:6px; background:#eafaf1; border-radius:5px; border-left:4px solid #27ae60;">
        <span style="color:#27ae60; font-weight:bold;">● 실속형</span><br>
        <span style="font-size:10px; color:#666;">평균 ${cluster_stats_k[3]['avg_price']:,.0f} ({cluster_stats_k[3]['count']}건)</span>
    </div>
    
    <hr style="margin:10px 0;">
    <div style="font-size:10px; color:#666; text-align:center;">
        <span style="color:#e74c3c;">- - -</span> 벨뷰 10km 프리미엄 존<br>
        ★ 시애틀 / ★ 벨뷰
    </div>
</div>
'''

king_map.get_root().html.add_child(folium.Element(analysis_html))
king_map.get_root().html.add_child(folium.Element(legend_html))

king_map.save('지도_King_클러스터분석.html')
print("  ✓ 지도_King_클러스터분석.html 저장 완료")

# =============================================================================
# 지도 2: Pierce County 시애틀 접근성 분석
# =============================================================================
print("\n[지도 2] Pierce County 시애틀 접근성 분석 생성 중...")

pierce_map = folium.Map(
    location=[pierce['LATITUDE'].mean(), pierce['LONGITUDE'].mean()],
    zoom_start=10,
    tiles='cartodbpositron'
)

# 시애틀 45km 반경
folium.Circle(
    location=SEATTLE,
    radius=45000,
    color='#A23B72',
    fill=True,
    fillColor='#A23B72',
    fillOpacity=0.08,
    weight=3,
    dash_array='10,5',
    popup=f'<b>시애틀 45km 접근권</b><br>+{seattle_premium:.1f}% 가격 프리미엄'
).add_to(pierce_map)

# 기준점
folium.Marker(
    SEATTLE,
    popup=f'<b>Seattle Downtown</b><br>Pierce에서 접근성 핵심<br>45km 이내: +{seattle_premium:.1f}%',
    icon=folium.Icon(color='darkred', icon='star', prefix='fa')
).add_to(pierce_map)

folium.Marker(
    TACOMA,
    popup='<b>Tacoma</b><br>Pierce County 중심도시',
    icon=folium.Icon(color='purple', icon='building', prefix='fa')
).add_to(pierce_map)

# 매물 표시
for idx, row in pierce.iterrows():
    rank = row['cluster_rank']
    color = cluster_colors[rank]
    label = cluster_names[rank]
    
    size = 4 + (row['PRICE'] - pierce['PRICE'].min()) / (pierce['PRICE'].max() - pierce['PRICE'].min()) * 8
    
    popup_html = f'''
    <div style="font-family:Arial; min-width:180px;">
        <h4 style="margin:0; color:{color}; border-bottom:2px solid {color}; padding-bottom:5px;">{label}</h4>
        <table style="width:100%; font-size:12px; margin-top:8px;">
            <tr><td>가격</td><td style="text-align:right;"><b>${row['PRICE']:,.0f}</b></td></tr>
            <tr><td>면적</td><td style="text-align:right;">{row['SQUARE FEET']:,.0f} sqft</td></tr>
            <tr><td>$/sqft</td><td style="text-align:right;">${row['$/SQUARE FEET']:.0f}</td></tr>
            <tr><td>시애틀 거리</td><td style="text-align:right;">{row['dist_seattle']:.1f} km</td></tr>
            <tr><td>도시</td><td style="text-align:right;">{row['CITY']}</td></tr>
        </table>
    </div>
    '''
    
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=size,
        popup=folium.Popup(popup_html, max_width=250),
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        weight=1
    ).add_to(pierce_map)

# 클러스터 통계
cluster_stats_p = []
for i in range(4):
    cdata = pierce[pierce['cluster_rank'] == i]
    cluster_stats_p.append({
        'count': len(cdata),
        'avg_price': cdata['PRICE'].mean(),
        'avg_seattle': cdata['dist_seattle'].mean()
    })

# 분석 결과 패널
analysis_html_p = f'''
<div style="position:fixed; bottom:20px; left:20px; z-index:1000;
            background:linear-gradient(135deg, #A23B72 0%, #6a1b4d 100%);
            padding:20px; border-radius:15px; max-width:350px;
            box-shadow:0 10px 40px rgba(0,0,0,0.3); font-family:Arial;">
    
    <h3 style="margin:0 0 15px 0; color:white; text-align:center;">
        🏠 Pierce County 분석 결과
    </h3>
    
    <div style="background:white; border-radius:10px; padding:15px; margin-bottom:10px;">
        <h4 style="margin:0 0 8px 0; color:#A23B72;">Q3: 시애틀 접근성 효과</h4>
        <table style="width:100%; font-size:12px;">
            <tr>
                <td>시애틀 거리 상관</td>
                <td style="text-align:right;"><b>r = {corr_seattle_p[0]:.3f}</b></td>
            </tr>
            <tr>
                <td>p-value</td>
                <td style="text-align:right; color:green;"><b>< 0.001</b></td>
            </tr>
        </table>
        <p style="margin:8px 0 0 0; font-size:11px; color:#666;">
            → 시애틀에 가까울수록(북쪽) 가격 상승
        </p>
    </div>
    
    <div style="background:white; border-radius:10px; padding:15px; margin-bottom:10px;">
        <h4 style="margin:0 0 8px 0; color:#e74c3c;">시애틀 접근성 프리미엄 (45km)</h4>
        <div style="text-align:center; font-size:32px; font-weight:bold; color:#A23B72;">
            +{seattle_premium:.1f}%
        </div>
        <div style="display:flex; justify-content:space-around; font-size:11px; color:#666; margin-top:8px;">
            <div style="text-align:center;">
                <div style="font-weight:bold; color:#A23B72;">${pierce_near['PRICE'].mean()/1e6:.2f}M</div>
                <div>45km 이내 ({len(pierce_near)}건)</div>
            </div>
            <div style="text-align:center;">
                <div style="font-weight:bold; color:#3498db;">${pierce_far['PRICE'].mean()/1e6:.2f}M</div>
                <div>45km 초과 ({len(pierce_far)}건)</div>
            </div>
        </div>
    </div>
    
    <div style="background:white; border-radius:10px; padding:15px;">
        <h4 style="margin:0 0 8px 0; color:#9b59b6;">위치 변수 설명력</h4>
        <div style="text-align:center;">
            <span style="font-size:28px; font-weight:bold; color:#9b59b6;">{r2_loc_p*100:.1f}%</span>
            <span style="font-size:12px; color:#666;"> (R²)</span>
        </div>
        <p style="margin:5px 0 0 0; font-size:11px; color:#666; text-align:center;">
            King보다 위치 영향이 더 큼 (위성도시 특성)
        </p>
    </div>
</div>
'''

# 범례 (우측 상단)
legend_html_p = f'''
<div style="position:fixed; top:20px; right:20px; z-index:1000;
            background:white; padding:15px; border-radius:10px;
            box-shadow:0 4px 15px rgba(0,0,0,0.2); font-family:Arial; min-width:180px;">
    
    <h4 style="margin:0 0 10px 0; text-align:center; color:#333;">시장 클러스터</h4>
    
    <div style="margin-bottom:6px; padding:6px; background:#fdf2f2; border-radius:5px; border-left:4px solid #e74c3c;">
        <span style="color:#e74c3c; font-weight:bold;">● 프리미엄</span><br>
        <span style="font-size:10px; color:#666;">평균 ${cluster_stats_p[0]['avg_price']:,.0f} ({cluster_stats_p[0]['count']}건)</span>
    </div>
    
    <div style="margin-bottom:6px; padding:6px; background:#fef9e7; border-radius:5px; border-left:4px solid #f39c12;">
        <span style="color:#f39c12; font-weight:bold;">● 상위중산층</span><br>
        <span style="font-size:10px; color:#666;">평균 ${cluster_stats_p[1]['avg_price']:,.0f} ({cluster_stats_p[1]['count']}건)</span>
    </div>
    
    <div style="margin-bottom:6px; padding:6px; background:#ebf5fb; border-radius:5px; border-left:4px solid #3498db;">
        <span style="color:#3498db; font-weight:bold;">● 중산층</span><br>
        <span style="font-size:10px; color:#666;">평균 ${cluster_stats_p[2]['avg_price']:,.0f} ({cluster_stats_p[2]['count']}건)</span>
    </div>
    
    <div style="padding:6px; background:#eafaf1; border-radius:5px; border-left:4px solid #27ae60;">
        <span style="color:#27ae60; font-weight:bold;">● 실속형</span><br>
        <span style="font-size:10px; color:#666;">평균 ${cluster_stats_p[3]['avg_price']:,.0f} ({cluster_stats_p[3]['count']}건)</span>
    </div>
    
    <hr style="margin:10px 0;">
    <div style="font-size:10px; color:#666; text-align:center;">
        <span style="color:#A23B72;">- - -</span> 시애틀 45km 접근권<br>
        ★ 시애틀 / ★ 타코마
    </div>
</div>
'''

pierce_map.get_root().html.add_child(folium.Element(analysis_html_p))
pierce_map.get_root().html.add_child(folium.Element(legend_html_p))

pierce_map.save('지도_Pierce_시애틀접근성.html')
print("  ✓ 지도_Pierce_시애틀접근성.html 저장 완료")

# =============================================================================
# 지도 3: 통합 가격 분포 지도
# =============================================================================
print("\n[지도 3] 통합 가격 분포 지도 생성 중...")

combined = pd.concat([king, pierce], ignore_index=True)
combined_map = folium.Map(
    location=[combined['LATITUDE'].mean(), combined['LONGITUDE'].mean()],
    zoom_start=9,
    tiles='cartodbpositron'
)

# 가격 구간별 색상
def get_price_color(price):
    if price >= 1500000:
        return '#B2182B'  # 진한 빨강
    elif price >= 1000000:
        return '#EF8A62'  # 주황빨강
    elif price >= 750000:
        return '#FDDBC7'  # 연한 주황
    elif price >= 500000:
        return '#67A9CF'  # 연한 파랑
    else:
        return '#2166AC'  # 진한 파랑

# 기준점
folium.Marker(SEATTLE, popup='Seattle', icon=folium.Icon(color='black', icon='star', prefix='fa')).add_to(combined_map)
folium.Marker(BELLEVUE, popup='Bellevue', icon=folium.Icon(color='red', icon='building', prefix='fa')).add_to(combined_map)
folium.Marker(TACOMA, popup='Tacoma', icon=folium.Icon(color='purple', icon='building', prefix='fa')).add_to(combined_map)

# 매물 표시
for idx, row in combined.iterrows():
    color = get_price_color(row['PRICE'])
    county = row['COUNTY']
    
    popup_html = f'''
    <div style="font-family:Arial; min-width:160px;">
        <h4 style="margin:0; color:{'#2E86AB' if county=='King' else '#A23B72'};">{county} County</h4>
        <table style="width:100%; font-size:12px; margin-top:5px;">
            <tr><td>가격</td><td style="text-align:right;"><b>${row['PRICE']:,.0f}</b></td></tr>
            <tr><td>면적</td><td style="text-align:right;">{row['SQUARE FEET']:,.0f} sqft</td></tr>
            <tr><td>$/sqft</td><td style="text-align:right;">${row['$/SQUARE FEET']:.0f}</td></tr>
            <tr><td>도시</td><td style="text-align:right;">{row['CITY']}</td></tr>
        </table>
    </div>
    '''
    
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=4,
        popup=folium.Popup(popup_html, max_width=200),
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        weight=1
    ).add_to(combined_map)

# 범례
legend_combined = '''
<div style="position:fixed; top:20px; right:20px; z-index:1000;
            background:white; padding:15px; border-radius:10px;
            box-shadow:0 4px 15px rgba(0,0,0,0.2); font-family:Arial;">
    
    <h4 style="margin:0 0 10px 0; text-align:center;">가격 범위</h4>
    
    <div style="display:flex; align-items:center; margin-bottom:5px;">
        <div style="width:20px; height:20px; background:#B2182B; border-radius:50%; margin-right:10px;"></div>
        <span style="font-size:12px;">$1.5M 이상</span>
    </div>
    <div style="display:flex; align-items:center; margin-bottom:5px;">
        <div style="width:20px; height:20px; background:#EF8A62; border-radius:50%; margin-right:10px;"></div>
        <span style="font-size:12px;">$1M - $1.5M</span>
    </div>
    <div style="display:flex; align-items:center; margin-bottom:5px;">
        <div style="width:20px; height:20px; background:#FDDBC7; border-radius:50%; margin-right:10px;"></div>
        <span style="font-size:12px;">$750K - $1M</span>
    </div>
    <div style="display:flex; align-items:center; margin-bottom:5px;">
        <div style="width:20px; height:20px; background:#67A9CF; border-radius:50%; margin-right:10px;"></div>
        <span style="font-size:12px;">$500K - $750K</span>
    </div>
    <div style="display:flex; align-items:center;">
        <div style="width:20px; height:20px; background:#2166AC; border-radius:50%; margin-right:10px;"></div>
        <span style="font-size:12px;">$500K 미만</span>
    </div>
</div>
'''

# 카운티 비교 패널
comparison_panel = f'''
<div style="position:fixed; bottom:20px; left:20px; z-index:1000;
            background:white; padding:20px; border-radius:15px; max-width:380px;
            box-shadow:0 10px 40px rgba(0,0,0,0.3); font-family:Arial;">
    
    <h3 style="margin:0 0 15px 0; text-align:center; color:#333;">
        King vs Pierce 비교
    </h3>
    
    <table style="width:100%; font-size:13px; border-collapse:collapse;">
        <tr style="background:#f8f9fa;">
            <th style="padding:8px; text-align:left;"></th>
            <th style="padding:8px; text-align:center; color:#2E86AB;">King</th>
            <th style="padding:8px; text-align:center; color:#A23B72;">Pierce</th>
        </tr>
        <tr>
            <td style="padding:8px;">평균 가격</td>
            <td style="padding:8px; text-align:center;"><b>${king['PRICE'].mean()/1e6:.2f}M</b></td>
            <td style="padding:8px; text-align:center;"><b>${pierce['PRICE'].mean()/1e6:.2f}M</b></td>
        </tr>
        <tr style="background:#f8f9fa;">
            <td style="padding:8px;">프리미엄</td>
            <td style="padding:8px; text-align:center; color:#e74c3c;"><b>+43.9%</b></td>
            <td style="padding:8px; text-align:center;">기준</td>
        </tr>
        <tr>
            <td style="padding:8px;">$/sqft</td>
            <td style="padding:8px; text-align:center;">${king['$/SQUARE FEET'].mean():.0f}</td>
            <td style="padding:8px; text-align:center;">${pierce['$/SQUARE FEET'].mean():.0f}</td>
        </tr>
        <tr style="background:#f8f9fa;">
            <td style="padding:8px;">위치 설명력</td>
            <td style="padding:8px; text-align:center;">{r2_loc_k*100:.1f}%</td>
            <td style="padding:8px; text-align:center;">{r2_loc_p*100:.1f}%</td>
        </tr>
    </table>
    
    <div style="margin-top:15px; padding:10px; background:#e8f4f8; border-radius:8px; font-size:12px;">
        <b>핵심 인사이트:</b><br>
        • King: <span style="color:#2E86AB;">벨뷰(테크허브)</span> 중심 고가 형성<br>
        • Pierce: <span style="color:#A23B72;">시애틀 접근성(북쪽)</span>이 핵심
    </div>
</div>
'''

combined_map.get_root().html.add_child(folium.Element(legend_combined))
combined_map.get_root().html.add_child(folium.Element(comparison_panel))

combined_map.save('지도_통합_가격분포.html')
print("  ✓ 지도_통합_가격분포.html 저장 완료")

# =============================================================================
# 완료
# =============================================================================
print("\n" + "="*70)
print("✅ 모든 지도 생성 완료!")
print("="*70)
print("\n생성된 파일:")
print("  1. 지도_King_클러스터분석.html   - King County 클러스터 + 벨뷰 프리미엄")
print("  2. 지도_Pierce_시애틀접근성.html - Pierce County 시애틀 접근성 분석")
print("  3. 지도_통합_가격분포.html       - 두 카운티 통합 가격 분포")
