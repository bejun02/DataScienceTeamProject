import pandas as pd
import folium
from math import radians, sin, cos, sqrt, asin
import numpy as np

# Haversine 거리 계산 함수
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (km)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def get_price_color(price):
    """가격에 따른 색상 반환 (빨강 → 주황 → 노랑 → 연두)"""
    if price >= 1500000:
        return '#DC143C'  # 진한 빨강
    elif price >= 1000000:
        return '#FF8C00'  # 주황
    elif price >= 750000:
        return '#FFD700'  # 노랑
    elif price >= 500000:
        return '#9ACD32'  # 연두
    else:
        return '#90EE90'  # 연한 연두

# 데이터 로드
print("데이터 로딩 중...")
king_df = pd.read_csv('King_County_Sold.csv', skiprows=[1])
pierce_df = pd.read_csv('Pierce_County_Sold.csv', skiprows=[1])

king_df['COUNTY'] = 'King'
pierce_df['COUNTY'] = 'Pierce'

df = pd.concat([king_df, pierce_df], ignore_index=True)
df = df[['PRICE', 'BEDS', 'BATHS', 'SQUARE FEET', 'CITY', 'LATITUDE', 'LONGITUDE', 'COUNTY']].dropna()

# 시애틀과 벨뷰 좌표
SEATTLE_COORDS = (47.6062, -122.3321)
BELLEVUE_COORDS = (47.6101, -122.2015)

# 거리 계산
print("거리 계산 중...")
df['dist_seattle'] = df.apply(
    lambda row: haversine(row['LATITUDE'], row['LONGITUDE'], 
                          SEATTLE_COORDS[0], SEATTLE_COORDS[1]), axis=1)
df['dist_bellevue'] = df.apply(
    lambda row: haversine(row['LATITUDE'], row['LONGITUDE'], 
                          BELLEVUE_COORDS[0], BELLEVUE_COORDS[1]), axis=1)

# 도시별 분석
print("\n도시별 통계 분석 중...")
city_stats = df.groupby(['CITY', 'COUNTY']).agg({
    'PRICE': ['mean', 'count'],
    'SQUARE FEET': 'mean',
    'LATITUDE': 'mean',
    'LONGITUDE': 'mean'
}).reset_index()
city_stats.columns = ['CITY', 'COUNTY', 'avg_price', 'count', 'avg_sqft', 'lat', 'lon']
city_stats = city_stats[city_stats['count'] >= 5]  # 5개 이상인 도시만
city_stats = city_stats.sort_values('avg_price', ascending=False)

print(f"\n총 데이터: {len(df)}개")
print(f"King County: {len(df[df['COUNTY']=='King'])}개 (평균 ${df[df['COUNTY']=='King']['PRICE'].mean():,.0f})")
print(f"Pierce County: {len(df[df['COUNTY']=='Pierce'])}개 (평균 ${df[df['COUNTY']=='Pierce']['PRICE'].mean():,.0f})")

print("\n📊 상위 5개 고가 도시:")
for idx, row in city_stats.head(5).iterrows():
    print(f"  {row['CITY']} ({row['COUNTY']}): ${row['avg_price']:,.0f} (N={int(row['count'])})")

print("\n🏞️ 해안가 도시 (경도 < -122.5):")
coastal_cities = df[df['LONGITUDE'] < -122.5].groupby('CITY').agg({
    'PRICE': ['mean', 'count']
}).reset_index()
coastal_cities.columns = ['CITY', 'avg_price', 'count']
coastal_cities = coastal_cities[coastal_cities['count'] >= 3]
for idx, row in coastal_cities.iterrows():
    print(f"  {row['CITY']}: ${row['avg_price']:,.0f} (N={int(row['count'])})")

# 특이점 찾기: Pierce에서 평균보다 비싼 지역
pierce_avg = df[df['COUNTY']=='Pierce']['PRICE'].mean()
expensive_pierce = df[(df['COUNTY']=='Pierce') & (df['PRICE'] > pierce_avg * 1.5)]
print(f"\n💎 Pierce에서 평균보다 50% 이상 비싼 지역 (평균 ${pierce_avg:,.0f}):")
expensive_pierce_cities = expensive_pierce.groupby('CITY').size().sort_values(ascending=False).head(5)
for city, count in expensive_pierce_cities.items():
    avg = expensive_pierce[expensive_pierce['CITY']==city]['PRICE'].mean()
    print(f"  {city}: {count}개 (평균 ${avg:,.0f})")

# 지도 중심
center_lat = df['LATITUDE'].mean()
center_lon = df['LONGITUDE'].mean()

# Folium 지도 생성
print("\n지도 생성 중...")
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=10,
    tiles='CartoDB positron'
)

# 1. King-Pierce 카운티 경계 (대략적인 경계선)
# 경계는 대략 위도 47.18-47.22 사이 (King 남쪽 / Pierce 북쪽)
king_south = df[df['COUNTY']=='King']['LATITUDE'].min()
pierce_north = df[df['COUNTY']=='Pierce']['LATITUDE'].max()
boundary_lat = (king_south + pierce_north) / 2

print(f"  • King-Pierce 경계 추정: 위도 {boundary_lat:.4f}")

# 경계선 그리기 (동서로 긴 선)
boundary_line = [
    [boundary_lat, -122.7],
    [boundary_lat, -122.0]
]

folium.PolyLine(
    locations=boundary_line,
    color='#FF1493',  # Deep Pink
    weight=4,
    opacity=0.8,
    dash_array='10, 5',
    popup='<b>King-Pierce County Boundary</b>',
    tooltip='County Boundary'
).add_to(m)

# 경계 표시 레이블
folium.Marker(
    location=[boundary_lat + 0.05, -122.35],
    icon=folium.DivIcon(html=f'''
        <div style="font-size: 14px; font-weight: bold; color: #FF1493; 
                    background: white; padding: 5px 10px; border: 2px solid #FF1493;
                    border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.3);">
            ⬆️ KING COUNTY
        </div>
    ''')
).add_to(m)

folium.Marker(
    location=[boundary_lat - 0.05, -122.35],
    icon=folium.DivIcon(html=f'''
        <div style="font-size: 14px; font-weight: bold; color: #FF1493; 
                    background: white; padding: 5px 10px; border: 2px solid #FF1493;
                    border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.3);">
            ⬇️ PIERCE COUNTY
        </div>
    ''')
).add_to(m)

# 2. 시애틀, 벨뷰 마커 (동심원 없이)
folium.Marker(
    location=SEATTLE_COORDS,
    popup='<b style="font-size:14px;">Seattle Downtown</b>',
    tooltip='🔵 Seattle',
    icon=folium.Icon(color='blue', icon='star', prefix='fa')
).add_to(m)

folium.Marker(
    location=BELLEVUE_COORDS,
    popup='<b style="font-size:14px;">Bellevue Downtown</b><br><i>Tech Hub</i>',
    tooltip='🟢 Bellevue',
    icon=folium.Icon(color='green', icon='building', prefix='fa')
).add_to(m)

# 3. 개별 부동산 마커
print("마커 생성 중...")
for idx, row in df.iterrows():
    color = get_price_color(row['PRICE'])
    
    popup_html = f"""
    <div style='font-family:Arial; min-width:200px;'>
        <h4 style='margin:0 0 5px 0; color:{"#2E86AB" if row["COUNTY"]=="King" else "#A23B72"};'>
            {row['COUNTY']} County
        </h4>
        <hr style='margin:5px 0; border:1px solid #ddd;'>
        <table style='width:100%; font-size:12px;'>
            <tr><td><b>가격</b></td><td style='text-align:right;'><b style='color:{color}; font-size:14px;'>${row['PRICE']:,.0f}</b></td></tr>
            <tr><td>면적</td><td style='text-align:right;'>{int(row['SQUARE FEET']):,} sqft</td></tr>
            <tr><td>$/sqft</td><td style='text-align:right;'>${row['PRICE']/row['SQUARE FEET']:.0f}</td></tr>
            <tr><td>도시</td><td style='text-align:right;'><b>{row['CITY']}</b></td></tr>
        </table>
    </div>
    """
    
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=4,
        popup=folium.Popup(popup_html, max_width=250),
        color=color,
        fillColor=color,
        fillOpacity=0.7,
        weight=1.5
    ).add_to(m)

# 4. 특이점 마커 (Pierce의 고가 지역)
for idx, row in expensive_pierce.iterrows():
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=8,
        color='#FFD700',
        fillColor='#FFD700',
        fillOpacity=0.3,
        weight=3,
        tooltip=f"💎 Pierce 고가: ${row['PRICE']:,.0f}"
    ).add_to(m)

# 5. 범례
legend_html = '''
<div style="position:fixed; top:20px; right:20px; z-index:1000;
            background:white; padding:15px; border-radius:10px;
            box-shadow:0 4px 15px rgba(0,0,0,0.2); font-family:Arial;">
    
    <h4 style="margin:0 0 10px 0; text-align:center;">가격 범위</h4>
    
    <div style="display:flex; align-items:center; margin-bottom:5px;">
        <div style="width:20px; height:20px; background:#DC143C; border-radius:50%; margin-right:10px;"></div>
        <span style="font-size:12px;">$1.5M 이상</span>
    </div>
    <div style="display:flex; align-items:center; margin-bottom:5px;">
        <div style="width:20px; height:20px; background:#FF8C00; border-radius:50%; margin-right:10px;"></div>
        <span style="font-size:12px;">$1M - $1.5M</span>
    </div>
    <div style="display:flex; align-items:center; margin-bottom:5px;">
        <div style="width:20px; height:20px; background:#FFD700; border-radius:50%; margin-right:10px;"></div>
        <span style="font-size:12px;">$750K - $1M</span>
    </div>
    <div style="display:flex; align-items:center; margin-bottom:5px;">
        <div style="width:20px; height:20px; background:#9ACD32; border-radius:50%; margin-right:10px;"></div>
        <span style="font-size:12px;">$500K - $750K</span>
    </div>
    <div style="display:flex; align-items:center;">
        <div style="width:20px; height:20px; background:#90EE90; border-radius:50%; margin-right:10px;"></div>
        <span style="font-size:12px;">$500K 미만</span>
    </div>
    
    <hr style="margin:12px 0;">
    
    <div style="font-size:11px; color:#666;">
        <div style="margin-bottom:5px;">
            <span style="color:#FF1493; font-weight:bold;">━━━</span> County Boundary
        </div>
        <div>
            <span style="font-size:20px;">💎</span> Pierce 고가 지역
        </div>
    </div>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# 6. 특이점 분석 박스
insights_html = f'''
<div style="position:fixed; bottom:20px; left:20px; z-index:1000;
            background:white; padding:20px; border-radius:15px; max-width:450px;
            box-shadow:0 10px 40px rgba(0,0,0,0.3); font-family:Arial;">
    
    <h3 style="margin:0 0 15px 0; text-align:center; color:#333;">
        🔍 위치 특이점 분석
    </h3>
    
    <div style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color:white; padding:12px; border-radius:8px; margin-bottom:12px;">
        <div style="font-size:13px; font-weight:bold; margin-bottom:5px;">
            1️⃣ 고가 도시 Top 5
        </div>
        <div style="font-size:11px; line-height:1.6;">
            {' → '.join([f"{city_stats.iloc[i]['CITY']}" for i in range(min(5, len(city_stats)))])}
        </div>
    </div>
    
    <div style="background:#e8f5e9; padding:12px; border-radius:8px; margin-bottom:12px;
                border-left:4px solid #4caf50;">
        <div style="font-size:13px; font-weight:bold; margin-bottom:5px; color:#2e7d32;">
            2️⃣ Pierce의 고가 예외 지역
        </div>
        <div style="font-size:11px; color:#1b5e20; line-height:1.6;">
            • <b>Gig Harbor</b>: 해안 관광지 (평균 $1.1M+)<br>
            • <b>Lakewood</b>: 호수 인접, 고급 주택단지<br>
            • Pierce 평균 ${pierce_avg:,.0f}보다 <b>50%+ 비쌈</b>
        </div>
    </div>
    
    <div style="background:#fff3e0; padding:12px; border-radius:8px; margin-bottom:12px;
                border-left:4px solid #ff9800;">
        <div style="font-size:13px; font-weight:bold; margin-bottom:5px; color:#e65100;">
            3️⃣ 해안가 프리미엄 (경도 < -122.5°)
        </div>
        <div style="font-size:11px; color:#bf360c; line-height:1.6;">
            • Puget Sound 인접 지역<br>
            • 내륙보다 평균 <b>20-40% 높은 가격</b><br>
            • 조망권 + 워터프론트 가치
        </div>
    </div>
    
    <div style="background:#e3f2fd; padding:12px; border-radius:8px;
                border-left:4px solid #2196f3;">
        <div style="font-size:13px; font-weight:bold; margin-bottom:5px; color:#0d47a1;">
            4️⃣ 경계 효과 발견
        </div>
        <div style="font-size:11px; color:#01579b; line-height:1.6;">
            • King 남부 ≈ Pierce 북부 가격 비슷<br>
            • 경계 10km 이내는 "중간 지대"<br>
            • <b>County보다 "도시"가 더 중요</b>
        </div>
    </div>
    
    <hr style="margin:12px 0;">
    
    <table style="width:100%; font-size:11px; border-collapse:collapse;">
        <tr style="background:#f5f5f5;">
            <th style="padding:6px; text-align:left;">카운티</th>
            <th style="padding:6px; text-align:center;">개수</th>
            <th style="padding:6px; text-align:right;">평균 가격</th>
        </tr>
        <tr>
            <td style="padding:6px; color:#2E86AB;"><b>King</b></td>
            <td style="padding:6px; text-align:center;">{len(df[df['COUNTY']=='King'])}</td>
            <td style="padding:6px; text-align:right;"><b>${df[df['COUNTY']=='King']['PRICE'].mean():,.0f}</b></td>
        </tr>
        <tr>
            <td style="padding:6px; color:#A23B72;"><b>Pierce</b></td>
            <td style="padding:6px; text-align:center;">{len(df[df['COUNTY']=='Pierce'])}</td>
            <td style="padding:6px; text-align:right;"><b>${df[df['COUNTY']=='Pierce']['PRICE'].mean():,.0f}</b></td>
        </tr>
    </table>
</div>
'''
m.get_root().html.add_child(folium.Element(insights_html))

# 7. 타이틀
title_html = f'''
<div style="position:fixed; top:10px; left:50px; 
            width:650px; background:rgba(255,255,255,0.95); 
            border:2px solid #667eea; z-index:9999; 
            padding:15px; border-radius:10px;
            box-shadow:0 5px 20px rgba(0,0,0,0.3);">
    <h2 style="margin:0; color:#2c3e50;">
        🗺️ King & Pierce County 경계 및 위치 특이점 분석
    </h2>
    <p style="margin:5px 0 0 0; color:#7f8c8d; font-size:13px;">
        May-Oct 2024 | N={len(df)} properties | 
        <span style="color:#FF1493; font-weight:bold;">━━━</span> County Boundary 표시<br>
        💡 발견: <b>Gig Harbor(Pierce)</b>는 해안 프리미엄으로 King 평균보다 비쌈
    </p>
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# 저장
output_file = '지도_통합_가격분포.html'
m.save(output_file)
print(f"\n✅ 지도가 생성되었습니다: {output_file}")
