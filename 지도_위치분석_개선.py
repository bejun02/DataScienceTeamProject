import pandas as pd
import folium
from math import radians, sin, cos, sqrt, asin

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

print(f"총 데이터: {len(df)}개")
print(f"King County: {len(df[df['COUNTY']=='King'])}개 (평균 ${df[df['COUNTY']=='King']['PRICE'].mean():,.0f})")
print(f"Pierce County: {len(df[df['COUNTY']=='Pierce'])}개 (평균 ${df[df['COUNTY']=='Pierce']['PRICE'].mean():,.0f})")

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

# 1. 시애틀 중심 마커 + 동심원 (파란색)
folium.Marker(
    location=SEATTLE_COORDS,
    popup='<b style="font-size:14px;">Seattle Downtown</b><br><i>Distance effect: -$9,800/km</i>',
    tooltip='🔵 Seattle',
    icon=folium.Icon(color='blue', icon='star', prefix='fa')
).add_to(m)

for radius_km in [10, 20, 30, 40]:
    folium.Circle(
        location=SEATTLE_COORDS,
        radius=radius_km * 1000,
        color='#4169E1',  # Royal Blue
        fill=False,
        weight=2,
        opacity=0.6,
        dashArray='5, 5',
        tooltip=f'Seattle {radius_km}km'
    ).add_to(m)

# 2. 벨뷰 중심 마커 + 동심원 (초록색)
folium.Marker(
    location=BELLEVUE_COORDS,
    popup='<b style="font-size:14px;">Bellevue Downtown</b><br><i>Tech Hub (Microsoft, Amazon)<br>Distance effect: -$28,500/km<br>(2.9x stronger than Seattle)</i>',
    tooltip='🟢 Bellevue',
    icon=folium.Icon(color='green', icon='building', prefix='fa')
).add_to(m)

for radius_km in [10, 20, 30, 40]:
    folium.Circle(
        location=BELLEVUE_COORDS,
        radius=radius_km * 1000,
        color='#32CD32',  # Lime Green
        fill=False,
        weight=2.5,
        opacity=0.7,
        dashArray='5, 5',
        tooltip=f'Bellevue {radius_km}km'
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
            <tr><td>도시</td><td style='text-align:right;'>{row['CITY']}</td></tr>
        </table>
        <hr style='margin:8px 0; border:1px solid #ddd;'>
        <div style='font-size:11px; background:#f8f9fa; padding:8px; border-radius:3px;'>
            <b>위치 분석:</b><br>
            • 시애틀: {row['dist_seattle']:.1f}km<br>
            • 벨뷰: {row['dist_bellevue']:.1f}km<br>
            <hr style='margin:5px 0; border:0.5px solid #ddd;'>
            <i style='color:#666;'>예상 거리 영향:</i><br>
            <span style='color:#4169E1;'>• 시애틀: -${row['dist_seattle']*9.8:.0f}K</span><br>
            <span style='color:#32CD32;'><b>• 벨뷰: -${row['dist_bellevue']*28.5:.0f}K</b></span>
        </div>
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

# 4. 범례
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
        <b>동심원 (Concentric Circles):</b><br>
        <span style="color:#4169E1;">🔵 시애틀 (10-40km)</span><br>
        <span style="color:#32CD32;">🟢 벨뷰 (10-40km)</span>
    </div>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# 5. 통계 박스
stats_king = df[df['COUNTY']=='King']
stats_pierce = df[df['COUNTY']=='Pierce']

# 벨뷰/시애틀 근접 지역 통계
close_bellevue = df[df['dist_bellevue'] <= 10]
close_seattle = df[df['dist_seattle'] <= 15]
far_both = df[(df['dist_seattle'] > 40) & (df['dist_bellevue'] > 40)]

stats_html = f'''
<div style="position:fixed; bottom:20px; left:20px; z-index:1000;
            background:white; padding:20px; border-radius:15px; max-width:420px;
            box-shadow:0 10px 40px rgba(0,0,0,0.3); font-family:Arial;">
    
    <h3 style="margin:0 0 15px 0; text-align:center; color:#333;">
        위치 기반 가격 분석
    </h3>
    
    <div style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color:white; padding:12px; border-radius:8px; margin-bottom:12px;">
        <div style="font-size:13px; margin-bottom:5px;">
            <b>핵심 발견:</b> 벨뷰 거리가 시애틀보다 <b>2.9배</b> 중요
        </div>
        <div style="font-size:12px; opacity:0.9;">
            • 벨뷰: <b>-$28,500/km</b><br>
            • 시애틀: -$9,800/km
        </div>
    </div>
    
    <table style="width:100%; font-size:12px; border-collapse:collapse;">
        <tr style="background:#f8f9fa;">
            <th style="padding:8px; text-align:left; border-bottom:2px solid #dee2e6;">구역</th>
            <th style="padding:8px; text-align:center; border-bottom:2px solid #dee2e6;">개수</th>
            <th style="padding:8px; text-align:right; border-bottom:2px solid #dee2e6;">평균 가격</th>
        </tr>
        <tr style="background:#e8f5e9;">
            <td style="padding:8px; border-bottom:1px solid #dee2e6;">
                <span style="color:#32CD32;">●</span> 벨뷰 10km 이내
            </td>
            <td style="padding:8px; text-align:center; border-bottom:1px solid #dee2e6;">{len(close_bellevue)}</td>
            <td style="padding:8px; text-align:right; border-bottom:1px solid #dee2e6;">
                <b style="color:#2e7d32;">${close_bellevue['PRICE'].mean():,.0f}</b>
            </td>
        </tr>
        <tr style="background:#e3f2fd;">
            <td style="padding:8px; border-bottom:1px solid #dee2e6;">
                <span style="color:#4169E1;">●</span> 시애틀 15km 이내
            </td>
            <td style="padding:8px; text-align:center; border-bottom:1px solid #dee2e6;">{len(close_seattle)}</td>
            <td style="padding:8px; text-align:right; border-bottom:1px solid #dee2e6;">
                <b style="color:#1565c0;">${close_seattle['PRICE'].mean():,.0f}</b>
            </td>
        </tr>
        <tr>
            <td style="padding:8px;">
                <span style="color:#999;">●</span> 두 도시 40km+ 이상
            </td>
            <td style="padding:8px; text-align:center;">{len(far_both)}</td>
            <td style="padding:8px; text-align:right;">
                <b>${far_both['PRICE'].mean():,.0f}</b>
            </td>
        </tr>
    </table>
    
    <hr style="margin:12px 0;">
    
    <table style="width:100%; font-size:12px; border-collapse:collapse;">
        <tr style="background:#f8f9fa;">
            <th style="padding:6px; text-align:left;"></th>
            <th style="padding:6px; text-align:center; color:#2E86AB;">King</th>
            <th style="padding:6px; text-align:center; color:#A23B72;">Pierce</th>
        </tr>
        <tr>
            <td style="padding:6px;">평균 가격</td>
            <td style="padding:6px; text-align:center;"><b>${stats_king['PRICE'].mean():,.0f}</b></td>
            <td style="padding:6px; text-align:center;"><b>${stats_pierce['PRICE'].mean():,.0f}</b></td>
        </tr>
        <tr>
            <td style="padding:6px;">벨뷰 평균 거리</td>
            <td style="padding:6px; text-align:center;">{stats_king['dist_bellevue'].mean():.1f} km</td>
            <td style="padding:6px; text-align:center;">{stats_pierce['dist_bellevue'].mean():.1f} km</td>
        </tr>
        <tr>
            <td style="padding:6px;">시애틀 평균 거리</td>
            <td style="padding:6px; text-align:center;">{stats_king['dist_seattle'].mean():.1f} km</td>
            <td style="padding:6px; text-align:center;">{stats_pierce['dist_seattle'].mean():.1f} km</td>
        </tr>
    </table>
    
    <div style="margin-top:12px; padding:10px; background:#fff3cd; border-radius:5px; font-size:11px;">
        <b style="color:#856404;">💡 인사이트:</b><br>
        <span style="color:#533f03;">
        King County는 두 도시 모두에 가까워<br>
        "위치 프리미엄" 효과가 가격에 반영됨
        </span>
    </div>
</div>
'''
m.get_root().html.add_child(folium.Element(stats_html))

# 6. 타이틀
title_html = '''
<div style="position:fixed; top:10px; left:50px; 
            width:600px; background:rgba(255,255,255,0.95); 
            border:2px solid #667eea; z-index:9999; 
            padding:15px; border-radius:10px;
            box-shadow:0 5px 20px rgba(0,0,0,0.3);">
    <h2 style="margin:0; color:#2c3e50;">
        🏙️ 위치 기반 주택 가격 분석
    </h2>
    <p style="margin:5px 0 0 0; color:#7f8c8d; font-size:14px;">
        King & Pierce County | May-Oct 2024 | N=''' + str(len(df)) + ''' properties<br>
        <b style="color:#667eea;">벨뷰 접근성</b>이 시애틀보다 <b>2.9배 더 중요</b>한 가격 결정 요인
    </p>
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# 저장
output_file = '지도_통합_가격분포.html'
m.save(output_file)
print(f"\n✅ 지도가 생성되었습니다: {output_file}")
print("\n📊 위치별 가격 통계:")
print(f"  • 벨뷰 10km 이내: {len(close_bellevue)}개 (평균 ${close_bellevue['PRICE'].mean():,.0f}) - 프리미엄 {(close_bellevue['PRICE'].mean()/far_both['PRICE'].mean()-1)*100:.1f}%")
print(f"  • 시애틀 15km 이내: {len(close_seattle)}개 (평균 ${close_seattle['PRICE'].mean():,.0f}) - 프리미엄 {(close_seattle['PRICE'].mean()/far_both['PRICE'].mean()-1)*100:.1f}%")
print(f"  • 두 도시 40km+ 이상: {len(far_both)}개 (평균 ${far_both['PRICE'].mean():,.0f}) - 기준")
