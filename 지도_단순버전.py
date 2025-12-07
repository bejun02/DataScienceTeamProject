import pandas as pd
import folium
from math import radians, sin, cos, sqrt, asin

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

print(f"총 데이터: {len(df)}개")
print(f"King County: {len(df[df['COUNTY']=='King'])}개")
print(f"Pierce County: {len(df[df['COUNTY']=='Pierce'])}개")

# 지도 중심
center_lat = df['LATITUDE'].mean()
center_lon = df['LONGITUDE'].mean()

# Folium 지도 생성
print("지도 생성 중...")
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=10,
    tiles='CartoDB positron'
)

# 개별 부동산 마커
print("마커 생성 중...")
for idx, row in df.iterrows():
    color = get_price_color(row['PRICE'])
    
    popup_html = f"""
    <div style='font-family:Arial; min-width:180px;'>
        <h4 style='margin:0 0 5px 0; color:{"#2E86AB" if row["COUNTY"]=="King" else "#A23B72"};'>
            {row['COUNTY']} County
        </h4>
        <table style='width:100%; font-size:12px;'>
            <tr><td><b>Price</b></td><td style='text-align:right;'><b style='color:{color}; font-size:14px;'>${row['PRICE']:,.0f}</b></td></tr>
            <tr><td>Area</td><td style='text-align:right;'>{int(row['SQUARE FEET']):,} sqft</td></tr>
            <tr><td>$/sqft</td><td style='text-align:right;'>${row['PRICE']/row['SQUARE FEET']:.0f}</td></tr>
            <tr><td>City</td><td style='text-align:right;'>{row['CITY']}</td></tr>
        </table>
    </div>
    """
    
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=4,
        popup=folium.Popup(popup_html, max_width=220),
        color=color,
        fillColor=color,
        fillOpacity=0.7,
        weight=1.5
    ).add_to(m)

# Legend
legend_html = '''
<div style="position:fixed; top:20px; right:20px; z-index:1000;
            background:white; padding:15px; border-radius:10px;
            box-shadow:0 4px 15px rgba(0,0,0,0.2); font-family:Arial;">
    
    <h4 style="margin:0 0 10px 0; text-align:center;">Price Range</h4>
    
    <div style="display:flex; align-items:center; margin-bottom:5px;">
        <div style="width:20px; height:20px; background:#DC143C; border-radius:50%; margin-right:10px;"></div>
        <span style="font-size:12px;">≥ $1.5M</span>
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
        <span style="font-size:12px;">< $500K</span>
    </div>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# 저장
output_file = '지도_통합_가격분포.html'
m.save(output_file)
print(f"\n✅ 지도가 생성되었습니다: {output_file}")
