import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
print("데이터 로딩 중...")
king_df = pd.read_csv('King_County_Sold.csv', skiprows=[1])
pierce_df = pd.read_csv('Pierce_County_Sold.csv', skiprows=[1])

king_df['COUNTY'] = 'King'
pierce_df['COUNTY'] = 'Pierce'

df = pd.concat([king_df, pierce_df], ignore_index=True)
df = df[['PRICE', 'LATITUDE', 'LONGITUDE', 'COUNTY', 'CITY']].dropna()

print(f"총 데이터: {len(df)}개")
print(f"위도 범위: {df['LATITUDE'].min():.4f} ~ {df['LATITUDE'].max():.4f}")

# 위도 구간 나누기 (0.1도 간격 = 약 11km)
df['lat_bin'] = pd.cut(df['LATITUDE'], bins=20)

# 각 구간별 통계
lat_stats = df.groupby('lat_bin').agg({
    'PRICE': ['mean', 'count'],
    'LATITUDE': 'mean',
    'COUNTY': lambda x: (x == 'King').sum() / len(x) * 100  # King 비율
}).reset_index()

lat_stats.columns = ['lat_bin', 'avg_price', 'count', 'lat_center', 'king_ratio']
lat_stats = lat_stats[lat_stats['count'] >= 5]  # 5개 이상인 구간만
lat_stats = lat_stats.sort_values('lat_center')

print("\n📊 위도 구간별 평균 가격:")
for idx, row in lat_stats.iterrows():
    county_type = "King 위주" if row['king_ratio'] > 80 else "Pierce 위주" if row['king_ratio'] < 20 else "혼합"
    print(f"  위도 {row['lat_center']:.3f}: ${row['avg_price']:,.0f} (N={int(row['count'])}, {county_type})")

# King-Pierce 경계 찾기
king_south = df[df['COUNTY']=='King']['LATITUDE'].min()
pierce_north = df[df['COUNTY']=='Pierce']['LATITUDE'].max()
boundary_lat = (king_south + pierce_north) / 2

print(f"\n🔍 King-Pierce 경계 추정: 위도 {boundary_lat:.4f}")
print(f"  • King 최남단: {king_south:.4f}")
print(f"  • Pierce 최북단: {pierce_north:.4f}")

# 시각화
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# 1. 위도별 평균 가격
ax1 = axes[0]
colors = ['#2E86AB' if r > 80 else '#A23B72' if r < 20 else '#9370DB' 
          for r in lat_stats['king_ratio']]

bars = ax1.bar(lat_stats['lat_center'], lat_stats['avg_price']/1000, 
               width=0.015, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)

# 경계선 표시
ax1.axvline(boundary_lat, color='#FF1493', linestyle='--', linewidth=2.5, 
            label=f'County Boundary (~{boundary_lat:.3f}°)', alpha=0.7)

ax1.set_xlabel('Latitude (위도)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Average Price ($1000s)', fontsize=13, fontweight='bold')
ax1.set_title('위도별 평균 가격 분석: 북쪽(King)으로 갈수록 가격 상승', 
              fontsize=15, fontweight='bold', pad=15)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.legend(fontsize=11)

# 가격 라벨 추가 (5개 간격)
for i, (idx, row) in enumerate(lat_stats.iterrows()):
    if i % 5 == 0:
        ax1.text(row['lat_center'], row['avg_price']/1000 + 30, 
                f'${row["avg_price"]/1000:.0f}K',
                ha='center', fontsize=9, fontweight='bold')

# 범례 추가
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E86AB', label='King County 위주 (>80%)'),
    Patch(facecolor='#9370DB', label='혼합 지역 (20-80%)'),
    Patch(facecolor='#A23B72', label='Pierce County 위주 (>80%)')
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

# 2. 위도별 데이터 개수 및 County 구성
ax2 = axes[1]

# King과 Pierce를 분리하여 스택 바 차트
king_counts = []
pierce_counts = []

for lat_bin in lat_stats['lat_bin']:
    bin_data = df[df['lat_bin'] == lat_bin]
    king_counts.append(len(bin_data[bin_data['COUNTY'] == 'King']))
    pierce_counts.append(len(bin_data[bin_data['COUNTY'] == 'Pierce']))

x_pos = lat_stats['lat_center']
ax2.bar(x_pos, king_counts, width=0.015, label='King County', 
        color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.bar(x_pos, pierce_counts, width=0.015, bottom=king_counts, 
        label='Pierce County', color='#A23B72', alpha=0.8, 
        edgecolor='black', linewidth=0.5)

ax2.axvline(boundary_lat, color='#FF1493', linestyle='--', linewidth=2.5, alpha=0.7)

ax2.set_xlabel('Latitude (위도)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Number of Properties', fontsize=13, fontweight='bold')
ax2.set_title('위도별 부동산 개수 및 County 구성', fontsize=15, fontweight='bold', pad=15)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('위도_가격분석.png', dpi=300, bbox_inches='tight')
print("\n✅ 그래프 저장 완료: 위도_가격분석.png")

# 추가 분석: 위도와 가격의 상관관계
correlation = df['LATITUDE'].corr(df['PRICE'])
print(f"\n📈 위도-가격 상관계수: {correlation:.4f}")
print(f"  → {'북쪽으로 갈수록 비쌈 (양의 상관)' if correlation > 0 else '남쪽으로 갈수록 비쌈 (음의 상관)'}")

# King과 Pierce 각각의 위도-가격 상관
king_corr = df[df['COUNTY']=='King']['LATITUDE'].corr(df[df['COUNTY']=='King']['PRICE'])
pierce_corr = df[df['COUNTY']=='Pierce']['LATITUDE'].corr(df[df['COUNTY']=='Pierce']['PRICE'])

print(f"\n📊 County별 위도-가격 상관:")
print(f"  • King: {king_corr:.4f}")
print(f"  • Pierce: {pierce_corr:.4f}")

# 경계 전후 가격 비교
boundary_range = 0.05  # 경계 ±0.05도 (약 5.5km)
near_boundary_north = df[(df['LATITUDE'] >= boundary_lat) & 
                         (df['LATITUDE'] <= boundary_lat + boundary_range)]
near_boundary_south = df[(df['LATITUDE'] <= boundary_lat) & 
                         (df['LATITUDE'] >= boundary_lat - boundary_range)]

print(f"\n🔍 경계 인근 가격 비교 (경계선 ±{boundary_range}도, 약 ±5.5km):")
print(f"  • 경계 북쪽 (King 남부): ${near_boundary_north['PRICE'].mean():,.0f} (N={len(near_boundary_north)})")
print(f"  • 경계 남쪽 (Pierce 북부): ${near_boundary_south['PRICE'].mean():,.0f} (N={len(near_boundary_south)})")
print(f"  • 차이: ${near_boundary_north['PRICE'].mean() - near_boundary_south['PRICE'].mean():,.0f}")

plt.show()
