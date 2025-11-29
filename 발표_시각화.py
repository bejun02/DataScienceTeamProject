# -*- coding: utf-8 -*-
"""
프로젝트 발표용 시각화 자료 생성
King County vs Pierce County 부동산 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 한글 폰트 설정
# ============================================================================
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 데이터 전처리 함수
# ============================================================================
def preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    columns_to_drop = [
        'SALE TYPE', 'STATE OR PROVINCE', 'STATUS',
        'NEXT OPEN HOUSE START TIME', 'NEXT OPEN HOUSE END TIME',
        'SOURCE', 'FAVORITE', 'INTERESTED', 'SOLD DATE', 'MLS#',
        'ZIP OR POSTAL CODE', 'HOA/MONTH', 'ADDRESS', 'LOCATION',
        'LOT SIZE', 'DAYS ON MARKET',
        'URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    df = df.dropna()
    types_to_remove = ['Mobile/Manufactured Home', 'Multi-Family (2-4 Unit)', 'Vacant Land']
    df = df[~df['PROPERTY TYPE'].isin(types_to_remove)]
    df = df[(df['PRICE'] != 7050000) & (df['PRICE'] != 4500000)]
    return df

# 데이터 로드
print("데이터 로드 중...")
king_df = preprocess_data('King_County_Sold.csv')
pierce_df = preprocess_data('Pierce_County_Sold.csv')
print(f"King County: {len(king_df)}건, Pierce County: {len(pierce_df)}건")

# ============================================================================
# Figure 1: 가격 분포 비교 (히스토그램 + 박스플롯)
# ============================================================================
print("\n[Figure 1] 가격 분포 비교 생성 중...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Figure 1: 가격 분포 비교 (King County vs Pierce County)', fontsize=16, fontweight='bold')

# 1-1. 히스토그램 - King County
ax1 = axes[0, 0]
ax1.hist(king_df['PRICE']/1000000, bins=20, color='#2E86AB', edgecolor='white', alpha=0.8)
ax1.axvline(king_df['PRICE'].mean()/1000000, color='red', linestyle='--', linewidth=2, label=f'평균: ${king_df["PRICE"].mean()/1000000:.2f}M')
ax1.axvline(king_df['PRICE'].median()/1000000, color='orange', linestyle='--', linewidth=2, label=f'중앙값: ${king_df["PRICE"].median()/1000000:.2f}M')
ax1.set_xlabel('가격 (백만 달러)', fontsize=11)
ax1.set_ylabel('빈도', fontsize=11)
ax1.set_title('King County 가격 분포', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# 1-2. 히스토그램 - Pierce County
ax2 = axes[0, 1]
ax2.hist(pierce_df['PRICE']/1000000, bins=20, color='#A23B72', edgecolor='white', alpha=0.8)
ax2.axvline(pierce_df['PRICE'].mean()/1000000, color='red', linestyle='--', linewidth=2, label=f'평균: ${pierce_df["PRICE"].mean()/1000000:.2f}M')
ax2.axvline(pierce_df['PRICE'].median()/1000000, color='orange', linestyle='--', linewidth=2, label=f'중앙값: ${pierce_df["PRICE"].median()/1000000:.2f}M')
ax2.set_xlabel('가격 (백만 달러)', fontsize=11)
ax2.set_ylabel('빈도', fontsize=11)
ax2.set_title('Pierce County 가격 분포', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(axis='y', alpha=0.3)

# 1-3. 박스플롯 비교
ax3 = axes[1, 0]
bp = ax3.boxplot([king_df['PRICE']/1000000, pierce_df['PRICE']/1000000], 
                  labels=['King County', 'Pierce County'],
                  patch_artist=True)
bp['boxes'][0].set_facecolor('#2E86AB')
bp['boxes'][1].set_facecolor('#A23B72')
ax3.set_ylabel('가격 (백만 달러)', fontsize=11)
ax3.set_title('가격 분포 박스플롯 비교', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 1-4. 통계 요약 테이블
ax4 = axes[1, 1]
ax4.axis('off')
stats_data = [
    ['지표', 'King County', 'Pierce County', '차이'],
    ['평균 가격', f'${king_df["PRICE"].mean():,.0f}', f'${pierce_df["PRICE"].mean():,.0f}', f'+{(king_df["PRICE"].mean()-pierce_df["PRICE"].mean()):,.0f}'],
    ['중앙값', f'${king_df["PRICE"].median():,.0f}', f'${pierce_df["PRICE"].median():,.0f}', f'+{(king_df["PRICE"].median()-pierce_df["PRICE"].median()):,.0f}'],
    ['표준편차', f'${king_df["PRICE"].std():,.0f}', f'${pierce_df["PRICE"].std():,.0f}', ''],
    ['최소값', f'${king_df["PRICE"].min():,.0f}', f'${pierce_df["PRICE"].min():,.0f}', ''],
    ['최대값', f'${king_df["PRICE"].max():,.0f}', f'${pierce_df["PRICE"].max():,.0f}', ''],
    ['데이터 수', f'{len(king_df)}건', f'{len(pierce_df)}건', ''],
    ['', '', '', ''],
    ['가격 프리미엄', '', '', f'+{((king_df["PRICE"].mean()/pierce_df["PRICE"].mean())-1)*100:.1f}%']
]
table = ax4.table(cellText=stats_data, loc='center', cellLoc='center',
                   colWidths=[0.25, 0.25, 0.25, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)
# 헤더 스타일
for j in range(4):
    table[(0, j)].set_facecolor('#34495e')
    table[(0, j)].set_text_props(color='white', fontweight='bold')
# 마지막 행 강조
for j in range(4):
    table[(8, j)].set_facecolor('#e8f4f8')
    table[(8, j)].set_text_props(fontweight='bold')
ax4.set_title('기술통계 요약', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('시각화_1_가격분포비교.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   저장 완료: 시각화_1_가격분포비교.png")

# ============================================================================
# Figure 2: 상관관계 분석 (산점도 + 상관계수)
# ============================================================================
print("\n[Figure 2] 상관관계 분석 생성 중...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Figure 2: 가격과 주요 변수 간 상관관계', fontsize=16, fontweight='bold')

variables = ['SQUARE FEET', 'BEDS', 'BATHS', 'YEAR BUILT']
colors_king = '#2E86AB'
colors_pierce = '#A23B72'

for idx, var in enumerate(variables[:3]):
    ax = axes[0, idx]
    
    # King County
    ax.scatter(king_df[var], king_df['PRICE']/1000000, alpha=0.5, c=colors_king, label='King', s=30)
    ax.scatter(pierce_df[var], pierce_df['PRICE']/1000000, alpha=0.5, c=colors_pierce, label='Pierce', s=30)
    
    # 상관계수 계산
    r_king = king_df[var].corr(king_df['PRICE'])
    r_pierce = pierce_df[var].corr(pierce_df['PRICE'])
    
    ax.set_xlabel(var, fontsize=11)
    ax.set_ylabel('가격 (백만 달러)', fontsize=11)
    ax.set_title(f'{var} vs PRICE\nKing r={r_king:.3f}, Pierce r={r_pierce:.3f}', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

# YEAR BUILT
ax = axes[1, 0]
ax.scatter(king_df['YEAR BUILT'], king_df['PRICE']/1000000, alpha=0.5, c=colors_king, label='King', s=30)
ax.scatter(pierce_df['YEAR BUILT'], pierce_df['PRICE']/1000000, alpha=0.5, c=colors_pierce, label='Pierce', s=30)
r_king = king_df['YEAR BUILT'].corr(king_df['PRICE'])
r_pierce = pierce_df['YEAR BUILT'].corr(pierce_df['PRICE'])
ax.set_xlabel('YEAR BUILT', fontsize=11)
ax.set_ylabel('가격 (백만 달러)', fontsize=11)
ax.set_title(f'YEAR BUILT vs PRICE\nKing r={r_king:.3f}, Pierce r={r_pierce:.3f}', fontsize=11, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)

# 상관계수 히트맵 스타일 바 차트
ax = axes[1, 1]
vars_list = ['SQUARE FEET', 'BATHS', 'BEDS', 'YEAR BUILT', '$/SQUARE FEET']
king_corrs = [king_df[v].corr(king_df['PRICE']) for v in vars_list]
pierce_corrs = [pierce_df[v].corr(pierce_df['PRICE']) for v in vars_list]

x = np.arange(len(vars_list))
width = 0.35
bars1 = ax.barh(x - width/2, king_corrs, width, label='King County', color=colors_king)
bars2 = ax.barh(x + width/2, pierce_corrs, width, label='Pierce County', color=colors_pierce)
ax.set_yticks(x)
ax.set_yticklabels(vars_list)
ax.set_xlabel('상관계수 (Pearson r)', fontsize=11)
ax.set_title('PRICE와의 상관계수 비교', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlim(-0.1, 0.85)
ax.grid(axis='x', alpha=0.3)

# 핵심 발견 텍스트
ax = axes[1, 2]
ax.axis('off')
findings = """
【 상관관계 분석 핵심 발견 】

1. SQUARE FEET (건물 면적)
   → 두 카운티 모두 가장 강한 상관관계
   → King: r = 0.754, Pierce: r = 0.734
   → "면적이 클수록 가격이 높다"

2. BATHS (욕실 수)
   → 두 번째로 강한 상관관계
   → King: r = 0.608, Pierce: r = 0.534

3. YEAR BUILT (건축연도)
   → Pierce에서 더 강한 영향
   → King: r = 0.131, Pierce: r = 0.229
   → "Pierce는 신축 프리미엄이 더 큼"

※ 핵심 결론:
   건물 크기(SQUARE FEET)가 가격의
   가장 중요한 결정 요인
"""
ax.text(0.1, 0.95, findings, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='Malgun Gothic',
        bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

plt.tight_layout()
plt.savefig('시각화_2_상관관계분석.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   저장 완료: 시각화_2_상관관계분석.png")

# ============================================================================
# Figure 3: 카운티 특성 비교 (경제 기능 차이)
# ============================================================================
print("\n[Figure 3] 카운티 특성 비교 생성 중...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Figure 3: King County vs Pierce County 특성 비교', fontsize=16, fontweight='bold')

# 3-1. 평균 비교 (면적, 침실, 욕실)
ax = axes[0, 0]
categories = ['평균 면적\n(sqft)', '평균 침실\n(개)', '평균 욕실\n(개)']
king_vals = [king_df['SQUARE FEET'].mean(), king_df['BEDS'].mean()*500, king_df['BATHS'].mean()*500]
pierce_vals = [pierce_df['SQUARE FEET'].mean(), pierce_df['BEDS'].mean()*500, pierce_df['BATHS'].mean()*500]

x = np.arange(len(categories))
width = 0.35
ax.bar(x - width/2, [king_df['SQUARE FEET'].mean(), king_df['BEDS'].mean(), king_df['BATHS'].mean()], 
       width, label='King', color='#2E86AB')
ax.bar(x + width/2, [pierce_df['SQUARE FEET'].mean(), pierce_df['BEDS'].mean(), pierce_df['BATHS'].mean()], 
       width, label='Pierce', color='#A23B72')
ax.set_xticks(x)
ax.set_xticklabels(['SQUARE FEET', 'BEDS', 'BATHS'])
ax.set_title('건물 속성 평균 비교', fontsize=12, fontweight='bold')
ax.legend()
# 값 표시
for i, (k, p) in enumerate(zip([king_df['SQUARE FEET'].mean(), king_df['BEDS'].mean(), king_df['BATHS'].mean()],
                                [pierce_df['SQUARE FEET'].mean(), pierce_df['BEDS'].mean(), pierce_df['BATHS'].mean()])):
    if i == 0:
        ax.text(i - width/2, k + 50, f'{k:.0f}', ha='center', fontsize=9)
        ax.text(i + width/2, p + 50, f'{p:.0f}', ha='center', fontsize=9)
    else:
        ax.text(i - width/2, k + 0.05, f'{k:.2f}', ha='center', fontsize=9)
        ax.text(i + width/2, p + 0.05, f'{p:.2f}', ha='center', fontsize=9)

# 3-2. $/SQUARE FEET 비교
ax = axes[0, 1]
data = [king_df['$/SQUARE FEET'], pierce_df['$/SQUARE FEET']]
bp = ax.boxplot(data, labels=['King County', 'Pierce County'], patch_artist=True)
bp['boxes'][0].set_facecolor('#2E86AB')
bp['boxes'][1].set_facecolor('#A23B72')
ax.set_ylabel('$/SQUARE FEET', fontsize=11)
ax.set_title('면적당 가격 ($/sqft) 비교', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
# 평균 표시
ax.text(1, king_df['$/SQUARE FEET'].mean() + 20, f'평균: ${king_df["$/SQUARE FEET"].mean():.0f}', ha='center', fontsize=10, color='#2E86AB', fontweight='bold')
ax.text(2, pierce_df['$/SQUARE FEET'].mean() + 20, f'평균: ${pierce_df["$/SQUARE FEET"].mean():.0f}', ha='center', fontsize=10, color='#A23B72', fontweight='bold')

# 3-3. 건축년도 분포
ax = axes[0, 2]
ax.hist(king_df['YEAR BUILT'], bins=15, alpha=0.6, label=f'King (평균: {king_df["YEAR BUILT"].mean():.0f})', color='#2E86AB')
ax.hist(pierce_df['YEAR BUILT'], bins=15, alpha=0.6, label=f'Pierce (평균: {pierce_df["YEAR BUILT"].mean():.0f})', color='#A23B72')
ax.set_xlabel('건축년도', fontsize=11)
ax.set_ylabel('빈도', fontsize=11)
ax.set_title('건축년도 분포', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 3-4. PROPERTY TYPE 비교 - King
ax = axes[1, 0]
king_types = king_df['PROPERTY TYPE'].value_counts()
colors = ['#2E86AB', '#5DADE2', '#85C1E9']
wedges, texts, autotexts = ax.pie(king_types.values, labels=king_types.index, autopct='%1.1f%%', 
                                   colors=colors, explode=[0.02]*len(king_types))
ax.set_title('King County 주택 유형', fontsize=12, fontweight='bold')

# 3-5. PROPERTY TYPE 비교 - Pierce
ax = axes[1, 1]
pierce_types = pierce_df['PROPERTY TYPE'].value_counts()
colors = ['#A23B72', '#D35D90', '#E88AAE']
wedges, texts, autotexts = ax.pie(pierce_types.values, labels=pierce_types.index, autopct='%1.1f%%',
                                   colors=colors, explode=[0.02]*len(pierce_types))
ax.set_title('Pierce County 주택 유형', fontsize=12, fontweight='bold')

# 3-6. 특성 요약 텍스트
ax = axes[1, 2]
ax.axis('off')
summary = """
【 두 카운티 경제적 역할 비교 】

┌─────────────────────────────────────┐
│  King County (일자리 중심지)         │
├─────────────────────────────────────┤
│  • 평균 면적: 1,914 sqft            │
│  • $/sqft: $498 (1.52배 비쌈)       │
│  • 다양한 주택 유형                   │
│  • "비싸지만 작은 도시형 주택"        │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  Pierce County (위성 주거지)         │
├─────────────────────────────────────┤
│  • 평균 면적: 2,040 sqft (+6.5%)    │
│  • $/sqft: $329                     │
│  • 단독주택 92% 집중                  │
│  • "저렴하고 넓은 교외형 주택"        │
└─────────────────────────────────────┘

→ 가설 검증: King은 직장 접근성 프리미엄,
   Pierce는 넓은 주거공간 제공
"""
ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='Malgun Gothic',
        bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

plt.tight_layout()
plt.savefig('시각화_3_카운티특성비교.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   저장 완료: 시각화_3_카운티특성비교.png")

# ============================================================================
# Figure 4: 위치 기반 분석
# ============================================================================
print("\n[Figure 4] 위치 기반 분석 생성 중...")

# 거리 계산 함수
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# 기준점
SEATTLE = (47.6062, -122.3321)
BELLEVUE = (47.6101, -122.2015)
TACOMA = (47.2529, -122.4443)

# 거리 계산
king_df['dist_seattle'] = haversine(king_df['LATITUDE'], king_df['LONGITUDE'], SEATTLE[0], SEATTLE[1])
king_df['dist_bellevue'] = haversine(king_df['LATITUDE'], king_df['LONGITUDE'], BELLEVUE[0], BELLEVUE[1])
pierce_df['dist_seattle'] = haversine(pierce_df['LATITUDE'], pierce_df['LONGITUDE'], SEATTLE[0], SEATTLE[1])
pierce_df['dist_tacoma'] = haversine(pierce_df['LATITUDE'], pierce_df['LONGITUDE'], TACOMA[0], TACOMA[1])

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Figure 4: 위치(위도/경도) 기반 가격 분석', fontsize=16, fontweight='bold')

# 4-1. King - 벨뷰 거리 vs 가격
ax = axes[0, 0]
scatter = ax.scatter(king_df['dist_bellevue'], king_df['PRICE']/1000000, 
                     c=king_df['PRICE']/1000000, cmap='Blues', alpha=0.6, s=40)
# 추세선
z = np.polyfit(king_df['dist_bellevue'], king_df['PRICE']/1000000, 1)
p = np.poly1d(z)
x_line = np.linspace(king_df['dist_bellevue'].min(), king_df['dist_bellevue'].max(), 100)
ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='추세선')
r = king_df['dist_bellevue'].corr(king_df['PRICE'])
ax.set_xlabel('벨뷰까지 거리 (km)', fontsize=11)
ax.set_ylabel('가격 (백만 달러)', fontsize=11)
ax.set_title(f'King: 벨뷰 거리 vs 가격\nr = {r:.3f} (가까울수록 비쌈)', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 4-2. Pierce - 시애틀 거리 vs 가격
ax = axes[0, 1]
scatter = ax.scatter(pierce_df['dist_seattle'], pierce_df['PRICE']/1000000,
                     c=pierce_df['PRICE']/1000000, cmap='Purples', alpha=0.6, s=40)
z = np.polyfit(pierce_df['dist_seattle'], pierce_df['PRICE']/1000000, 1)
p = np.poly1d(z)
x_line = np.linspace(pierce_df['dist_seattle'].min(), pierce_df['dist_seattle'].max(), 100)
ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='추세선')
r = pierce_df['dist_seattle'].corr(pierce_df['PRICE'])
ax.set_xlabel('시애틀까지 거리 (km)', fontsize=11)
ax.set_ylabel('가격 (백만 달러)', fontsize=11)
ax.set_title(f'Pierce: 시애틀 거리 vs 가격\nr = {r:.3f} (가까울수록 비쌈)', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 4-3. 위치 프리미엄 바 차트
ax = axes[0, 2]
# 벨뷰 프리미엄
bellevue_near = king_df[king_df['dist_bellevue'] <= 10]['PRICE'].mean()
bellevue_far = king_df[king_df['dist_bellevue'] > 10]['PRICE'].mean()
bellevue_premium = (bellevue_near / bellevue_far - 1) * 100

# 시애틀 접근성 프리미엄 (Pierce)
seattle_near = pierce_df[pierce_df['dist_seattle'] <= 45]['PRICE'].mean()
seattle_far = pierce_df[pierce_df['dist_seattle'] > 45]['PRICE'].mean()
seattle_premium = (seattle_near / seattle_far - 1) * 100

premiums = [bellevue_premium, seattle_premium]
labels = ['벨뷰 프리미엄\n(King, 10km 이내)', '시애틀 접근성\n(Pierce, 45km 이내)']
colors = ['#2E86AB', '#A23B72']
bars = ax.bar(labels, premiums, color=colors, edgecolor='white', linewidth=2)
ax.set_ylabel('가격 프리미엄 (%)', fontsize=11)
ax.set_title('위치 프리미엄 정량화', fontsize=12, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.5)
for bar, val in zip(bars, premiums):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'+{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(premiums) * 1.2)
ax.grid(axis='y', alpha=0.3)

# 4-4. 위치 지도 시각화 (산점도 맵)
ax = axes[1, 0]
scatter = ax.scatter(king_df['LONGITUDE'], king_df['LATITUDE'], 
                     c=king_df['PRICE']/1000000, cmap='YlOrRd', alpha=0.7, s=50)
ax.scatter(SEATTLE[1], SEATTLE[0], c='blue', s=200, marker='*', label='Seattle', edgecolors='white', linewidths=2)
ax.scatter(BELLEVUE[1], BELLEVUE[0], c='green', s=200, marker='*', label='Bellevue', edgecolors='white', linewidths=2)
ax.set_xlabel('경도', fontsize=11)
ax.set_ylabel('위도', fontsize=11)
ax.set_title('King County 가격 분포 지도', fontsize=12, fontweight='bold')
ax.legend(loc='lower left')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('가격 (백만$)')

# 4-5. Pierce 위치 지도
ax = axes[1, 1]
scatter = ax.scatter(pierce_df['LONGITUDE'], pierce_df['LATITUDE'],
                     c=pierce_df['PRICE']/1000000, cmap='YlOrRd', alpha=0.7, s=50)
ax.scatter(TACOMA[1], TACOMA[0], c='purple', s=200, marker='*', label='Tacoma', edgecolors='white', linewidths=2)
ax.set_xlabel('경도', fontsize=11)
ax.set_ylabel('위도', fontsize=11)
ax.set_title('Pierce County 가격 분포 지도', fontsize=12, fontweight='bold')
ax.legend(loc='lower left')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('가격 (백만$)')

# 4-6. 위치 분석 요약
ax = axes[1, 2]
ax.axis('off')
location_summary = """
【 위치 기반 분석 핵심 발견 】

1. King County
   • 시애틀 도심: 가격과 무관 (r = -0.07)
   • 벨뷰(테크허브): 강한 영향 (r = -0.28)
   → "진짜 고가 지역은 벨뷰(Eastside)"
   
   벨뷰 10km 이내: 평균 $1,163,934
   벨뷰 10km 이외: 평균 $872,094
   → 프리미엄: +33.5%

2. Pierce County
   • 시애틀 거리: 유의미 (r = -0.19)
   → "북쪽(King 방향)일수록 비쌈"
   
   시애틀 45km 이내: $700,787
   시애틀 45km 이외: $605,750
   → 프리미엄: +15.7%

【 결론 】
King: 벨뷰(MS, Amazon) 접근성이 핵심
Pierce: 시애틀 접근성(통근 거리)이 핵심
"""
ax.text(0.05, 0.95, location_summary, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='Malgun Gothic',
        bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

plt.tight_layout()
plt.savefig('시각화_4_위치분석.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   저장 완료: 시각화_4_위치분석.png")

# ============================================================================
# Figure 5: 선형 회귀 모델링 결과
# ============================================================================
print("\n[Figure 5] 선형 회귀 모델링 결과 생성 중...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Figure 5: 선형 회귀 모델링 결과', fontsize=16, fontweight='bold')

# 모델 학습
features = ['SQUARE FEET', 'BEDS', 'BATHS', 'YEAR BUILT']

# King County
X_king = king_df[features]
y_king = king_df['PRICE']
X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X_king, y_king, test_size=0.2, random_state=42)
model_king = LinearRegression()
model_king.fit(X_train_k, y_train_k)
y_pred_king = model_king.predict(X_test_k)
r2_king = model_king.score(X_test_k, y_test_k)

# Pierce County
X_pierce = pierce_df[features]
y_pierce = pierce_df['PRICE']
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_pierce, y_pierce, test_size=0.2, random_state=42)
model_pierce = LinearRegression()
model_pierce.fit(X_train_p, y_train_p)
y_pred_pierce = model_pierce.predict(X_test_p)
r2_pierce = model_pierce.score(X_test_p, y_test_p)

# 5-1. King 실제 vs 예측
ax = axes[0, 0]
ax.scatter(y_test_k/1000000, y_pred_king/1000000, alpha=0.6, c='#2E86AB', s=40)
ax.plot([0, 3], [0, 3], 'r--', linewidth=2, label='완벽 예측선')
ax.set_xlabel('실제 가격 (백만 달러)', fontsize=11)
ax.set_ylabel('예측 가격 (백만 달러)', fontsize=11)
ax.set_title(f'King County: 실제 vs 예측\nR² = {r2_king:.3f}', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)

# 5-2. Pierce 실제 vs 예측
ax = axes[0, 1]
ax.scatter(y_test_p/1000000, y_pred_pierce/1000000, alpha=0.6, c='#A23B72', s=40)
ax.plot([0, 2.5], [0, 2.5], 'r--', linewidth=2, label='완벽 예측선')
ax.set_xlabel('실제 가격 (백만 달러)', fontsize=11)
ax.set_ylabel('예측 가격 (백만 달러)', fontsize=11)
ax.set_title(f'Pierce County: 실제 vs 예측\nR² = {r2_pierce:.3f}', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 2.5)
ax.set_ylim(0, 2.5)

# 5-3. R² 비교
ax = axes[0, 2]
r2_values = [r2_king, r2_pierce]
labels = ['King County', 'Pierce County']
colors = ['#2E86AB', '#A23B72']
bars = ax.bar(labels, r2_values, color=colors, edgecolor='white', linewidth=2)
ax.set_ylabel('R² (결정계수)', fontsize=11)
ax.set_title('모델 설명력 비교', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='R²=0.5 기준선')
for bar, val in zip(bars, r2_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.1%}', ha='center', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 5-4. 회귀 계수 비교
ax = axes[1, 0]
coef_king = model_king.coef_
coef_pierce = model_pierce.coef_
x = np.arange(len(features))
width = 0.35
ax.bar(x - width/2, coef_king, width, label='King', color='#2E86AB')
ax.bar(x + width/2, coef_pierce, width, label='Pierce', color='#A23B72')
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=15)
ax.set_ylabel('회귀 계수', fontsize=11)
ax.set_title('회귀 계수 비교', fontsize=12, fontweight='bold')
ax.legend()
ax.axhline(0, color='black', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)

# 5-5. 특성 중요도 (절대값 기준)
ax = axes[1, 1]
# 표준화된 계수로 중요도 계산
importance_king = np.abs(coef_king * X_king.std().values) / (np.abs(coef_king * X_king.std().values).sum())
importance_pierce = np.abs(coef_pierce * X_pierce.std().values) / (np.abs(coef_pierce * X_pierce.std().values).sum())

x = np.arange(len(features))
ax.barh(x - width/2, importance_king * 100, width, label='King', color='#2E86AB')
ax.barh(x + width/2, importance_pierce * 100, width, label='Pierce', color='#A23B72')
ax.set_yticks(x)
ax.set_yticklabels(features)
ax.set_xlabel('상대적 중요도 (%)', fontsize=11)
ax.set_title('특성 중요도 비교', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# 5-6. 모델 결과 요약
ax = axes[1, 2]
ax.axis('off')
model_summary = f"""
【 선형 회귀 모델 결과 요약 】

┌─────────────────────────────────────┐
│  모델 수식                            │
├─────────────────────────────────────┤
│  PRICE = β₀ + β₁(SQFT) + β₂(BEDS)   │
│        + β₃(BATHS) + β₄(YEAR BUILT)  │
└─────────────────────────────────────┘

【 모델 성능 】
┌─────────────┬─────────┬─────────┐
│    지표      │  King   │ Pierce  │
├─────────────┼─────────┼─────────┤
│    R²       │  {r2_king:.1%}  │  {r2_pierce:.1%}  │
└─────────────┴─────────┴─────────┘

【 계수 해석 (King County) 】
• SQUARE FEET: +${coef_king[0]:.0f}/sqft
  → 100sqft 증가 시 ~${coef_king[0]*100:,.0f} 상승
  
• YEAR BUILT: +${coef_king[3]:.0f}/년
  → 10년 신축 시 ~${coef_king[3]*10:,.0f} 상승

【 핵심 결론 】
→ 건물 면적(SQUARE FEET)이 가격의
   가장 중요한 결정 요인
→ 모델로 가격의 44~54% 설명 가능
"""
ax.text(0.02, 0.98, model_summary, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='Malgun Gothic',
        bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

plt.tight_layout()
plt.savefig('시각화_5_회귀모델링.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   저장 완료: 시각화_5_회귀모델링.png")

# ============================================================================
# Figure 6: 최종 결론 인포그래픽
# ============================================================================
print("\n[Figure 6] 최종 결론 인포그래픽 생성 중...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle('King County vs Pierce County 부동산 분석 결론', fontsize=20, fontweight='bold', y=0.98)

# 전체 레이아웃
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 6-1. 핵심 질문 답변 1
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis('off')
q1 = """
【 Q1. 가격 결정 요인 】

1위: SQUARE FEET
     r = 0.73~0.75
     중요도 75%+

2위: BATHS (욕실)
     r = 0.53~0.61

3위: BEDS (침실)
     r = 0.34~0.50

4위: YEAR BUILT
     r = 0.13~0.23
"""
ax1.text(0.5, 0.5, q1, transform=ax1.transAxes, fontsize=11,
         ha='center', va='center', fontfamily='Malgun Gothic',
         bbox=dict(boxstyle='round', facecolor='#e3f2fd', edgecolor='#2196f3', linewidth=2))

# 6-2. 핵심 질문 답변 2
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')
q2 = """
【 Q2. 카운티 간 차이 】

King County (도심)
• 평균: $919,192
• 역할: 일자리 중심지
• 특징: 비싸고 작음
• 핵심: 벨뷰 접근성

Pierce County (교외)
• 평균: $638,716
• 역할: 위성 주거지
• 특징: 저렴하고 넓음
• 핵심: 시애틀 접근성

가격 프리미엄: +43.9%
"""
ax2.text(0.5, 0.5, q2, transform=ax2.transAxes, fontsize=10,
         ha='center', va='center', fontfamily='Malgun Gothic',
         bbox=dict(boxstyle='round', facecolor='#fff3e0', edgecolor='#ff9800', linewidth=2))

# 6-3. 핵심 질문 답변 3
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')
q3 = f"""
【 Q3. 회귀 모델 설명력 】

King County
R² = {r2_king:.1%}

Pierce County
R² = {r2_pierce:.1%}

→ 가격의 절반 정도
   설명 가능

→ 나머지는 위치 세부사항,
   주택 상태 등 미포함 요소
"""
ax3.text(0.5, 0.5, q3, transform=ax3.transAxes, fontsize=11,
         ha='center', va='center', fontfamily='Malgun Gothic',
         bbox=dict(boxstyle='round', facecolor='#e8f5e9', edgecolor='#4caf50', linewidth=2))

# 6-4. 가격 비교 바 차트
ax4 = fig.add_subplot(gs[1, 0])
categories = ['평균 가격\n(만$)', '$/sqft', '평균 면적\n(백sqft)']
king_norm = [king_df['PRICE'].mean()/10000, king_df['$/SQUARE FEET'].mean(), king_df['SQUARE FEET'].mean()/100]
pierce_norm = [pierce_df['PRICE'].mean()/10000, pierce_df['$/SQUARE FEET'].mean(), pierce_df['SQUARE FEET'].mean()/100]
x = np.arange(3)
width = 0.35
ax4.bar(x - width/2, king_norm, width, label='King', color='#2E86AB')
ax4.bar(x + width/2, pierce_norm, width, label='Pierce', color='#A23B72')
ax4.set_xticks(x)
ax4.set_xticklabels(categories)
ax4.set_title('주요 지표 비교', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# 6-5. 위치 프리미엄
ax5 = fig.add_subplot(gs[1, 1])
premiums = [33.5, 15.7]
labels = ['벨뷰\n프리미엄', '시애틀\n접근성']
colors = ['#2E86AB', '#A23B72']
bars = ax5.bar(labels, premiums, color=colors, edgecolor='white', linewidth=2)
ax5.set_ylabel('프리미엄 (%)', fontsize=11)
ax5.set_title('위치 프리미엄', fontsize=12, fontweight='bold')
for bar, val in zip(bars, premiums):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'+{val}%', ha='center', fontsize=12, fontweight='bold')
ax5.set_ylim(0, 45)
ax5.grid(axis='y', alpha=0.3)

# 6-6. R² 비교
ax6 = fig.add_subplot(gs[1, 2])
sizes = [r2_king*100, r2_pierce*100]
colors = ['#2E86AB', '#A23B72']
labels = [f'King\n{r2_king:.1%}', f'Pierce\n{r2_pierce:.1%}']
ax6.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90,
        explode=[0.05, 0.05], textprops={'fontsize': 11, 'fontweight': 'bold'})
ax6.set_title('모델 R² 비교', fontsize=12, fontweight='bold')

# 6-7. 시사점 & 결론
ax7 = fig.add_subplot(gs[2, :])
ax7.axis('off')
conclusion = """
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           【 투자/구매 시사점 】                                            │
├────────────────────────────────────────────────┬─────────────────────────────────────────────────────────┤
│              King County 선택                    │               Pierce County 선택                        │
├────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│  • 테크 기업 종사자 (통근 시간 최소화)              │  • 넓은 주거 공간 필요 (가족)                             │
│  • 도시 생활 선호 (다양한 주택 유형)               │  • 예산 제약 (동일 금액으로 더 넓은 집)                     │
│  • 위치 중심 투자 (벨뷰 인근 프리미엄)              │  • 건물 크기 중심 투자 (면적당 가격 유리)                   │
└────────────────────────────────────────────────┴─────────────────────────────────────────────────────────┘

                    【 핵심 결론 】  건물 크기(SQUARE FEET)가 가격의 가장 중요한 결정 요인이며,
                                    위치(벨뷰/시애틀 접근성)가 두 번째로 중요한 요인이다.
"""
ax7.text(0.5, 0.5, conclusion, transform=ax7.transAxes, fontsize=11,
         ha='center', va='center', fontfamily='Malgun Gothic',
         bbox=dict(boxstyle='round', facecolor='#fafafa', edgecolor='#333', linewidth=2))

plt.savefig('시각화_6_최종결론.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   저장 완료: 시각화_6_최종결론.png")

# ============================================================================
# Figure 7: 도시별 가격 비교 (Top 10)
# ============================================================================
print("\n[Figure 7] 도시별 가격 비교 생성 중...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Figure 7: 도시별 평균 가격 Top 10', fontsize=16, fontweight='bold')

# King County Top 10
ax = axes[0]
king_city = king_df.groupby('CITY').agg({'PRICE': ['mean', 'count']}).reset_index()
king_city.columns = ['CITY', 'avg_price', 'count']
king_city = king_city[king_city['count'] >= 2].nlargest(10, 'avg_price')

colors = plt.cm.Blues(np.linspace(0.4, 0.9, 10))[::-1]
bars = ax.barh(king_city['CITY'], king_city['avg_price']/1000000, color=colors, edgecolor='white')
ax.set_xlabel('평균 가격 (백만 달러)', fontsize=11)
ax.set_title('King County Top 10 도시', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for bar, (_, row) in zip(bars, king_city.iterrows()):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f'${row["avg_price"]/1000000:.2f}M ({int(row["count"])}건)', 
            va='center', fontsize=9)
ax.set_xlim(0, king_city['avg_price'].max()/1000000 * 1.3)
ax.grid(axis='x', alpha=0.3)

# Pierce County Top 10
ax = axes[1]
pierce_city = pierce_df.groupby('CITY').agg({'PRICE': ['mean', 'count']}).reset_index()
pierce_city.columns = ['CITY', 'avg_price', 'count']
pierce_city = pierce_city[pierce_city['count'] >= 2].nlargest(10, 'avg_price')

colors = plt.cm.Purples(np.linspace(0.4, 0.9, 10))[::-1]
bars = ax.barh(pierce_city['CITY'], pierce_city['avg_price']/1000000, color=colors, edgecolor='white')
ax.set_xlabel('평균 가격 (백만 달러)', fontsize=11)
ax.set_title('Pierce County Top 10 도시', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for bar, (_, row) in zip(bars, pierce_city.iterrows()):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f'${row["avg_price"]/1000000:.2f}M ({int(row["count"])}건)',
            va='center', fontsize=9)
ax.set_xlim(0, pierce_city['avg_price'].max()/1000000 * 1.3)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('시각화_7_도시별가격.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   저장 완료: 시각화_7_도시별가격.png")

# ============================================================================
# Figure 8: 가격대별 시장 분포
# ============================================================================
print("\n[Figure 8] 가격대별 시장 분포 생성 중...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Figure 8: 가격대별 시장 분포', fontsize=16, fontweight='bold')

# 가격대 구간
bins = [0, 400000, 600000, 800000, 1000000, 1500000, 2000000, float('inf')]
labels = ['~$400K', '$400-600K', '$600-800K', '$800K-1M', '$1-1.5M', '$1.5-2M', '$2M+']

king_segments = pd.cut(king_df['PRICE'], bins=bins, labels=labels).value_counts().sort_index()
pierce_segments = pd.cut(pierce_df['PRICE'], bins=bins, labels=labels).value_counts().sort_index()

# King County
ax = axes[0]
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(labels)))
wedges, texts, autotexts = ax.pie(king_segments.values, labels=labels, autopct='%1.1f%%',
                                   colors=colors, explode=[0.02]*len(labels))
ax.set_title('King County 가격대 분포', fontsize=12, fontweight='bold')

# Pierce County
ax = axes[1]
colors = plt.cm.Purples(np.linspace(0.3, 0.9, len(labels)))
wedges, texts, autotexts = ax.pie(pierce_segments.values, labels=labels, autopct='%1.1f%%',
                                   colors=colors, explode=[0.02]*len(labels))
ax.set_title('Pierce County 가격대 분포', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('시각화_8_가격대분포.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   저장 완료: 시각화_8_가격대분포.png")

# ============================================================================
# 완료 메시지
# ============================================================================
print("\n" + "="*60)
print("【 시각화 자료 생성 완료 】")
print("="*60)
print("""
생성된 파일 목록:
1. 시각화_1_가격분포비교.png    - 히스토그램, 박스플롯, 통계표
2. 시각화_2_상관관계분석.png    - 산점도, 상관계수 비교
3. 시각화_3_카운티특성비교.png  - 건물속성, 주택유형, 경제기능
4. 시각화_4_위치분석.png        - 거리-가격, 위치프리미엄, 지도
5. 시각화_5_회귀모델링.png      - 실제vs예측, 계수, R²
6. 시각화_6_최종결론.png        - 핵심질문 답변, 시사점
7. 시각화_7_도시별가격.png      - Top 10 도시 비교
8. 시각화_8_가격대분포.png      - 가격대별 파이차트

총 8개 이미지 파일이 생성되었습니다.
발표 슬라이드에 활용하세요!
""")
