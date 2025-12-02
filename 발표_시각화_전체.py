# -*- coding: utf-8 -*-
"""
발표용 시각화 자료 생성 스크립트
프로젝트_개요.md 기반 14개 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 색상 설정
KING_COLOR = '#2E86AB'
PIERCE_COLOR = '#A23B72'
ACCENT_COLOR = '#F6AE2D'

# =============================================================================
# 데이터 로드 및 전처리
# =============================================================================
def load_and_preprocess():
    """데이터 로드 및 전처리"""
    king = pd.read_csv('King_County_Sold.csv')
    pierce = pd.read_csv('Pierce_County_Sold.csv')
    
    # 필요 칼럼
    cols = ['PROPERTY TYPE', 'CITY', 'PRICE', 'BEDS', 'BATHS', 
            'SQUARE FEET', 'YEAR BUILT', '$/SQUARE FEET', 'LATITUDE', 'LONGITUDE']
    
    king = king[cols].dropna()
    pierce = pierce[cols].dropna()
    
    # 주거용 필터링
    property_types = ['Single Family Residential', 'Townhouse', 'Condo/Co-op']
    king = king[king['PROPERTY TYPE'].isin(property_types)]
    pierce = pierce[pierce['PROPERTY TYPE'].isin(property_types)]
    
    # 이상치 제거
    king = king[king['PRICE'] <= 5000000]
    pierce = pierce[pierce['PRICE'] <= 4000000]
    
    # 카운티 라벨 추가
    king['COUNTY'] = 'King'
    pierce['COUNTY'] = 'Pierce'
    
    # 거리 계산 함수 (Haversine)
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
    
    for df in [king, pierce]:
        df['dist_seattle'] = haversine(df['LATITUDE'], df['LONGITUDE'], SEATTLE[0], SEATTLE[1])
        df['dist_bellevue'] = haversine(df['LATITUDE'], df['LONGITUDE'], BELLEVUE[0], BELLEVUE[1])
    
    combined = pd.concat([king, pierce], ignore_index=True)
    
    print(f"King County: {len(king)}건, Pierce County: {len(pierce)}건")
    return king, pierce, combined

# =============================================================================
# (1) 분석 파이프라인 다이어그램
# =============================================================================
def fig1_pipeline():
    """분석 파이프라인 다이어그램"""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    steps = [
        ('1. 데이터 수집', 'Redfin\nKing/Pierce'),
        ('2. 전처리', '27→10 칼럼\n667건'),
        ('3. EDA', '기술통계\n분포분석'),
        ('4. 통계분석', 't-test\n상관분석'),
        ('5. 위치분석', 'Haversine\n프리미엄'),
        ('6. 회귀모델', 'OLS\nRandom Forest'),
        ('7. 결론', '면적 1순위\n위치 2순위')
    ]
    
    colors = ['#E8F4F8', '#D1E8E4', '#B8DED8', '#9FD4CC', '#86CAC0', '#6DC0B4', '#54B6A8']
    
    for i, ((title, desc), color) in enumerate(zip(steps, colors)):
        x = i * 2 + 0.5
        # 박스
        box = FancyBboxPatch((x, 1), 1.6, 2, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='#2E86AB', linewidth=2)
        ax.add_patch(box)
        # 텍스트
        ax.text(x + 0.8, 2.5, title, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x + 0.8, 1.7, desc, ha='center', va='center', fontsize=8, color='#333')
        # 화살표
        if i < len(steps) - 1:
            ax.annotate('', xy=(x + 2.1, 2), xytext=(x + 1.7, 2),
                       arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=2))
    
    ax.set_title('【 프로젝트 분석 파이프라인 】', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('슬라이드_01_파이프라인.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[1/14] 파이프라인 다이어그램 저장 완료")

# =============================================================================
# (2) 두 카운티 기본 특성 비교 막대그래프
# =============================================================================
def fig2_basic_comparison(king, pierce):
    """기본 특성 비교 그룹 막대그래프"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    metrics = [
        ('평균 가격 ($)', king['PRICE'].mean(), pierce['PRICE'].mean()),
        ('평균 면적 (sqft)', king['SQUARE FEET'].mean(), pierce['SQUARE FEET'].mean()),
        ('$/sqft', king['$/SQUARE FEET'].mean(), pierce['$/SQUARE FEET'].mean()),
        ('평균 침실 (개)', king['BEDS'].mean(), pierce['BEDS'].mean()),
        ('평균 욕실 (개)', king['BATHS'].mean(), pierce['BATHS'].mean()),
        ('평균 건축연도', king['YEAR BUILT'].mean(), pierce['YEAR BUILT'].mean())
    ]
    
    for ax, (label, king_val, pierce_val) in zip(axes.flatten(), metrics):
        x = np.arange(2)
        bars = ax.bar(x, [king_val, pierce_val], color=[KING_COLOR, PIERCE_COLOR], width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(['King', 'Pierce'])
        ax.set_ylabel(label)
        ax.set_title(label, fontweight='bold')
        
        # 값 표시
        for bar, val in zip(bars, [king_val, pierce_val]):
            if '가격' in label:
                text = f'${val/1000:.0f}K'
            elif '연도' in label:
                text = f'{val:.0f}'
            elif 'sqft' in label.lower() and '면적' not in label:
                text = f'${val:.0f}'
            elif '면적' in label:
                text = f'{val:.0f}'
            else:
                text = f'{val:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.02,
                   text, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    fig.suptitle('King County vs Pierce County 기본 특성 비교\n"King: 비싸고 작다 | Pierce: 넓고 싸다"',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('슬라이드_02_기본특성비교.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[2/14] 기본 특성 비교 저장 완료")

# =============================================================================
# (3) 가격 분포 히스토그램
# =============================================================================
def fig3_price_histogram(king, pierce):
    """가격 분포 히스토그램"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 일반 가격 히스토그램
    ax1 = axes[0]
    ax1.hist(king['PRICE']/1e6, bins=20, alpha=0.7, color=KING_COLOR, label='King', edgecolor='white')
    ax1.hist(pierce['PRICE']/1e6, bins=20, alpha=0.7, color=PIERCE_COLOR, label='Pierce', edgecolor='white')
    ax1.set_xlabel('가격 (백만 달러)', fontsize=11)
    ax1.set_ylabel('빈도', fontsize=11)
    ax1.set_title('가격 분포 비교', fontweight='bold')
    ax1.legend()
    ax1.axvline(king['PRICE'].mean()/1e6, color=KING_COLOR, linestyle='--', lw=2, label=f'King 평균: ${king["PRICE"].mean()/1e6:.2f}M')
    ax1.axvline(pierce['PRICE'].mean()/1e6, color=PIERCE_COLOR, linestyle='--', lw=2, label=f'Pierce 평균: ${pierce["PRICE"].mean()/1e6:.2f}M')
    
    # Log 변환 히스토그램
    ax2 = axes[1]
    ax2.hist(np.log10(king['PRICE']), bins=20, alpha=0.7, color=KING_COLOR, label='King', edgecolor='white')
    ax2.hist(np.log10(pierce['PRICE']), bins=20, alpha=0.7, color=PIERCE_COLOR, label='Pierce', edgecolor='white')
    ax2.set_xlabel('log₁₀(가격)', fontsize=11)
    ax2.set_ylabel('빈도', fontsize=11)
    ax2.set_title('Log 변환 가격 분포 (정규성 개선)', fontweight='bold')
    ax2.legend()
    
    fig.suptitle('Q1: 두 카운티의 가격 분포는 어떻게 다른가?\n"King: $60만~100만 분산 | Pierce: $40만~60만 집중"',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('슬라이드_03_가격히스토그램.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[3/14] 가격 히스토그램 저장 완료")

# =============================================================================
# (4) 가격 박스플롯
# =============================================================================
def fig4_price_boxplot(combined):
    """카운티별 가격 박스플롯"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 일반 가격
    ax1 = axes[0]
    bp1 = ax1.boxplot([combined[combined['COUNTY']=='King']['PRICE']/1e6,
                       combined[combined['COUNTY']=='Pierce']['PRICE']/1e6],
                      labels=['King', 'Pierce'], patch_artist=True)
    bp1['boxes'][0].set_facecolor(KING_COLOR)
    bp1['boxes'][1].set_facecolor(PIERCE_COLOR)
    ax1.set_ylabel('가격 (백만 달러)', fontsize=11)
    ax1.set_title('가격 분포 박스플롯', fontweight='bold')
    
    # 통계값 표시
    king_med = combined[combined['COUNTY']=='King']['PRICE'].median()/1e6
    pierce_med = combined[combined['COUNTY']=='Pierce']['PRICE'].median()/1e6
    ax1.text(1, king_med + 0.1, f'중앙값: ${king_med:.2f}M', ha='center', fontsize=9)
    ax1.text(2, pierce_med + 0.1, f'중앙값: ${pierce_med:.2f}M', ha='center', fontsize=9)
    
    # Log 가격
    ax2 = axes[1]
    bp2 = ax2.boxplot([np.log10(combined[combined['COUNTY']=='King']['PRICE']),
                       np.log10(combined[combined['COUNTY']=='Pierce']['PRICE'])],
                      labels=['King', 'Pierce'], patch_artist=True)
    bp2['boxes'][0].set_facecolor(KING_COLOR)
    bp2['boxes'][1].set_facecolor(PIERCE_COLOR)
    ax2.set_ylabel('log₁₀(가격)', fontsize=11)
    ax2.set_title('Log 가격 박스플롯', fontweight='bold')
    
    fig.suptitle('카운티별 가격 분포 비교\n"King이 43.9% 더 비싸고, 전체 분포가 위로 이동"',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('슬라이드_04_가격박스플롯.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[4/14] 가격 박스플롯 저장 완료")

# =============================================================================
# (5) 상관계수 히트맵
# =============================================================================
def fig5_correlation_heatmap(combined):
    """상관계수 히트맵"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    numeric_cols = ['PRICE', 'SQUARE FEET', 'BEDS', 'BATHS', 'YEAR BUILT', '$/SQUARE FEET']
    
    for ax, (county, color) in zip(axes, [('King', KING_COLOR), ('Pierce', PIERCE_COLOR)]):
        data = combined[combined['COUNTY'] == county][numeric_cols]
        corr = data.corr()
        
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, vmin=-1, vmax=1, ax=ax, square=True,
                   cbar_kws={'shrink': 0.8})
        ax.set_title(f'{county} County 상관관계', fontweight='bold')
    
    fig.suptitle('Q2: 어떤 요인이 가격을 가장 많이 움직이는가?\n"PRICE와 SQUARE FEET: r ≈ 0.75 (가장 강한 상관)"',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('슬라이드_05_상관히트맵.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[5/14] 상관 히트맵 저장 완료")

# =============================================================================
# (6) 면적 vs 가격 산점도
# =============================================================================
def fig6_sqft_price_scatter(king, pierce):
    """면적 vs 가격 산점도 + 회귀선"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 산점도
    ax.scatter(king['SQUARE FEET'], king['PRICE']/1e6, alpha=0.5, c=KING_COLOR, 
               label='King', s=30, edgecolors='white', linewidth=0.5)
    ax.scatter(pierce['SQUARE FEET'], pierce['PRICE']/1e6, alpha=0.5, c=PIERCE_COLOR, 
               label='Pierce', s=30, edgecolors='white', linewidth=0.5)
    
    # 회귀선 (King)
    X_king = king['SQUARE FEET'].values.reshape(-1, 1)
    y_king = king['PRICE'].values / 1e6
    reg_king = LinearRegression().fit(X_king, y_king)
    x_line = np.linspace(500, 5000, 100).reshape(-1, 1)
    ax.plot(x_line, reg_king.predict(x_line), color=KING_COLOR, lw=2, linestyle='--',
           label=f'King 회귀선 (r={np.corrcoef(king["SQUARE FEET"], king["PRICE"])[0,1]:.2f})')
    
    # 회귀선 (Pierce)
    X_pierce = pierce['SQUARE FEET'].values.reshape(-1, 1)
    y_pierce = pierce['PRICE'].values / 1e6
    reg_pierce = LinearRegression().fit(X_pierce, y_pierce)
    ax.plot(x_line, reg_pierce.predict(x_line), color=PIERCE_COLOR, lw=2, linestyle='--',
           label=f'Pierce 회귀선 (r={np.corrcoef(pierce["SQUARE FEET"], pierce["PRICE"])[0,1]:.2f})')
    
    ax.set_xlabel('면적 (SQUARE FEET)', fontsize=12)
    ax.set_ylabel('가격 (백만 달러)', fontsize=12)
    ax.legend(loc='upper left')
    ax.set_title('면적 vs 가격: 1순위 가격 결정 요인\n"면적이 커질수록 가격이 증가 → 상관계수 r ≈ 0.75"',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('슬라이드_06_면적가격산점도.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[6/14] 면적-가격 산점도 저장 완료")

# =============================================================================
# (7) Random Forest Feature Importance
# =============================================================================
def fig7_feature_importance(combined):
    """Random Forest 변수 중요도"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = ['SQUARE FEET', 'BEDS', 'BATHS', 'YEAR BUILT', 'dist_seattle', 'dist_bellevue']
    X = combined[features].copy()
    y = combined['PRICE']
    
    # 결측치 처리
    X = X.fillna(X.mean())
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importance = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True)
    
    colors = [ACCENT_COLOR if imp > 0.5 else KING_COLOR for imp in importance['importance']]
    bars = ax.barh(importance['feature'], importance['importance'], color=colors, edgecolor='white')
    
    ax.set_xlabel('변수 중요도 (Feature Importance)', fontsize=11)
    ax.set_title('Random Forest 변수 중요도 분석\n"면적(SQUARE FEET)이 전체 중요도의 75% 이상 차지"',
                fontsize=13, fontweight='bold')
    
    # 값 표시
    for bar, val in zip(bars, importance['importance']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val*100:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('슬라이드_07_변수중요도.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[7/14] 변수 중요도 저장 완료")

# =============================================================================
# (8) 지도 시각화 (정적 scatter)
# =============================================================================
def fig8_map_visualization(king, pierce):
    """지도 위 매물 위치 시각화 (정적)"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    combined = pd.concat([king, pierce])
    
    # 가격 구간별 색상
    price_bins = [0, 500000, 750000, 1000000, 1500000, np.inf]
    price_labels = ['<$500K', '$500K-750K', '$750K-1M', '$1M-1.5M', '>$1.5M']
    combined['price_cat'] = pd.cut(combined['PRICE'], bins=price_bins, labels=price_labels)
    
    colors_map = {'<$500K': '#2166AC', '$500K-750K': '#67A9CF', '$750K-1M': '#FDDBC7',
                  '$1M-1.5M': '#EF8A62', '>$1.5M': '#B2182B'}
    
    for cat in price_labels:
        subset = combined[combined['price_cat'] == cat]
        ax.scatter(subset['LONGITUDE'], subset['LATITUDE'], 
                  c=colors_map[cat], label=cat, alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
    
    # 기준점 표시
    ax.scatter(-122.3321, 47.6062, c='black', s=200, marker='*', label='Seattle', zorder=5)
    ax.scatter(-122.2015, 47.6101, c='red', s=200, marker='*', label='Bellevue', zorder=5)
    
    ax.set_xlabel('경도 (Longitude)', fontsize=11)
    ax.set_ylabel('위도 (Latitude)', fontsize=11)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_title('Q3: 위치가 가격에 미치는 영향\n"고가 매물이 Bellevue/Seattle 인근에 집중"',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('슬라이드_08_지도시각화.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[8/14] 지도 시각화 저장 완료")

# =============================================================================
# (9) 벨뷰 거리 vs 가격 산점도
# =============================================================================
def fig9_bellevue_distance(king):
    """벨뷰 거리 vs 가격 (King County)"""
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # 10km 기준 색상 구분
    within_10km = king['dist_bellevue'] <= 10
    
    ax.scatter(king[~within_10km]['dist_bellevue'], king[~within_10km]['PRICE']/1e6,
              alpha=0.5, c=KING_COLOR, label='10km 초과', s=40, edgecolors='white')
    ax.scatter(king[within_10km]['dist_bellevue'], king[within_10km]['PRICE']/1e6,
              alpha=0.7, c=ACCENT_COLOR, label='10km 이내', s=50, edgecolors='white')
    
    # 10km 수직선
    ax.axvline(10, color='red', linestyle='--', lw=2, label='10km 기준선')
    
    # 평균 가격 표시
    avg_within = king[within_10km]['PRICE'].mean()
    avg_outside = king[~within_10km]['PRICE'].mean()
    premium = (avg_within - avg_outside) / avg_outside * 100
    
    ax.text(5, 2.5, f'10km 이내 평균: ${avg_within/1e6:.2f}M\n({len(king[within_10km])}건)',
           fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax.text(25, 2.5, f'10km 초과 평균: ${avg_outside/1e6:.2f}M\n({len(king[~within_10km])}건)',
           fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    ax.set_xlabel('벨뷰까지 거리 (km)', fontsize=11)
    ax.set_ylabel('가격 (백만 달러)', fontsize=11)
    ax.legend()
    ax.set_title(f'벨뷰 프리미엄 분석 (King County)\n"10km 이내 평균 가격 +{premium:.1f}% 프리미엄"',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('슬라이드_09_벨뷰프리미엄.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[9/14] 벨뷰 프리미엄 저장 완료")

# =============================================================================
# (10) 시애틀 거리 vs 가격 (Pierce)
# =============================================================================
def fig10_seattle_distance(pierce):
    """시애틀 거리 vs 가격 (Pierce County)"""
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # 회귀선 추가
    X = pierce['dist_seattle'].values.reshape(-1, 1)
    y = pierce['PRICE'].values / 1e6
    reg = LinearRegression().fit(X, y)
    
    ax.scatter(pierce['dist_seattle'], pierce['PRICE']/1e6, alpha=0.5, c=PIERCE_COLOR, 
               s=40, edgecolors='white')
    
    x_line = np.linspace(pierce['dist_seattle'].min(), pierce['dist_seattle'].max(), 100).reshape(-1, 1)
    ax.plot(x_line, reg.predict(x_line), color='red', lw=2, linestyle='--',
           label=f'회귀선 (r={np.corrcoef(pierce["dist_seattle"], pierce["PRICE"])[0,1]:.3f})')
    
    ax.set_xlabel('시애틀까지 거리 (km)', fontsize=11)
    ax.set_ylabel('가격 (백만 달러)', fontsize=11)
    ax.legend()
    ax.set_title('시애틀 접근성 분석 (Pierce County)\n"시애틀과 가까울수록 가격 상승 → 위성 주거지 특성"',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('슬라이드_10_시애틀접근성.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[10/14] 시애틀 접근성 저장 완료")

# =============================================================================
# (11) 실제값 vs 예측값 산점도
# =============================================================================
def fig11_actual_vs_predicted(combined):
    """실제값 vs 예측값 산점도"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    features = ['SQUARE FEET', 'BEDS', 'BATHS', 'YEAR BUILT']
    
    for ax, county, color in zip(axes, ['King', 'Pierce'], [KING_COLOR, PIERCE_COLOR]):
        data = combined[combined['COUNTY'] == county].copy()
        X = data[features]
        y = data['PRICE'] / 1e6
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        
        ax.scatter(y_test, y_pred, alpha=0.6, c=color, s=40, edgecolors='white')
        
        # 대각선
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', lw=2, label='y=x (완벽한 예측)')
        
        ax.set_xlabel('실제 가격 (백만 달러)', fontsize=11)
        ax.set_ylabel('예측 가격 (백만 달러)', fontsize=11)
        ax.set_title(f'{county} County (R² = {r2:.2f})', fontweight='bold')
        ax.legend()
    
    fig.suptitle('회귀 모델 성능: 실제값 vs 예측값\n"대각선에 가까울수록 예측 정확도 높음"',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('슬라이드_11_실제vs예측.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[11/14] 실제vs예측 저장 완료")

# =============================================================================
# (12) 모델 성능 비교 막대그래프
# =============================================================================
def fig12_model_comparison(combined):
    """모델 성능 비교 (R² / Adj R²)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 모델 성능 데이터 (프로젝트_개요.md 기준)
    models = ['King\n기본', 'King\n위치포함', 'Pierce\n기본', 'Pierce\n위치포함']
    r2_scores = [0.54, 0.73, 0.44, 0.66]
    adj_r2_scores = [0.52, 0.71, 0.42, 0.64]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, r2_scores, width, label='R²', color=KING_COLOR, edgecolor='white')
    bars2 = ax.bar(x + width/2, adj_r2_scores, width, label='Adjusted R²', color=PIERCE_COLOR, edgecolor='white')
    
    ax.set_ylabel('결정계수 (R²)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 개선폭 화살표
    ax.annotate('', xy=(1, 0.73), xytext=(0, 0.54),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.5, 0.65, '+19%p', fontsize=11, color='red', fontweight='bold')
    
    ax.annotate('', xy=(3, 0.66), xytext=(2, 0.44),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(2.5, 0.56, '+22%p', fontsize=11, color='red', fontweight='bold')
    
    ax.set_title('모델 성능 비교: 위치 변수 추가 효과\n"위치 변수 추가 시 R² +20%p 향상 → 위치가 핵심 요인"',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('슬라이드_12_모델성능비교.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[12/14] 모델 성능 비교 저장 완료")

# =============================================================================
# (13) 결론 인포그래픽
# =============================================================================
def fig13_conclusion_infographic():
    """결론 인포그래픽"""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # 제목
    ax.text(7, 6.5, '🏠 가격 결정 공식', fontsize=20, ha='center', fontweight='bold')
    ax.text(7, 5.8, 'PRICE ≈ 면적 × 위치 프리미엄', fontsize=16, ha='center', 
           style='italic', color='#333')
    
    # 박스 1: 면적
    box1 = FancyBboxPatch((1, 2.5), 3.5, 2.5, boxstyle="round,pad=0.1",
                          facecolor='#E3F2FD', edgecolor=KING_COLOR, linewidth=3)
    ax.add_patch(box1)
    ax.text(2.75, 4.3, '📐 면적 (1순위)', fontsize=14, ha='center', fontweight='bold')
    ax.text(2.75, 3.5, 'r = 0.75', fontsize=12, ha='center')
    ax.text(2.75, 3.0, '중요도 75%+', fontsize=12, ha='center')
    
    # 박스 2: 위치
    box2 = FancyBboxPatch((5.25, 2.5), 3.5, 2.5, boxstyle="round,pad=0.1",
                          facecolor='#FFF3E0', edgecolor=ACCENT_COLOR, linewidth=3)
    ax.add_patch(box2)
    ax.text(7, 4.3, '📍 위치 (2순위)', fontsize=14, ha='center', fontweight='bold')
    ax.text(7, 3.5, '벨뷰 +33.5%', fontsize=12, ha='center')
    ax.text(7, 3.0, '시애틀 +15.7%', fontsize=12, ha='center')
    
    # 박스 3: 결과
    box3 = FancyBboxPatch((9.5, 2.5), 3.5, 2.5, boxstyle="round,pad=0.1",
                          facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=3)
    ax.add_patch(box3)
    ax.text(11.25, 4.3, '💰 가격', fontsize=14, ha='center', fontweight='bold')
    ax.text(11.25, 3.5, 'King: $0.92M', fontsize=12, ha='center', color=KING_COLOR)
    ax.text(11.25, 3.0, 'Pierce: $0.64M', fontsize=12, ha='center', color=PIERCE_COLOR)
    
    # 화살표
    ax.annotate('', xy=(5.1, 3.75), xytext=(4.6, 3.75),
               arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    ax.annotate('', xy=(9.35, 3.75), xytext=(8.85, 3.75),
               arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # 하단 요약
    ax.text(7, 1.3, '✓ King County: 테크 허브 접근성 프리미엄, 소형·고가 도시형 주택', 
           fontsize=11, ha='center')
    ax.text(7, 0.7, '✓ Pierce County: 위성 주거지, 대형·저가 가족형 주택', 
           fontsize=11, ha='center')
    
    plt.tight_layout()
    plt.savefig('슬라이드_13_결론인포그래픽.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[13/14] 결론 인포그래픽 저장 완료")

# =============================================================================
# (14) 투자 관점 비교 막대그래프
# =============================================================================
def fig14_investment_comparison(king, pierce):
    """투자 관점 비교: $80만 예산 시 면적 비교"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 그래프 1: $80만으로 얻을 수 있는 면적
    ax1 = axes[0]
    
    # $/sqft 기준 계산
    king_sqft_per_dollar = 800000 / king['$/SQUARE FEET'].mean()
    pierce_sqft_per_dollar = 800000 / pierce['$/SQUARE FEET'].mean()
    
    bars = ax1.bar(['King', 'Pierce'], [king_sqft_per_dollar, pierce_sqft_per_dollar],
                  color=[KING_COLOR, PIERCE_COLOR], edgecolor='white', width=0.5)
    
    ax1.set_ylabel('면적 (sqft)', fontsize=11)
    ax1.set_title('$800,000 예산으로\n구매 가능한 평균 면적', fontweight='bold')
    
    for bar, val in zip(bars, [king_sqft_per_dollar, pierce_sqft_per_dollar]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{val:.0f} sqft', ha='center', fontsize=12, fontweight='bold')
    
    diff = pierce_sqft_per_dollar - king_sqft_per_dollar
    ax1.text(0.5, max(king_sqft_per_dollar, pierce_sqft_per_dollar) * 0.5,
            f'+{diff:.0f} sqft\n(+{diff/king_sqft_per_dollar*100:.0f}%)',
            ha='center', fontsize=11, color='red', fontweight='bold')
    
    # 그래프 2: 카운티별 선택 가이드
    ax2 = axes[1]
    ax2.axis('off')
    
    # King 박스
    king_box = FancyBboxPatch((0.05, 0.5), 0.4, 0.45, boxstyle="round,pad=0.02",
                              facecolor='#E3F2FD', edgecolor=KING_COLOR, linewidth=2,
                              transform=ax2.transAxes)
    ax2.add_patch(king_box)
    ax2.text(0.25, 0.85, 'King County 추천', transform=ax2.transAxes,
            ha='center', fontsize=12, fontweight='bold', color=KING_COLOR)
    ax2.text(0.25, 0.72, '• 테크 기업 종사자', transform=ax2.transAxes, ha='center', fontsize=10)
    ax2.text(0.25, 0.62, '• 도시 생활 선호', transform=ax2.transAxes, ha='center', fontsize=10)
    ax2.text(0.25, 0.52, '• 위치 프리미엄 투자', transform=ax2.transAxes, ha='center', fontsize=10)
    
    # Pierce 박스
    pierce_box = FancyBboxPatch((0.55, 0.5), 0.4, 0.45, boxstyle="round,pad=0.02",
                                facecolor='#FCE4EC', edgecolor=PIERCE_COLOR, linewidth=2,
                                transform=ax2.transAxes)
    ax2.add_patch(pierce_box)
    ax2.text(0.75, 0.85, 'Pierce County 추천', transform=ax2.transAxes,
            ha='center', fontsize=12, fontweight='bold', color=PIERCE_COLOR)
    ax2.text(0.75, 0.72, '• 가족 단위 거주', transform=ax2.transAxes, ha='center', fontsize=10)
    ax2.text(0.75, 0.62, '• 넓은 공간 필요', transform=ax2.transAxes, ha='center', fontsize=10)
    ax2.text(0.75, 0.52, '• 예산 효율성 중시', transform=ax2.transAxes, ha='center', fontsize=10)
    
    ax2.text(0.5, 0.25, '"같은 예산으로 Pierce에서\n더 넓은 집을 살 수 있다"',
            transform=ax2.transAxes, ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig.suptitle('투자 관점 비교: 예산 고정 시 면적 차이', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('슬라이드_14_투자비교.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[14/14] 투자 비교 저장 완료")

# =============================================================================
# 메인 실행
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("발표용 시각화 자료 생성 시작")
    print("=" * 60)
    
    # 데이터 로드
    king, pierce, combined = load_and_preprocess()
    
    print("\n시각화 생성 중...")
    print("-" * 60)
    
    # 14개 시각화 생성
    fig1_pipeline()
    fig2_basic_comparison(king, pierce)
    fig3_price_histogram(king, pierce)
    fig4_price_boxplot(combined)
    fig5_correlation_heatmap(combined)
    fig6_sqft_price_scatter(king, pierce)
    fig7_feature_importance(combined)
    fig8_map_visualization(king, pierce)
    fig9_bellevue_distance(king)
    fig10_seattle_distance(pierce)
    fig11_actual_vs_predicted(combined)
    fig12_model_comparison(combined)
    fig13_conclusion_infographic()
    fig14_investment_comparison(king, pierce)
    
    print("-" * 60)
    print("\n✅ 모든 시각화 생성 완료!")
    print("\n생성된 파일 목록:")
    print("  슬라이드_01_파이프라인.png")
    print("  슬라이드_02_기본특성비교.png")
    print("  슬라이드_03_가격히스토그램.png")
    print("  슬라이드_04_가격박스플롯.png")
    print("  슬라이드_05_상관히트맵.png")
    print("  슬라이드_06_면적가격산점도.png")
    print("  슬라이드_07_변수중요도.png")
    print("  슬라이드_08_지도시각화.png")
    print("  슬라이드_09_벨뷰프리미엄.png")
    print("  슬라이드_10_시애틀접근성.png")
    print("  슬라이드_11_실제vs예측.png")
    print("  슬라이드_12_모델성능비교.png")
    print("  슬라이드_13_결론인포그래픽.png")
    print("  슬라이드_14_투자비교.png")
