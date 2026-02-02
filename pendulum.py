import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# -------------------------------
# 1. 진자 운동 시뮬레이션 함수 (개선된 버전)
# -------------------------------
def simulate_pendulum_detailed(L, theta0_deg, g=9.81, dt=0.001, t_max=10):
    """진자 운동을 상세하게 시뮬레이션하여 위치 데이터도 반환 (시간 정확도 개선)"""
    theta0 = np.radians(theta0_deg)
    omega = 0
    theta = theta0
    time = 0
    
    times = []
    angles = []
    positions_x = []
    positions_y = []
    
    while time < t_max:
        # 운동방정식: d²θ/dt² = -(g/L)sin(θ)
        alpha = -(g / L) * np.sin(theta)
        omega += alpha * dt
        theta += omega * dt
        time += dt
        
        # 위치 계산 (원점에서 진자 끝까지)
        x = L * np.sin(theta)
        y = -L * np.cos(theta)
        
        times.append(time)
        angles.append(theta)
        positions_x.append(x)
        positions_y.append(y)
    
    # 주기 측정 개선
    zero_crossings = []
    for i in range(1, len(angles)):
        if angles[i-1] * angles[i] < 0 and omega > 0:  # 상승하며 0을 지날 때
            # 선형 보간으로 정확한 교차점 찾기
            t_cross = times[i-1] + (times[i] - times[i-1]) * (-angles[i-1] / (angles[i] - angles[i-1]))
            zero_crossings.append(t_cross)
    
    if len(zero_crossings) >= 2:
        period = 2 * (zero_crossings[1] - zero_crossings[0])
    else:
        period = np.nan
    
    return period, np.array(times), np.array(angles), np.array(positions_x), np.array(positions_y)

def simulate_pendulum(L, theta0_deg, g=9.81, dt=0.01, t_max=10):
    """기존 주기 측정용 함수 (호환성 유지)"""
    period, _, _, _, _ = simulate_pendulum_detailed(L, theta0_deg, g, dt, t_max)
    return period

# -------------------------------
# 2. 이론적 주기 계산
# -------------------------------
def theoretical_period(L, theta0_deg, g=9.81):
    """소각 근사와 타원적분을 이용한 이론적 주기"""
    theta0 = np.radians(theta0_deg)
    
    # 소각 근사 (θ << 1일 때)
    T_small = 2 * np.pi * np.sqrt(L / g)
    
    # 큰 각도에 대한 1차 보정
    # T ≈ T0 * (1 + (1/4) * sin²(θ0/2))
    T_corrected = T_small * (1 + 0.25 * np.sin(theta0/2)**2)
    
    return T_small, T_corrected

# -------------------------------
# 3. 데이터셋 생성 (개선된 버전)
# -------------------------------
def generate_dataset():
    """더 넓은 범위의 데이터셋 생성"""
    lengths = np.linspace(0.1, 2.0, 30)  # 줄 길이
    angles = np.linspace(5, 45, 8)       # 초기 각도 범위 확대
    
    X, y = [], []
    theoretical_periods = []
    
    print("데이터셋 생성 중...")
    for i, L in enumerate(lengths):
        for j, theta in enumerate(angles):
            T_sim = simulate_pendulum(L, theta)
            T_small, T_corrected = theoretical_period(L, theta)
            
            if not np.isnan(T_sim):
                X.append([L, theta])
                y.append(T_sim)
                theoretical_periods.append([T_small, T_corrected])
        
        if (i + 1) % 10 == 0:
            print(f"진행률: {(i+1)/len(lengths)*100:.1f}%")
    
    return np.array(X), np.array(y), np.array(theoretical_periods)

# -------------------------------
# 4. 머신러닝 모델 학습 (개선된 버전)
# -------------------------------
def train_models(X, y):
    """다양한 모델 학습"""
    # 특성 공학: 길이와 각도의 조합 특성 추가
    X_enhanced = np.column_stack([
        X[:, 0],  # 길이 L
        X[:, 1],  # 각도 θ
        np.sqrt(X[:, 0]),  # √L (주기와 관련)
        np.sin(np.radians(X[:, 1]/2))**2,  # sin²(θ/2) (보정항)
        X[:, 0] * np.sin(np.radians(X[:, 1]/2))**2  # L × sin²(θ/2)
    ])
    
    # 모델들
    lr = LinearRegression()
    tree = DecisionTreeRegressor(max_depth=8, random_state=42)
    
    # 학습
    lr.fit(X_enhanced, y)
    tree.fit(X, y)  # 트리는 원본 특성 사용
    
    return lr, tree, X_enhanced

# -------------------------------
# 5. 애니메이션 비교 함수 (시간 동기화 개선)
# -------------------------------
def create_comparison_animation(L=1.0, theta0_deg=20, duration=6):
    """AI 예측 vs 실제 물리 현상 애니메이션 비교 (시간 동기화)"""
    
    # 실제 물리 시뮬레이션 (dt=0.01로 정밀 시뮬레이션)
    T_actual, times, angles, pos_x, pos_y = simulate_pendulum_detailed(L, theta0_deg, dt=0.01, t_max=duration)
    
    # AI 모델 예측 주기
    X_test = np.array([[L, theta0_deg]])
    X_test_enhanced = np.column_stack([
        X_test[:, 0],
        X_test[:, 1], 
        np.sqrt(X_test[:, 0]),
        np.sin(np.radians(X_test[:, 1]/2))**2,
        X_test[:, 0] * np.sin(np.radians(X_test[:, 1]/2))**2
    ])
    
    # 전역 모델 사용
    global lr_model, tree_model
    
    T_pred_lr = lr_model.predict(X_test_enhanced)[0]
    T_pred_tree = tree_model.predict(X_test)[0]
    
    # AI 예측 기반 운동 (실제와 같은 시간 축 사용)
    theta0_rad = np.radians(theta0_deg)
    
    # 선형회귀 예측: 소각 근사 기반 단순 조화 운동
    omega_lr = 2 * np.pi / T_pred_lr
    angles_lr = theta0_rad * np.cos(omega_lr * times)
    pos_x_lr = L * np.sin(angles_lr)  # 작은 각도에서 sin(θ) ≈ θ
    pos_y_lr = -L * np.cos(angles_lr)  # y도 변화하도록 수정
    
    # 결정트리 예측: 비선형 효과를 어느 정도 반영
    omega_tree = 2 * np.pi / T_pred_tree
    angles_tree = theta0_rad * np.cos(omega_tree * times)
    pos_x_tree = L * np.sin(angles_tree)
    pos_y_tree = -L * np.cos(angles_tree)
    
    # 애니메이션 설정
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'단진자 비교 (길이(L)={L}m, 초기각각(θ)={theta0_deg}°)', fontsize=16, fontweight='bold')
    
    # 각 서브플롯 설정
    titles = ['실제 진공상태 진자 (물리 시뮬레이션)', '선형회귀 AI 모델의 예측 시뮬레이션', '결정트리 AI 모델의 예측 시뮬레이션']
    for ax, title in zip([ax1, ax2, ax3], titles):
        ax.set_xlim(-L*1.2, L*1.2)
        ax.set_ylim(-L*1.2, L*0.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        # 진자 고정점 표시
        ax.plot(0, 0, 'ko', markersize=8)
    
    # 진자 요소들 (선 굵기와 크기 조정)
    line1, = ax1.plot([], [], 'b-', linewidth=3, label='실제 물리')
    bob1, = ax1.plot([], [], 'bo', markersize=12)
    trail1, = ax1.plot([], [], 'b-', alpha=0.4, linewidth=2)
    
    line2, = ax2.plot([], [], 'r-', linewidth=3, label='선형회귀 예측')
    bob2, = ax2.plot([], [], 'ro', markersize=12)
    trail2, = ax2.plot([], [], 'r-', alpha=0.4, linewidth=2)
    
    line3, = ax3.plot([], [], 'g-', linewidth=3, label='결정트리 예측') 
    bob3, = ax3.plot([], [], 'go', markersize=12)
    trail3, = ax3.plot([], [], 'g-', alpha=0.4, linewidth=2)
    
    # 성능 정보 텍스트 박스 (위치와 크기 개선)
    info_text1 = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9), 
                         fontsize=10, fontweight='bold')
    info_text2 = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.9), 
                         fontsize=10, fontweight='bold')
    info_text3 = ax3.text(0.02, 0.98, '', transform=ax3.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9), 
                         fontsize=10, fontweight='bold')
    
    # 주기 오차 정보 (하단)
    error_text = fig.text(0.5, 0.02, '', ha='center', va='bottom', fontsize=12,
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    
    # 궤적 저장용
    trail_x1, trail_y1 = [], []
    trail_x2, trail_y2 = [], []
    trail_x3, trail_y3 = [], []
    
    # 프레임 건너뛰기 (애니메이션 속도 조절)
    frame_skip = max(1, len(times) // 600)  # 최대 600프레임으로 제한
    
    def animate(frame):
        # 프레임 건너뛰기 적용
        actual_frame = frame * frame_skip
        if actual_frame >= len(times):
            return line1, bob1, trail1, line2, bob2, trail2, line3, bob3, trail3
        
        t = times[actual_frame]
        trail_length = 80  # 궤적 길이
        
        # 실제 물리 시뮬레이션
        x1, y1 = pos_x[actual_frame], pos_y[actual_frame]
        line1.set_data([0, x1], [0, y1])
        bob1.set_data([x1], [y1])
        
        # 궤적 업데이트
        trail_x1.append(x1)
        trail_y1.append(y1)
        if len(trail_x1) > trail_length:
            trail_x1.pop(0)
            trail_y1.pop(0)
        trail1.set_data(trail_x1, trail_y1)
        
        # AI 선형회귀 예측 (같은 시간 축)
        x2, y2 = pos_x_lr[actual_frame], pos_y_lr[actual_frame]
        line2.set_data([0, x2], [0, y2])
        bob2.set_data([x2], [y2])
        
        trail_x2.append(x2)
        trail_y2.append(y2)
        if len(trail_x2) > trail_length:
            trail_x2.pop(0)
            trail_y2.pop(0)
        trail2.set_data(trail_x2, trail_y2)
        
        # AI 결정트리 예측 (같은 시간 축)
        x3, y3 = pos_x_tree[actual_frame], pos_y_tree[actual_frame]
        line3.set_data([0, x3], [0, y3])
        bob3.set_data([x3], [y3])
        
        trail_x3.append(x3)
        trail_y3.append(y3)
        if len(trail_x3) > trail_length:
            trail_x3.pop(0)
            trail_y3.pop(0)
        trail3.set_data(trail_x3, trail_y3)
        
        # 실시간 정보 업데이트
        info_text1.set_text(f'시간: {t:.2f}s\n실제 주기: {T_actual:.3f}s\n각도: {np.degrees(angles[actual_frame]):.1f}°')
        
        # 오차 계산 및 표시
        lr_error = abs(T_actual - T_pred_lr)
        tree_error = abs(T_actual - T_pred_tree)
        lr_error_percent = (lr_error / T_actual) * 100
        tree_error_percent = (tree_error / T_actual) * 100
        
        info_text2.set_text(f'시간: {t:.2f}s\n예측 주기: {T_pred_lr:.3f}s\n절대오차: {lr_error:.3f}s\n상대오차: {lr_error_percent:.1f}%')
        info_text3.set_text(f'시간: {t:.2f}s\n예측 주기: {T_pred_tree:.3f}s\n절대오차: {tree_error:.3f}s\n상대오차: {tree_error_percent:.1f}%')
        
        # 전체 성능 비교 (하단)
        better_model = "선형회귀" if lr_error < tree_error else "결정트리"
        error_text.set_text(f'성능 비교: {better_model} 모델이 더 정확함 | 실제 vs 선형회귀: {lr_error_percent:.1f}% | 실제 vs 결정트리: {tree_error_percent:.1f}%')
        
        return line1, bob1, trail1, line2, bob2, trail2, line3, bob3, trail3
    
    # 애니메이션 생성 (실제 시간과 동기화)
    total_frames = len(times) // frame_skip
    # interval을 실제 물리 시간과 맞춤 (ms 단위)
    interval = int(times[frame_skip] * 1000) if len(times) > frame_skip else 50
    
    anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                 interval=interval, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim

# -------------------------------
# 6. 결과 시각화 
# -------------------------------
def visualize_results(X, y, theoretical_periods, lr_model, tree_model, X_enhanced):
    """예측 결과와 이론값 비교"""
    y_pred_lr = lr_model.predict(X_enhanced)
    y_pred_tree = tree_model.predict(X)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('단진자 주기 예측 모델 성능 분석', fontsize=16, fontweight='bold')
    
    # 1. 실제 vs 예측 (선형회귀)
    ax1 = axes[0, 0]
    ax1.scatter(y, y_pred_lr, c='red', alpha=0.6, label='선형회귀')
    ax1.plot([min(y), max(y)], [min(y), max(y)], 'k--', linewidth=2)
    ax1.set_xlabel('실제 주기 (s)')
    ax1.set_ylabel('예측 주기 (s)')
    ax1.set_title('선형회귀 모델 성능')
    ax1.grid(True)
    ax1.legend()
    
    # 2. 실제 vs 예측 (결정트리)
    ax2 = axes[0, 1]
    ax2.scatter(y, y_pred_tree, c='green', alpha=0.6, label='결정트리')
    ax2.plot([min(y), max(y)], [min(y), max(y)], 'k--', linewidth=2)
    ax2.set_xlabel('실제 주기 (s)')
    ax2.set_ylabel('예측 주기 (s)')
    ax2.set_title('결정트리 모델 성능')
    ax2.grid(True)
    ax2.legend()
    
    # 3. 길이에 따른 주기 변화
    ax3 = axes[1, 0]
    lengths_unique = np.unique(X[:, 0])
    for L in lengths_unique[::3]:  # 일부만 표시
        mask = X[:, 0] == L
        ax3.plot(X[mask, 1], y[mask], 'o-', label=f'L={L:.1f}m', alpha=0.7)
    ax3.set_xlabel('초기 각도 (도)')
    ax3.set_ylabel('주기 (s)')
    ax3.set_title('길이별 주기 변화')
    ax3.grid(True)
    ax3.legend()
    
    # 4. 이론값과의 비교
    ax4 = axes[1, 1]
    T_small_angle = theoretical_periods[:, 0]  # 소각 근사
    T_corrected = theoretical_periods[:, 1]    # 보정된 이론값
    
    ax4.scatter(y, T_small_angle, c='blue', alpha=0.5, label='소각 근사', s=20)
    ax4.scatter(y, T_corrected, c='orange', alpha=0.5, label='보정된 이론값', s=20)
    ax4.plot([min(y), max(y)], [min(y), max(y)], 'k--', linewidth=2)
    ax4.set_xlabel('실제 주기 (s)')
    ax4.set_ylabel('이론 주기 (s)')
    ax4.set_title('이론값과의 비교')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 성능 지표 출력
    mse_lr = mean_squared_error(y, y_pred_lr)
    mse_tree = mean_squared_error(y, y_pred_tree)
    mse_small = mean_squared_error(y, T_small_angle)
    mse_corrected = mean_squared_error(y, T_corrected)
    
    print("\n" + "="*50)
    print("모델 성능 비교 (평균제곱오차, MSE)")
    print("="*50)
    print(f"선형회귀 모델:    {mse_lr:.6f}")
    print(f"결정트리 모델:    {mse_tree:.6f}")
    print(f"소각 근사 이론:   {mse_small:.6f}")
    print(f"보정된 이론값:    {mse_corrected:.6f}")
    print("="*50)
    
    return y_pred_lr, y_pred_tree

# -------------------------------
# 7. 메인 실행 함수
# -------------------------------
def main():
    global lr_model, tree_model  # 애니메이션에서 사용하기 위해 전역 변수로 설정
    
    print("단진자 AI 예측 vs 실제 현상 비교 실험")
    print("="*60)
    
    # 1. 데이터셋 생성
    print("\n단진자 데이터 생성 중...")
    X, y, theoretical_periods = generate_dataset()
    print(f"데이터셋 생성 완료: {X.shape[0]}개 샘플")
    
    # 2. 모델 학습
    print("\nAI 모델 학습 중...")
    lr_model, tree_model, X_enhanced = train_models(X, y)
    print("모델 학습 완료")
    
    # 3. 결과 시각화
    print("\n결과 분석 중...")
    y_pred_lr, y_pred_tree = visualize_results(X, y, theoretical_periods, lr_model, tree_model, X_enhanced)
    
    # 4. 애니메이션 비교
    print("\n애니메이션 비교 실행...")
    print("다양한 조건으로 비교해보세요!")
    
    # 예시 조건들 (다양한 물리적 상황)
    test_conditions = [
        {"L": 1.0, "theta0_deg": 10, "duration": 20},  
        {"L": 0.5, "theta0_deg": 30, "duration": 10},  
        {"L": 1.5, "theta0_deg": 45, "duration": 20},  
        {"L": 2.0, "theta0_deg": 20, "duration": 25}   
    ]
    
    for i, condition in enumerate(test_conditions):
        print(f"\n실험 {i+1}: 길이 {condition['L']}m, 초기각 {condition['theta0_deg']}°")
        input("Enter를 눌러 애니메이션을 시작하세요...")
        anim = create_comparison_animation(**condition)
        
        if i < len(test_conditions) - 1:
            input("다음 실험으로 넘어가려면 Enter를 누르세요...")
    
    print("\n실험 종료")

if __name__ == "__main__":
    main()