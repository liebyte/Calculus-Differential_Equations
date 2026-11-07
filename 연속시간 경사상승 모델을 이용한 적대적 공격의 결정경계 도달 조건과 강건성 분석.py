from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import brentq


# <1> 실험 조건 및 함수
OUTPUT_DIR = Path(__file__).resolve().parent / "python_execution_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

W = np.array([2.0, 1.0], dtype=float)  # 고정 가중치
B = -1.0                              # 편향
X0 = np.array([1.0, 0.0], dtype=float) # 초기 입력
A = float(np.dot(W, W))               # ||w||² = 5
Z0 = float(np.dot(W, X0) + B)         # 초기 분류 점수 = 1

assert A > 0 and Z0 > 0, "가정: w != 0, 초기 분류 점수 z0 > 0"


def sigmoid(z):
    # 클래스 1의 예측 확률
    return 1.0 / (1.0 + np.exp(-z))


def dz_dt(z, a=A):
    # 분류 점수의 연속시간 변화율
    return -a / (1.0 + np.exp(z))


def exact_score(t, z0=Z0, a=A):
    # 음함수 z + exp(z) = z0 + exp(z0) - a*t를 Brent법으로 풂.
    target = z0 + math.exp(z0) - a * float(t)
    return brentq(lambda z: z + math.exp(z) - target,
                  -100.0, 100.0, xtol=1e-13, rtol=1e-14)


def exact_scores(t_values, z0=Z0, a=A):
    return np.array([exact_score(t, z0, a) for t in t_values])


def euler_score(h, t_end=0.5, z0=Z0, a=A):
    # 오일러 방법
    steps = round(t_end / h)
    if not math.isclose(steps * h, t_end, rel_tol=0, abs_tol=1e-10):
        raise ValueError("t_end는 h의 정수배여야 합니다.")
    z = float(z0)
    for _ in range(steps):
        z += h * dz_dt(z, a)
    return z


def boundary_time(z0=Z0, a=A):
    # t*
    return (z0 + math.exp(z0) - 1.0) / a


def minimum_perturbation(z0=Z0, w=W):
    # 결정경계까지의 최소 L2 거리(오분류 교란량의 하한)
    return z0 / np.linalg.norm(w)


def boundary_time_scaled(c, z0=Z0, a=A):
    # w->c*w, b->c*b (c>0)일 때의 경계 도달 진행 매개변수
    c = np.asarray(c, dtype=float)
    if np.any(c <= 0):
        raise ValueError("c는 양수여야 합니다.")
    return (c*z0 + np.exp(c*z0) - 1.0) / (c**2 * a)


def setup_plot_font():
    # 그래프 글꼴 설정
    candidates = ["NanumGothic", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    plt.rcParams["axes.unicode_minus"] = False
    return False


KOREAN = setup_plot_font()

#그래프 색상 설정
COLOR_BLUE = "#1E569C"       # 공격 경로 / 분류 점수 / 배율별 이론 곡선
COLOR_TEAL = "#168376"       # 예측 확률
COLOR_RED = "#C43B4D"        # 결정경계 도달점 / 표의 실험값
COLOR_BOUNDARY = "#78879B"   # 결정경계 (회색 파선)
COLOR_GREEN = "#188754"      # 초기 입력점
COLOR_GUIDE_RED = "#B84A4A"  # 분류 경계 기준선
COLOR_GUIDE_TEAL = "#8DC9C3" # 예측 확률 0.5 기준선


def label(korean, english):
    return korean if KOREAN else english


# <2> 이론값 및 Table. 2
# 오일러 방법의 수치해 비교
T_STAR = boundary_time()
EPS_MIN = minimum_perturbation()
DELTA_STAR = -(Z0 / A) * W
X_STAR = X0 + DELTA_STAR

TARGET_T = 0.5
EXACT_Z_AT_HALF = exact_score(TARGET_T)
EXACT_P_AT_HALF = float(sigmoid(EXACT_Z_AT_HALF))
EULER_STEPS = [0.100, 0.050, 0.025, 0.010]

print("[초기 조건과 이론값]")
print(f"w = {W.tolist()}, b = {B}, x0 = {X0.tolist()}")
print(f"z0 = {Z0:.6f}, a = ||w||² = {A:.6f}")
print(f"t* = {T_STAR:.6f}")
print(f"경계 도달점 = ({X_STAR[0]:.6f}, {X_STAR[1]:.6f})")
print(f"최소 교란량 = {EPS_MIN:.6f}")
print(f"z(0.5) = {EXACT_Z_AT_HALF:.6f}")
print(f"p(0.5) = {EXACT_P_AT_HALF:.6f}")
print("\n[Table. 2: 오일러 방법과 음함수 해의 비교]")
print("   h       오일러 z(0.5)     절댓값 오차")

TABLE_1 = []
for h in EULER_STEPS:
    z_euler = euler_score(h, TARGET_T)
    err = abs(z_euler - EXACT_Z_AT_HALF)
    TABLE_1.append((h, z_euler, err))
    print(f"{h:6.3f}      {z_euler:10.6f}      {err:10.6f}")


# <3> Table. 3 (가중치, 편향 배율에 따른 강건성 비교)
SCALES = [0.5, 1.0, 2.0, 4.0]
TABLE_2 = []
print("\n[Table. 3: 가중치·편향 배율 비교]")
print("  c       초기 예측 확률      t*(c)       최소 L2 교란량")
for c in SCALES:
    initial_probability = float(sigmoid(c * Z0))
    scaled_tstar = float(boundary_time_scaled(c))
    # w와 b를 모두 c배하면 결정경계의 위치는 그대로임
    scaled_eps = (c * Z0) / np.linalg.norm(c * W)
    TABLE_2.append((c, initial_probability, scaled_tstar, scaled_eps))
    print(f"{c:4.1f}        {100*initial_probability:6.2f}%       "
          f"{scaled_tstar:8.6f}       {scaled_eps:8.6f}")

C_MIN = 2.0 / Z0
T_MIN = (Z0**2 * (1.0 + math.exp(2.0))) / (4.0 * A)
print(f"\nt*(c)의 최솟값: c = {C_MIN:.6f}, t*_min = {T_MIN:.6f}")


# <4> Fig. 2 (입력 공간의 결정경계 및 공격 경로)
fig, ax = plt.subplots(figsize=(6.2, 5.4))
x1 = np.linspace(0.10, 1.30, 300)
# w1*x1 + w2*x2 + b = 0  ->  x2 = (-b-w1*x1)/w2
boundary_x2 = (-B - W[0] * x1) / W[1]
ax.plot(x1, boundary_x2, "--", color=COLOR_BOUNDARY, linewidth=1.8,
        label=label(r"결정경계 ($2x_1 + x_2 - 1 = 0$)",
                                          "Decision boundary"))

t_grid = np.linspace(0.0, 1.05, 180)
z_grid = exact_scores(t_grid)
attack_path = X0[None, :] + ((z_grid - Z0)/A)[:, None] * W[None, :]
ax.plot(attack_path[:, 0], attack_path[:, 1], color=COLOR_BLUE, linewidth=2.3,
        label=label("경사상승 공격 경로", "Gradient-ascent path"))
ax.scatter([X0[0]], [X0[1]], color=COLOR_GREEN, s=60, zorder=4)
ax.scatter([X_STAR[0]], [X_STAR[1]], color=COLOR_RED, s=60, zorder=4)
ax.annotate(label("초기 입력 (1, 0)", "Initial input (1, 0)"),
            xy=X0, xytext=(-58, 16), textcoords="offset points")
ax.annotate(label("경계 도달 (0.6, -0.2)", "Boundary (0.6, -0.2)"),
            xy=X_STAR, xytext=(10, -28), textcoords="offset points")
ax.set_xlim(0.15, 1.30)
ax.set_ylim(-0.65, 0.65)
ax.set_xlabel(label(r"정규화된 특징 $x_1$", "Normalized feature x1"))
ax.set_ylabel(label(r"정규화된 특징 $x_2$", "Normalized feature x2"))
ax.grid(alpha=0.22)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig1_decision_boundary.png", dpi=200)
plt.close(fig)


# <5> Fig. 3 (공격 진행에 따른 분류 점수와 예측 확률)
fig, ax_score = plt.subplots(figsize=(8.0, 4.2))
ax_prob = ax_score.twinx()

t_grid = np.linspace(0, 1.05, 351)
z_grid = exact_scores(t_grid)
p_grid = sigmoid(z_grid)
line1, = ax_score.plot(t_grid, z_grid, color=COLOR_BLUE, linewidth=2.2,
                       label=label("분류 점수 z(t)", "Score z(t)"))
line2, = ax_prob.plot(t_grid, p_grid, "-.", color=COLOR_TEAL, linewidth=1.9,
                      label=label("클래스 1 예측 확률 p(t)", "Class-1 probability p(t)"))
ax_score.axhline(0, color=COLOR_GUIDE_RED, linestyle="--", linewidth=1.0, alpha=0.9)
ax_prob.axhline(0.5, color=COLOR_GUIDE_TEAL, linestyle=":", linewidth=1.0, alpha=0.8)
ax_score.axvline(T_STAR, color=COLOR_GUIDE_RED, linestyle=":", linewidth=1.2, alpha=0.9)
ax_score.annotate(f"t* = {T_STAR:.4f}",
                  (T_STAR, 0), xytext=(T_STAR+0.035, 0.26),
                  arrowprops={"arrowstyle": "-", "lw": 0.8})
ax_score.set_xlim(0, 1.05)
ax_score.set_ylim(-1.8, 1.1)
ax_prob.set_ylim(0, 1.0)
ax_score.set_xlabel(label("공격 진행 매개변수 t (무차원)",
                          "Attack-progress parameter t (dimensionless)"))
ax_score.set_ylabel(label("분류 점수 z(t)", "Score z(t)"), color=COLOR_BLUE)
ax_prob.set_ylabel(label("예측 확률 p(t)", "Probability p(t)"), color=COLOR_TEAL)
ax_score.grid(alpha=0.22)
ax_score.legend([line1, line2], [line1.get_label(), line2.get_label()],
                loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig2_score_probability.png", dpi=200)
plt.close(fig)


# 6. Fig. 4 (가중치·편향의 배율에 따른 경계 도달 매개변수)
fig, ax = plt.subplots(figsize=(8.0, 4.2))
c_grid = np.linspace(0.30, 4.50, 500)
ax.plot(c_grid, boundary_time_scaled(c_grid), color=COLOR_BLUE, linewidth=2.2,
        label=label("결정경계 도달 매개변수 t*", "Boundary time t*"))
ax.scatter(SCALES, boundary_time_scaled(np.array(SCALES)), color=COLOR_RED, s=50,
           label=label("Table. 3의 실험값", "Table 2 values"), zorder=3)
ax.axvline(C_MIN, color=COLOR_BOUNDARY, linestyle=":", linewidth=1.0, alpha=0.7)
ax.set_xlim(0.3, 4.5)
ax.set_xlabel(label("가중치·편향 배율 c", "Weight/bias multiplier c"))
ax.set_ylabel(label("t* (무차원)", "t* (dimensionless)"))
ax.grid(alpha=0.22)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig3_scaling_analysis.png", dpi=200)
plt.close(fig)