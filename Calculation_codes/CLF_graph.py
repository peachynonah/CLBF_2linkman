import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov

def compute_Py(k_py, k_dy):
    A_y = np.array([[0, 1],
                    [-k_py, -k_dy]])
    # Q = -np.eye(2)

    Q = np.array([[1.0, -0.5],
                    [-0.5, 1.0]])
    Q = -Q
    P_y = solve_continuous_lyapunov(A_y.T, Q)
    return P_y

def compute_Vy(P_y, y, y_dot, y_d, y_dot_d):
    e_state = np.array([[y - y_d],
                        [y_dot - y_dot_d]])
    V_y = 0.5 * (e_state.T @ P_y @ e_state)
    return V_y[0, 0]

# 파라미터 설정
k_py = 1.0
k_dy = 1.0
y_d = 1.0
y_dot_d = 0.0

P_y = compute_Py(k_py, k_dy)

# 그리드 생성
y = np.linspace(-2, 2, 400)
y_dot = np.linspace(-2, 2, 400)
Y, Y_dot = np.meshgrid(y, y_dot)
V_values = np.zeros_like(Y)

for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        V_values[i, j] = compute_Vy(P_y, Y[i, j], Y_dot[i, j], y_d, y_dot_d)

# 특정 레벨(1,2,3,4,5...)에서 등고선 표시
plt.figure(figsize=(8, 6))
levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 원하는 레벨을 추가/조정 가능
cs = plt.contour(Y, Y_dot, V_values, levels=levels, colors='black')
plt.clabel(cs, inline=True, fontsize=8)

plt.xlabel('y')
plt.ylabel('y_dot')
plt.title('Lyapunov Function V_y Contour (Discrete Levels)')
plt.grid(True)
plt.show()