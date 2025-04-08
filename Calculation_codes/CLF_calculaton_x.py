import numpy as np
from scipy.linalg import solve_continuous_lyapunov
import math

def compute_Px(k_px, k_dx):
    # 상태행렬 A_x 정의
    A_x = np.array([[0, 1],
                    [-k_px, -k_dx]])
    
    # Q = -np.eye(2)
    Q = np.array([[1.0, -0.9],
                    [-0.9, 1.0]])
    Q = -Q

    P_x = solve_continuous_lyapunov(A_x.T, Q)
    p11 = P_x[0, 0]
    p12 = P_x[0, 1]
    p21 = P_x[1, 0]
    p22 = P_x[1, 1]
    return P_x, p11, p12, p21, p22

def compute_Vx_at_point(P_x, x_now, x_dot_now, x_d, x_dot_d):
    e_state = np.array([[x_now - x_d],
                        [x_dot_now - x_dot_d]], dtype=np.float64)
    
    V_x = 0.5 * (e_state.T @ P_x @ e_state)
    V_x = V_x[0, 0]
    return V_x

if __name__ == "__main__":
    # 예시: 게인 값 설정
    k_px = 1.5
    k_dx = 1.0
    
    x_now = -1.0
    x_dot_now = -1.5
    
    x_d = -0.3
    x_dot_d = 0.0
    
    P_x, p11, p12, p21, p22 = compute_Px(k_px, k_dx)
    print("P_x =\n", P_x)
    print("p22/p12 =\n", (p22)/(p12))
    
    V_2x_value = compute_Vx_at_point(P_x, x_now, x_dot_now, x_d, x_dot_d)
    print(f'v2: Lyapunov value at initial condition (x,dot x) = {x_now, x_dot_now} is {V_2x_value}.')

    # unsafe 집합 D에서 최소 V
    x_danger = -1.30
    x_dot_for_minimum = ((-p12)/(p22)) * x_danger

    V_1x_value = compute_Vx_at_point(P_x, x_danger, x_dot_for_minimum, x_d, x_dot_d)
    print(f'v1: minimum Lyapunov value at (x,dot x) = {x_danger, x_dot_for_minimum} is {V_1x_value}.')

    print(f'So the value (v2)/(v1) is {(V_2x_value) / (V_1x_value) }.')
    
    x_pos_max_consider = 0.5
    # l = 2 / (((p22)/(p12))* abs(x_dot_now))
    l = 2 / 0.5
    print(f'slope l is {l}.')

    delta = (1 / l) * math.log(V_2x_value / V_1x_value)
    print(f'margin delta is {delta}.')

    delta_margin = 1.2
    chosen_delta = delta_margin * delta
    print(f'chosen delta is {chosen_delta}.')
    sigma_1_x = 1 / (1 + np.exp(-l * chosen_delta))
    sigma_2_x = 1 / (1 + np.exp(l * chosen_delta))
    minimum_theta = (V_2x_value - V_1x_value) / (sigma_1_x * V_1x_value - sigma_2_x * V_2x_value)
    print(f'minimum_theta is {minimum_theta}.')

    theta_margin = 1.5
    chosen_theta = theta_margin * minimum_theta
    print(f'chosen_theta is {chosen_theta}.')

    k = (1 + chosen_theta * sigma_2_x) * V_2x_value
    sigma_init_x = 1 / (1 + np.exp(l * (x_now - x_danger - chosen_delta))) 
    modW_x = (1 + chosen_theta * sigma_init_x)
    init_W = modW_x * V_2x_value - k
    print(f'CLBF W of initial position is {init_W}.')
    print(f'scaling value k is {k}.')

    minW_of_unsafe = (1 + chosen_theta * sigma_1_x) * V_1x_value - k
    print(f'CLBF W of minimum position of unsafe is {minW_of_unsafe}.')
    print(f'W(init) - W(min.unsafe): {init_W - minW_of_unsafe}.')
