import numpy as np
from scipy.linalg import solve_continuous_lyapunov
import math

def compute_Py(k_py, k_dy):
    # 상태행렬 A_y 정의
    A_y = np.array([[0, 1],
                    [-k_py, -k_dy]])
    
    # Q = -np.eye(2)
    Q = np.array([[1.0, -0.9],
                  [-0.9, 1.0]])
    Q = -Q
    
    P_y = solve_continuous_lyapunov(A_y.T, Q)
    p11 = P_y[0, 0]
    p12 = P_y[0, 1]
    p21 = P_y[1, 0]
    p22 = P_y[1, 1]
    return P_y, p11, p12, p21, p22

def compute_Vy_at_point(P_y, y_now, y_dot_now, y_d, y_dot_d):
    e_state = np.array([[y_now - y_d],
                        [y_dot_now - y_dot_d]], dtype=np.float64)
    
    V_y = 0.5 * (e_state.T @ P_y @ e_state)
    V_y = V_y[0,0]
    return V_y

if __name__ == "__main__":
    # 예시: 게인 값 설정
    k_py = 1.0
    k_dy = 1.0
    
    y_now = 0.4
    y_dot_now = -2.5
    
    y_d = 1.0
    y_dot_d = 0.0
    
    P_y, p11,p12,p21,p22 = compute_Py(k_py, k_dy)
    print("P_y =\n", P_y)
    print("p22/p12 =\n", (p22)/(p12))
    
    V_2y_value = compute_Vy_at_point(P_y, y_now, y_dot_now, y_d, y_dot_d)

    print(f'v2: Lyapunov value at initial condition (y,dot y) = {y_now, y_dot_now} is {V_2y_value}.')

    #minimum V at unsafe set D
    y_danger = -0.3
    y_dot_for_minimum = ((-p12)/(p22)) * y_danger

    V_1y_value = compute_Vy_at_point(P_y, y_danger, y_dot_for_minimum, y_d, y_dot_d)
    print(f'v1: minimum Lyapunov value at (y,dot y) = {y_danger, y_dot_for_minimum} is {V_1y_value}.')

    print(f'So the value (v2)/(v1) is {(V_2y_value) / (V_1y_value) }.')
    
    error_y_max_consider = 0.5
    # l = 2 / (((p22)/(p12))* abs(y_dot_now))
    l = 2 / 0.5
    print(f'slope l is {l}.')

    delta = (1 / l) * math.log(V_2y_value / V_1y_value)

    print(f'margin delta is {delta}.')

    delta_margin = 1.4
    chosen_delta = delta_margin*delta
    print(f'chosen delta is {chosen_delta}.')
    sigma_1_y = 1 / (1 + np.exp(-l * chosen_delta))
    sigma_2_y = 1 / (1 + np.exp(l * chosen_delta))
    minimum_theta = (V_2y_value - V_1y_value) / (sigma_1_y*V_1y_value - sigma_2_y*V_2y_value)

    print(f'minimum_theta is {minimum_theta}.')

    theta_margin = 1.01
    chosen_theta = theta_margin * minimum_theta
    print(f'chosen_theta is {chosen_theta}.')

    k = (1 + chosen_theta * sigma_2_y) * V_2y_value
    sigma_init_y = 1 / (1 + np.exp(l * (y_now - y_danger -chosen_delta))) 
    modW_y = (1 + chosen_theta * sigma_init_y)
    init_W = modW_y * V_2y_value - k
    print(f'CLBF W of initial position is {init_W}.')
    print(f'scaling value k is {k}.')

    minW_of_unsafe = (1 + chosen_theta * sigma_1_y) * V_1y_value - k
    print(f'CLBF W of minimum position of unsafe is {minW_of_unsafe}.')
    print(f'W(init) - W(min.unsafe): {init_W - minW_of_unsafe}.')






