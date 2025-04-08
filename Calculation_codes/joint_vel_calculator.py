import numpy as np

def compute_joint_velocities(theta1, theta2, x_dot, y_dot, L1, L2):
    """
    현재 관절각 (theta1, theta2)에서 원하는 엔드이펙터 속도 (x_dot, y_dot)를 만족하는
    joint velocities (theta1_dot, theta2_dot)를 계산합니다.
    
    주의: Jacobian이 특이하지 않아야 합니다.
    """
    J = np.array([
        [-L1*np.sin(theta1) - L2*np.sin(theta1+theta2), -L2*np.sin(theta1+theta2)],
        [ L1*np.cos(theta1) + L2*np.cos(theta1+theta2),  L2*np.cos(theta1+theta2)]
    ])
    
    x_dot_vec = np.array([x_dot, y_dot])
    joint_vel = np.linalg.inv(J) @ x_dot_vec
    return joint_vel[0], joint_vel[1]

def compute_inverse_kinematics(x, y, L1, L2, elbow_up=False):
    """
    주어진 (x, y) 좌표를 만족하는 (theta1, theta2)를 계산합니다.
    
    2링크 로봇의 경우,
      x = L1*cos(theta1) + L2*cos(theta1 + theta2)
      y = L1*sin(theta1) + L2*sin(theta1 + theta2)
    
    두 가지 해(엘보우 업/다운)가 존재하는데, 기본적으로 엘보우 다운(θ₂ < 0) 해를 사용합니다.
    만약 elbow_up을 True로 설정하면 엘보우 업(θ₂ ≥ 0) 해를 사용합니다.
    """
    # cos(theta2) 계산
    cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if cos_theta2 < -1 or cos_theta2 > 1:
        raise ValueError("주어진 (x, y) 위치에 대해 해가 존재하지 않습니다.")
    
    theta2 = np.arccos(cos_theta2)
    if not elbow_up:
        # 기본적으로 엘보우 다운을 위해 theta2를 음수로 만듭니다.
        theta2 = -theta2
    
    # theta1 계산
    k1 = L1 + L2 * np.cos(theta2)
    k2 = L2 * np.sin(theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    return theta1, theta2

if __name__ == "__main__":
    # 링크 길이 설정 (예시)
    L1 = 1.0
    L2 = 1.0

    # t=0 순간의 joint velocity 계산 예시
    theta1_init = 1.382  # 예: 45도
    theta2_init = -2.004  # 예: 30도
    x_dot_desired = 1.5         # x 방향 속도
    y_dot_desired = -2.5           # y 방향 속도

    theta1_dot, theta2_dot = compute_joint_velocities(theta1_init, theta2_init,
                                                        x_dot_desired, y_dot_desired,
                                                        L1, L2)
    print("t=0에서의 joint velocities:")
    print("theta1_dot =", theta1_dot)
    print("theta2_dot =", theta2_dot)

    # 역기구학: 원하는 (x, y) 좌표에 대해 (theta1, theta2) 계산 (기본적으로 엘보우 다운)
    x_target = 1.0
    y_target = 0.4

    try:
        theta1_sol, theta2_sol = compute_inverse_kinematics(x_target, y_target, L1, L2)
        print("\n주어진 (x, y) =", (x_target, y_target), "에 대한 해 (기본적으로 엘보우 다운):")
        print("theta1 =", theta1_sol, "rad (", np.rad2deg(theta1_sol), "deg )")
        print("theta2 =", theta2_sol, "rad (", np.rad2deg(theta2_sol), "deg )")
    except ValueError as e:
        print(e)
