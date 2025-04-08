import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point, Wrench
import numpy as np
from scipy import linalg

class controller(Node):
    def __init__(self):
        super().__init__("controller")
        self.pub = self.create_publisher(Wrench, 'Torque_input', 10)
        self.ee_pub = self.create_publisher(Point, 'Crts_ee_point', 10)
        self.pos_jnt1_pub = self.create_publisher(Point, 'Crts_jnt1_point', 10)
        # self.vel_pub = self.create_publisher(Point, 'Crts_ee_vel', 10)
        # self.CLBF_pub = self.create_publisher(Point, 'CLBF_W', 10)
        self.Norm_pub = self.create_publisher(Point, 'Norm', 10)
        # self.Fsafe_pub = self.create_publisher(Point, 'Force_Fsafe', 10)
        self.twist_sub = self.create_subscription(Twist, 'Joint_Twist', self.twist_callback, 10)
        self.point_sub = self.create_subscription(Point, 'Joint_Point', self.point_callback, 10)
        
        #튜닝: nosafety
        self.ee_d = np.array([[0.30],
                              [1.00]], dtype=np.float64)
        #
        self.Kp1 =1.5
        self.Kp2 =1.0
        self.Kd1 = 1.0
        self.Kd2 = 1.0
        self.dt = 0.001
        self.proposition = 1.0
        #unsafe boundary
        self.unsafex = 1.3
        self.unsafey = -0.3
        #parameter
        self.x_inc = 4.0
        self.y_inc = 4.0
        self.xweight = 50
        self.yweight = 6.1
        self.delta_x = 2.0 * 0.14
        self.delta_y = 2.0 * 0.29
        self.ksafe = 0.0

        # Joint states as (2,1) 열 벡터
        self.pos_state = np.array([[0.0],
                                   [0.0]], dtype=np.float64)
        self.vel_state = np.array([[0.0],
                                   [0.0]], dtype=np.float64)

        self.poserr = np.zeros((2,1), dtype=np.float64)
        self.velerr = np.zeros((2,1), dtype=np.float64)

        self.link = 2
        # CLF, barrier shaped function and Sontag's universal formula
        self.V = 0.0
        self.W = 0.0
        
        self.alpha = 0.0
        self.beta = 0.0
        
        # robot information
        self.m = np.array([[0.8],
                           [0.8]], dtype=np.float64) 
        self.l = np.array([[1.0],
                           [1.0]], dtype=np.float64)
        
        # Joint dynamics matrices (모두 (2,1) 또는 (2,2))
        self.M = np.zeros((2,2), dtype=np.float64)
        self.C = np.zeros((2,1), dtype=np.float64)
        self.g = np.zeros((2,1), dtype=np.float64)

        # Cartesian dynamics 관련 변수
        self.Mcrt = np.zeros((2,2), dtype=np.float64)
        self.Ccrt = np.zeros((2,1), dtype=np.float64)
        self.gcrt = np.zeros((2,1), dtype=np.float64)
        self.G = 9.8

        self.jacobian = np.zeros((2,2), dtype=np.float64)
        self.jdot = np.zeros((2,2), dtype=np.float64)
        self.jpinv = np.zeros((2,2), dtype=np.float64)
        
    def twist_callback(self, sub_twist):
        # 업데이트: joint 속도는 (2,1) 열 벡터
        self.vel_state = np.array([[sub_twist.angular.x],
                                   [sub_twist.angular.y]], dtype=np.float64)

    def point_callback(self, sub_point):
        # 업데이트: joint 위치는 (2,1) 열 벡터
        self.pos_state = np.array([[sub_point.x],
                                   [sub_point.y]], dtype=np.float64)
        ctrl_rslt = self.ctsn_FL()
        pub_msg = Wrench()

        # 결과는 (2,1) 벡터이므로, 각 요소를 스칼라로 변환하여 할당
        if ctrl_rslt.shape == (2,1):
            pub_msg.torque.x = float(ctrl_rslt[0,0])
            pub_msg.torque.y = float(ctrl_rslt[1,0])
        else:
            pub_msg.torque.x = 0.0
            pub_msg.torque.y = 0.0

        self.pub.publish(pub_msg)

    def ctsn_FL(self):
        # joint state에서 스칼라 값 추출 (스칼라 연산에 필요)
        th1 = self.pos_state[0,0]
        th2 = self.pos_state[1,0]
        thd1 = self.vel_state[0,0]
        thd2 = self.vel_state[1,0]

        L1 = self.l[0,0]
        L2 = self.l[1,0]
        m1 = self.m[0,0]
        m2 = self.m[1,0]

        # Jacobian 계산 (스칼라 값을 사용)
        jacobian11 = -L1 * math.sin(th1) - L2 * math.sin(th1 + th2)
        jacobian12 = -L2 * math.sin(th1 + th2)
        jacobian21 = L1 * math.cos(th1) + L2 * math.cos(th1 + th2)
        jacobian22 = L2 * math.cos(th1 + th2)
        self.jacobian = np.array([[jacobian11, jacobian12],
                                  [jacobian21, jacobian22]], dtype=np.float64)
        
        # Forward kinematics: Joint 1, end-effector 위치 계산
        xpos = L1 * math.cos(th1) + L2 * math.cos(th1 + th2)
        ypos = L1 * math.sin(th1) + L2 * math.sin(th1 + th2)

        print(f'eepoint is : {xpos, ypos}')
        ee_msg = Point()
        ee_msg.x = xpos
        ee_msg.y = ypos
        self.ee_pub.publish(ee_msg)

        x1pos = L1 * math.cos(th1)
        y1pos = L1 * math.sin(th1)

        pos_jnt1_msg = Point()
        pos_jnt1_msg.x = x1pos
        pos_jnt1_msg.y = y1pos
        self.pos_jnt1_pub.publish(pos_jnt1_msg)

        # Cartesian 속도: (2,1) 벡터
        crtvel = self.jacobian @ np.array([[thd1],
                                           [thd2]], dtype=np.float64)
        
        # vel_msg = Point()
        # vel_msg.x = crtvel[0,0]
        # vel_msg.y = crtvel[1,0]
        # self.vel_pub.publish(vel_msg)
        # error state 선언: 모두 2차원 벡터로 처리. 
        self.poserr = np.array([[xpos - self.ee_d[0,0]],
                                [ypos - self.ee_d[1,0]]], dtype=np.float64)
        self.velerr = crtvel
        self.state_err = np.vstack((self.poserr, self.velerr))

        # Joint Dynamics 계산
        m11 = m1 * L1**2 + m2 * (L1**2 + 2 * L1 * L2 * math.cos(th2) + L2**2)
        m12 = m2 * (L1 * L2 * math.cos(th2) + L2**2)
        m21 = m12
        m22 = m2 * L2**2
        self.M = np.array([[m11, m12],
                           [m21, m22]], dtype=np.float64)
        
        self.C = np.array([[-(m2 * L1 * L2 * math.sin(th2)) * (2 * thd1 * thd2 + thd2**2)],
                           [m2 * L1 * L2 * (thd1**2) * math.sin(th2)]],
                           dtype=np.float64)
        
        self.g = np.array([[(m1 + m2) * L1 * self.G * math.cos(th1) + m2 * self.G * L2 * math.cos(th1 + th2)],
                           [m2 * self.G * L2 * math.cos(th1 + th2)]],
                            dtype=np.float64)

        # Cartesian Dynamics: Jacobian 도함수 계산
        jdot11 = -L1 * math.cos(th1) * thd1 - L2 * math.cos(th1 + th2) * (thd1 + thd2)
        jdot12 = -L2 * math.cos(th1 + th2) * (thd1 + thd2)
        jdot21 = -L1 * math.sin(th1) * thd1 - L2 * math.sin(th1 + th2) * (thd1 + thd2)
        jdot22 = -L2 * math.sin(th1 + th2) * (thd1 + thd2)
        self.jdot = np.array([[jdot11, jdot12],
                              [jdot21, jdot22]], dtype=np.float64)

        # Jacobian의 의사역행렬 계산
        self.jpinv = np.linalg.pinv(self.jacobian)
        invj = self.jpinv
        self.Mcrt: np.ndarray = invj.T @ self.M @ invj

        # 속도 벡터를 (2,1) 열 벡터로 사용
        vel_vec = np.array([[thd1],
                            [thd2]], dtype=np.float64)
        self.Ccrt = invj.T @ (self.C - self.M @ invj @ self.jdot @ vel_vec)
        self.gcrt = invj.T @ self.g

        # FL 제어 (Cartesian 공간에서)
        # 오차 제어 항: (2,1) 열 벡터로 구성
        cartesian_error = np.array([[self.Kp1 * self.poserr[0,0] + self.Kd1 * self.velerr[0,0]],
                                    [self.Kp2 * self.poserr[1,0] + self.Kd2 * self.velerr[1,0]]], dtype=np.float64)
        # 제어력 계산
        Fcomp: np.ndarray = - self.Mcrt @ cartesian_error + self.Ccrt + self.gcrt # (2,1)
        # Fcomp: np.ndarray = - self.Mcrt @ cartesian_error + self.Ccrt # (2,1)
        
        # 제어력을 관절 토크로 매핑 (역Jacobian 전치)
        FL = self.jacobian.T @ Fcomp  # (2,1)


        # Control Lyapunov function for scalar y
        a11 = 0.0
        a12 =  1.0
        a21 = -self.Kp2
        a22 = -self.Kd2
        A = np.array([[a11, a12],
                      [a21, a22]], dtype=np.float64)
        Q = np.eye(2)

        Q = np.array([[1.0, -0.9],      
                      [-0.9, 1.0]])
        
        P_y:np.ndarray = linalg.solve_continuous_lyapunov(A.T, -Q)
        err_y = np.array([[ypos - self.ee_d[1,0]],
                            [crtvel[1,0]]], dtype=np.float64)
        V_y:np.ndarray = 0.5 * err_y.T @ P_y @ err_y # self.V는 2차원 1 X 1 배열 
        V_y = V_y[0,0]

        B_y = np.array([[0.0],
                        [1.0]], dtype=np.float64)

        sigma_y = 1 / (1 + np.exp(self.y_inc * (err_y + self.ee_d[1,0] - self.unsafey - self.delta_y)))
        modW_y = (1 + self.yweight * sigma_y)
        W_y = modW_y * V_y


        ##입력부.
        self.alpha:np.ndarray = - modW_y * err_y.T @ Q @ err_y  - 0.5 * (self.yweight) * err_y.T @ P_y @ err_y * (self.y_inc * sigma_y * (1 - sigma_y) * err_y[1,0]) 
        self.alpha = self.alpha[0,0]
        self.beta:np.ndarray = modW_y * err_y.T @ P_y @ B_y
        self.beta = self.beta[0,0]
        beta_sq = (self.beta)**2  # 스칼라(2x1 @ 1x2 -> 1x1)
        asafe_y = - ((self.alpha + math.sqrt(self.alpha ** 2 + 1.0 * (self.beta) ** 4)) / (beta_sq)) * self.beta 


        #################################일단 x좌표 잘되는버전 세이브.
        # # Control Lyapunov function for scalar x
        # ax11 = 0.0
        # ax12 =  1.0
        # ax21 = -self.Kp1
        # ax22 = -self.Kd
        # A_x = np.array([[ax11, ax12],
        #                 [ax21, ax22]], dtype=np.float64)
        # Q = - np.eye(2)
        # P_x:np.ndarray = linalg.solve_continuous_lyapunov(A_x.T, Q)

        # overline_xpos = -xpos
        # overline_xvel = -crtvel[0,0]
        # err_x = np.array([[overline_xpos - (-self.ee_d[0,0])],
        #                     [overline_xvel]], dtype=np.float64)
        # V_x:np.ndarray = 0.5 * err_x.T @ P_x @ err_x # self.V는 2차원 1 X 1 배열 
        # V_x = V_x[0,0]

        # B_x = np.array([[0.0],
        #                 [1.0]], dtype=np.float64)

        # sigma_x = 1 / (1 + np.exp(self.x_inc * (err_x + self.ee_d[0,0] - self.unsafex)))
        # modW_x = (1 + self.xweight * sigma_x)
        # W_x = modW_x * V_x


        # Control Lyapunov function for scalar x: 좌표변환 ver.
        ax11 = 0.0
        ax12 =  1.0
        ax21 = -self.Kp1
        ax22 = -self.Kd1
        A_x = np.array([[ax11, ax12],
                        [ax21, ax22]], dtype=np.float64)
        Q_x = np.array([[1.0,-0.9],
                          [-0.9,1.0]],dtype = np.float64)
        P_x:np.ndarray = linalg.solve_continuous_lyapunov(A_x.T, -Q_x)

        overline_xpos = - xpos
        overline_xvel = - crtvel[0,0]
        overline_desired_x =  - self.ee_d[0,0]
        overline_unsafe_d = - self.unsafex
        overline_err_x = np.array([[overline_xpos - (-self.ee_d[0,0])],
                                   [overline_xvel]], dtype=np.float64)
        overline_V_x:np.ndarray = 0.5 * overline_err_x.T @ P_x @ overline_err_x # self.V는 2차원 1 X 1 배열 
        overline_V_x = overline_V_x[0,0]

        B_x = np.array([[0.0],
                        [1.0]], dtype=np.float64)

        sigma_x = 1 / (1 + np.exp(self.x_inc * (overline_err_x[0,0] + overline_desired_x - overline_unsafe_d - self.delta_x)))
        modW_x = (1 + self.xweight * sigma_x)
        W_x = modW_x * overline_V_x

        ################sontag 되는거 저장. 
        # self.alpha_x:np.ndarray = modW_x * err_x.T @ Q @ err_x
        # self.alpha_x = self.alpha_x[0,0]
        # self.beta_x:np.ndarray = modW_x * err_x.T @ P @ B_x
        # self.beta_x = self.beta_x[0,0]
        # beta_sq_x = (self.beta_x)**2  # 스칼라(2x1 @ 1x2 -> 1x1)
        # asafe_x = - ((self.alpha_x + math.sqrt(self.alpha_x ** 2 + 1.0 * (self.beta_x) ** 4)) / (beta_sq_x)) * self.beta_x 
 
        self.alpha_x = - modW_x * overline_err_x.T @ Q_x @ overline_err_x  - 0.5 * (self.xweight) * overline_V_x * (self.x_inc * sigma_x * (1 - sigma_x) * overline_xvel) 
        self.beta_x:np.ndarray = modW_x * overline_err_x.T @ P_x @ B_x
        self.beta_x = self.beta_x[0,0]
        beta_sq_x = (self.beta_x)**2  # 스칼라(2x1 @ 1x2 -> 1x1)
        overline_asafe_x = - ((self.alpha_x + math.sqrt(self.alpha_x ** 2 + 1.0 * (self.beta_x) ** 4)) / (beta_sq_x)) * self.beta_x 
        asafe_x = overline_asafe_x

        # if self.alpha < 0.0:
        #     asafe_y = asafe_y
        # else:
        #     asafe_y = 0
        #     # print(f'asafe is positive value.')

        # if self.alpha_x < 0.0:
        #     asafe_x = asafe_x
        # else:
        #     asafe_x = 0
        #     print(f'asafe_x is positive value.')

        # 이후 Fsafe를 최종 출력/사용
                # FL, Fsafe 모두 (2,1) 벡터일 때, Fsafe의 크기가 FL의 크기의 5배 이상이면 스케일 다운
        asafe = np.array([[asafe_x],
                          [asafe_y]], dtype=np.float64)
        
        Fsafe: np.ndarray = self.Mcrt @ asafe        
        norm_FL = np.linalg.norm(FL)
        norm_Fsafe = np.linalg.norm(Fsafe)

        # FL이 0보다 크고, Fsafe가 0.01 * FL 보다 크다면 반복적으로 스케일 다운
        # if norm_FL > 0:
        #     max_allowed_norm = self.proposition * norm_FL
        #     while norm_Fsafe > max_allowed_norm:
        #         scale_factor = max_allowed_norm / norm_Fsafe
        #         Fsafe *= scale_factor
        #         norm_Fsafe = np.linalg.norm(Fsafe)
        #         # 반복 과정 디버깅 필요하면 아래 프린트 사용 가능
        #         # print(f"Fsafe scaled by factor {scale_factor}, new norm: {norm_Fsafe}")

        # F = FL + Fsafe
        Norm = Point()
        Safe = Point()

        cnorm = math.sqrt(FL[0,0]**2 + FL[1,0]**2)

        safenorm = self.ksafe * math.sqrt(Fsafe[0,0]**2 + Fsafe[1,0]**2)
        if FL.shape == (2,1):
            Norm.x = float(cnorm)
            Norm.y = float(safenorm)
        else:
            Norm.x = 0.0
            Norm.y = 0.0

        self.Norm_pub.publish(Norm)
        
        F = FL + self.ksafe * Fsafe
        # F = FL
        return F


def main(args=None):
    rclpy.init(args=args)
    CTRL = controller()
    rclpy.spin(CTRL)
    rclpy.shutdown()

if __name__ == '__main__':
    main()


