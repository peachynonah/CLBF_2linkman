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
        self.twist_sub = self.create_subscription(Twist, 'Joint_Twist', self.twist_callback, 10)
        self.point_sub = self.create_subscription(Point, 'Joint_Point', self.point_callback, 10)
        
        #튜닝: nosafety
        self.ee_d = np.array([[0.30],
                              [1.00]], dtype=np.float64)
        #
        self.Kp1 =1.5; self.Kp2 =1.0
        self.Kd1 = 1.0; self.Kd2 = 1.0
        self.dt = 0.001
        self.proposition = 1.0

        self.xpos = 0.0; self.ypos = 0.0
        #unsafe boundary
        self.unsafex = 1.3; self.unsafey = -0.3
        #parameter
        self.x_inc = 4.0; self.y_inc = 4.0
        self.xweight = 50; self.yweight = 6.1
        self.delta_x = 2.0 * 0.14; self.delta_y = 2.0 * 0.29
        self.ksafe = 0.0

        # Joint states as (2,1) 열 벡터
        self.pos_state = np.array([[0.0],
                                   [0.0]], dtype=np.float64)
        self.vel_state = np.array([[0.0],
                                   [0.0]], dtype=np.float64)
        self.crtvel = np.array([[0.0],
                                [0.0]], dtype=np.float64)

        self.poserr = np.zeros((2,1), dtype=np.float64)
        self.velerr = np.zeros((2,1), dtype=np.float64)

        self.link = 2
        # CLF, barrier shaped function and Sontag's universal formula
        self.V = 0.0
        self.W = 0.0
        self.alpha_x = 0.0; self.beta_x = 0.0; self.alpha_y = 0.0; self.beta_y = 0.0
        
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
        

        ctrl_rslt = self.cal_tau()
        
        
        pub_msg = Wrench()

        # 결과는 (2,1) 벡터이므로, 각 요소를 스칼라로 변환하여 할당
        if ctrl_rslt.shape == (2,1):
            pub_msg.torque.x = float(ctrl_rslt[0,0])
            pub_msg.torque.y = float(ctrl_rslt[1,0])
        else:
            pub_msg.torque.x = 0.0
            pub_msg.torque.y = 0.0

        self.pub.publish(pub_msg)


    def cal_jacobian(self, th1, th2, L1, L2):
        jacobian11 = -L1 * math.sin(th1) - L2 * math.sin(th1 + th2); jacobian12 = -L2 * math.sin(th1 + th2)
        jacobian21 = L1 * math.cos(th1) + L2 * math.cos(th1 + th2); jacobian22 = L2 * math.cos(th1 + th2)
        self.jacobian = np.array([[jacobian11, jacobian12],
                                  [jacobian21, jacobian22]], dtype=np.float64)


    def FK(self, th1, th2, L1, L2):
        self.xpos = L1 * math.cos(th1) + L2 * math.cos(th1 + th2)
        self.ypos = L1 * math.sin(th1) + L2 * math.sin(th1 + th2)
        print(f'eepoint is : {self.xpos, self.ypos}')
        ee_msg = Point()
        ee_msg.x = self.xpos
        ee_msg.y = self.ypos
        self.ee_pub.publish(ee_msg)

        x1pos = L1 * math.cos(th1)
        y1pos = L1 * math.sin(th1)

        pos_jnt1_msg = Point()
        pos_jnt1_msg.x = x1pos
        pos_jnt1_msg.y = y1pos
        self.pos_jnt1_pub.publish(pos_jnt1_msg)


    def dynamics_joint_space(self):
        th1 = self.pos_state[0,0]; th2 = self.pos_state[1,0]
        thd1 = self.vel_state[0,0]; thd2 = self.vel_state[1,0]
        L1 = self.l[0,0]; L2 = self.l[1,0]; m1 = self.m[0,0]; m2 = self.m[1,0]

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


    def cal_jac_dot(self):
        th1 = self.pos_state[0,0]; th2 = self.pos_state[1,0]
        thd1 = self.vel_state[0,0]; thd2 = self.vel_state[1,0]
        L1 = self.l[0,0]; L2 = self.l[1,0]; m1 = self.m[0,0]; m2 = self.m[1,0]

        # Cartesian Dynamics: Jacobian 도함수 계산
        jdot11 = -L1 * math.cos(th1) * thd1 - L2 * math.cos(th1 + th2) * (thd1 + thd2)
        jdot12 = -L2 * math.cos(th1 + th2) * (thd1 + thd2)
        jdot21 = -L1 * math.sin(th1) * thd1 - L2 * math.sin(th1 + th2) * (thd1 + thd2)
        jdot22 = -L2 * math.sin(th1 + th2) * (thd1 + thd2)
        self.jdot = np.array([[jdot11, jdot12],
                              [jdot21, jdot22]], dtype=np.float64)


    def state_error(self):
        self.poserr = np.array([[self.xpos - self.ee_d[0,0]],
                                [self.ypos - self.ee_d[1,0]]], dtype=np.float64)
        self.velerr = self.crtvel
        self.state_err = np.vstack((self.poserr, self.velerr))
        



############################### (virtual) FL force input calculation ###################################################

    def ctsn_FL(self):
        # local variable declaration
        th1 = self.pos_state[0,0]; th2 = self.pos_state[1,0]
        thd1 = self.vel_state[0,0]; thd2 = self.vel_state[1,0]
        L1 = self.l[0,0]; L2 = self.l[1,0]; m1 = self.m[0,0]; m2 = self.m[1,0]

        self.cal_jacobian(th1, th2, L1, L2)
        self.FK(th1, th2, L1, L2)
        
        # Cartesian velocity
        self.crtvel = self.jacobian @ np.array([[thd1],
                                           [thd2]], dtype=np.float64)
        
        self.state_error()
        self.dynamics_joint_space()
        self.cal_jac_dot()

        # Jacobian의 의사역행렬 계산
        self.jpinv = np.linalg.pinv(self.jacobian)
        invj = self.jpinv

        # Cartesian model calculation
        self.Mcrt: np.ndarray = invj.T @ self.M @ invj
        vel_vec = np.array([[thd1],
                            [thd2]], dtype=np.float64)
        self.Ccrt = invj.T @ (self.C - self.M @ invj @ self.jdot @ vel_vec)
        self.gcrt = invj.T @ self.g

        # FL 제어 (Cartesian 공간에서)
        # 오차 제어 항: (2,1) 열 벡터로 구성
        cartesian_error_term = np.array([[self.Kp1 * self.poserr[0,0] + self.Kd1 * self.velerr[0,0]],
                                    [self.Kp2 * self.poserr[1,0] + self.Kd2 * self.velerr[1,0]]], dtype=np.float64)
        # 제어력 계산
        Fcomp: np.ndarray = - self.Mcrt @ cartesian_error_term + self.Ccrt + self.gcrt # (2,1)
        # Fcomp: np.ndarray = - self.Mcrt @ cartesian_error + self.Ccrt # (2,1)
        
        # 제어력을 관절 토크로 매핑 (역Jacobian 전치)
        force_FL = self.jacobian.T @ Fcomp  # (2,1)
        return force_FL


###############################CLBF based safety input calculation and related functions ###################################################


    def F_safe(self):
        # Control Lyapunov function for scalar y
        A_y = np.array([[0.0, 1.0],
                      [-self.Kp2, -self.Kd2]], dtype=np.float64)
        B_y = np.array([[0.0],
                        [1.0]], dtype=np.float64)
        
        ## Lyapunov equation calculation
        Q_y = np.array([[1.0, -0.9],      
                      [-0.9, 1.0]])        
        P_y:np.ndarray = linalg.solve_continuous_lyapunov(A_y.T, -Q_y)
        err_y = np.array([[self.ypos - self.ee_d[1,0]],
                            [self.crtvel[1,0]]], dtype=np.float64)
        V_y:np.ndarray = 0.5 * err_y.T @ P_y @ err_y; V_y = V_y[0,0]

        ## CLBF construction
        sigma_y = 1 / (1 + np.exp(self.y_inc * (err_y + self.ee_d[1,0] - self.unsafey - self.delta_y)))
        modW_y = (1 + self.yweight * sigma_y)
        W_y = modW_y * V_y
        
        ## Sontag's universal formula 
        self.alpha_y:np.ndarray = - modW_y * err_y.T @ Q_y @ err_y  - 0.5 * (self.yweight) * err_y.T @ P_y @ err_y * (self.y_inc * sigma_y * (1 - sigma_y) * err_y[1,0]) 
        self.beta_y:np.ndarray = modW_y * err_y.T @ P_y @ B_y
        asafe_y = self.sontag_input(self.alpha_y[0,0], self.beta_y[0,0])



        # Control Lyapunov function for scalar x: 좌표변환 ver.
        ol_xpos, ol_xvel, ol_desired_xpos, ol_unsafe_xd = self.coordinate_transformation_only_sign(self.xpos, self.crtvel[0,0], self.ee_d[0,0], self.unsafex)

        A_x = np.array([[0.0, 1.0],
                        [-self.Kp1, -self.Kd1]], dtype=np.float64)
        B_x = np.array([[0.0],
                        [1.0]], dtype=np.float64)
        
        ## Lyapunov equation calculation
        Q_x = np.array([[1.0,-0.9],
                        [-0.9,1.0]],dtype = np.float64)
        P_x:np.ndarray = linalg.solve_continuous_lyapunov(A_x.T, -Q_x)
        ol_err_x = np.array([[ol_xpos - ol_desired_xpos],
                                   [ol_xvel]], dtype=np.float64)
        ol_V_x:np.ndarray = 0.5 * ol_err_x.T @ P_x @ ol_err_x; ol_V_x = ol_V_x[0,0] # self.V는 2차원 1 X 1 배열

        ## CLBF construction
        sigma_x = 1 / (1 + np.exp(self.x_inc * (ol_err_x[0,0] + ol_desired_xpos - ol_unsafe_xd - self.delta_x)))
        ol_modW_x = (1 + self.xweight * sigma_x)
        ol_W_x = ol_modW_x * ol_V_x
 
        ## Sontag's universal formula 
        self.alpha_x = - ol_modW_x * ol_err_x.T @ Q_x @ ol_err_x  - 0.5 * (self.xweight) * ol_V_x * (self.x_inc * sigma_x * (1 - sigma_x) * ol_xvel) 
        self.beta_x:np.ndarray = ol_modW_x * ol_err_x.T @ P_x @ B_x
        asafe_x = self.sontag_input(self.alpha_x, self.beta_x[0,0])


        ######### (virtual) safety force calculation
        asafe = np.array([[asafe_x],
                          [asafe_y]], dtype=np.float64)
        
        force_safe: np.ndarray = self.Mcrt @ asafe        
        return force_safe


    def coordinate_transformation_only_sign(self, pos, vel, desired_pos, unsafe_d):
        ol_pos = - pos; ol_vel = - vel
        ol_desired_pos = - desired_pos
        ol_unsafe_d = - unsafe_d

        return ol_pos, ol_vel, ol_desired_pos, ol_unsafe_d

    def sontag_input(self, alpha, beta):
        sontag_input = - ((alpha + math.sqrt(alpha ** 2 + 1.0 * (beta) ** 4)) / ( beta ** 2)) * beta 
        return sontag_input


############################## (vitual) total input calculation ############################

    def cal_tau(self):

        force_safe = self.F_safe()
        force_FL = self.ctsn_FL()

        Norm = Point()

        cnorm = math.sqrt(force_FL[0,0]**2 + force_FL[1,0]**2)

        safenorm = self.ksafe * math.sqrt(force_safe[0,0]**2 + force_safe[1,0]**2)
        if force_FL.shape == (2,1):
            Norm.x = float(cnorm)
            Norm.y = float(safenorm)
        else:
            Norm.x = 0.0
            Norm.y = 0.0

        self.Norm_pub.publish(Norm)
        
        total_F = force_FL + self.ksafe * force_safe
        return total_F



def main(args=None):
    rclpy.init(args=args)
    CTRL = controller()
    rclpy.spin(CTRL)
    rclpy.shutdown()

if __name__ == '__main__':
    main()


