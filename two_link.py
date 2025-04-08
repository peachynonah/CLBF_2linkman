import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point, Wrench
import numpy as np

class system(Node):
    def __init__(self):
        super().__init__("system")
        self.twist_pub = self.create_publisher(Twist, 'Joint_Twist', 10)
        self.point_pub = self.create_publisher(Point, 'Joint_Point', 10)
        self.sub = self.create_subscription(Wrench, 'Torque_input', self.input_callback, 10)
        self.dt = 0.001
        self.m = np.array([[0.8],
                           [0.8]], dtype=np.float64)
        self.l = np.array([[1.0],
                           [1.0]], dtype=np.float64)
        self.link = 2
        self.gamma = 1500
        
        # Joint dynamics 관련 변수 (모두 (2,1) 또는 (2,2))
        self.M = np.zeros((2,2), dtype=np.float64)
        self.C = np.zeros((2,1), dtype=np.float64)
        self.g = np.zeros((2,1), dtype=np.float64)
        self.input = np.zeros((2,1), dtype=np.float64)
        self.G = 9.8

        # 초기 속도와 위치: (2,1) 열 벡터로 설정
        self.vel = np.array([[-2.948],
                              [0.553]], dtype=np.float64)
        self.pos = np.array([[1.383],
                             [-2.004]], dtype=np.float64)
        self.state = np.vstack((self.pos, self.vel))

        self.TIMER = self.create_timer(self.dt, self.timer_callback_joint)

    def timer_callback_joint(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        
        for i in range(2):
            if   self.vel[i, 0] > self.gamma:
                 self.vel[i, 0] = self.gamma
            elif self.vel[i, 0] < - self.gamma:
                 self.vel[i, 0] = - self.gamma

        twist_msg.angular.x = float(self.vel[0,0])
        twist_msg.angular.y = float(self.vel[1,0])
        twist_msg.angular.z = 0.0

        point_msg = Point()
        point_msg.x = float(self.pos[0,0])
        point_msg.y = float(self.pos[1,0])
        point_msg.z = 0.0
    
        self.twist_pub.publish(twist_msg)
        self.point_pub.publish(point_msg)

        print(f'publishing joint vel : {twist_msg.angular}')
        print(f'publishing joint pos : {point_msg}\n')

    def cal_matrix(self):
        m1 = self.m[0,0]
        m2 = self.m[1,0]
        L1 = self.l[0,0]
        L2 = self.l[1,0]
        th1 = self.pos[0,0]
        th2 = self.pos[1,0]
        thd1 = self.vel[0,0]
        thd2 = self.vel[1,0]
        
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

    def propagate_joint(self, input):
        # Euler integration: (2,1) 벡터 연산
        self.pos = self.pos + self.dt * self.vel 
        self.cal_matrix()
        # Dynamics: 모든 벡터가 (2,1)인 상태에서 계산 (C와 input은 (2,1))
        self.vel = self.vel + self.dt * np.linalg.pinv(self.M) @ (input - self.C - self.g)
        # self.vel = self.vel + self.dt * np.linalg.pinv(self.M) @ (input - self.C)

    def input_callback(self, sub_msg):
        # 토크 입력을 (2,1) 열 벡터로 변환
        input_arr = np.array([[sub_msg.torque.x],
                              [sub_msg.torque.y]], dtype=np.float64)
        self.propagate_joint(input_arr)
        print(f'propagateshape : {input_arr.shape}')

def main(args=None):
    rclpy.init(args=args)
    SYS = system()
    rclpy.spin(SYS)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
