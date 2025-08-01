import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64
import numpy as np

class system(Node):
    def __init__(self):
        super().__init__("system")
        self.pub = self.create_publisher(Float64MultiArray, 'state', 10)
        self.sub = self.create_subscription(Float64, 'input', self.input_callback,10)
        self.dt = 0.01
        self.Mass = 5; self.Bdamp = 2; self.Kspring = 10

        self.A_cont = np.array([[0.0, 0.0],
                           [0.0, 0.0]], dtype=np.float64)
        self.B_cont = np.array([[0.0],
                           [0.0]], dtype=np.float64)

        self.nx = self.B_cont.shape[0]
        self.nu = self.B_cont.shape[1]

        self.state = np.array([[1.0],[1.0]], dtype=np.float64)

        self.TIMER = self.create_timer(self.dt, self.timer_callback)

    def timer_callback(self):
        msg = Float64MultiArray()
        msg.data = self.state.T.tolist()[0]
        print(self.state)
        self.pub.publish(msg)

    def input_callback(self, sub_msg):
        # 1) 연속시간 계수 계산
        self.model()  
        A_cont, B_cont = self.A_cont, self.B_cont
        u = sub_msg.data

        # 2) 이산화 (Euler 전진법)
        Ad = np.eye(self.nx) + A_cont * self.dt
        Bd =     B_cont    * self.dt

        # 3) 상태 업데이트
        self.state = Ad @ self.state + Bd * u

    def model(self):
        self.A_cont = np.array([[0.0, 1.0],
                                [-(self.Bdamp)/(self.Mass), -(self.Kspring)/(self.Mass)]], dtype=np.float64)
        self.B_cont = np.array([[0.0],
                                [(1)/(self.Mass)]], dtype=np.float64)


def main(args = None):

    rclpy.init(args= args)

    SYS = system()
    rclpy.spin(SYS)

if __name__ == '__main__':
    main()
