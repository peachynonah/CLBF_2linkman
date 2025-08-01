import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64
import numpy as np
from cvxopt import matrix, solvers

class controller(Node):
    def __init__(self):
        super().__init__("controller")
        self.pub = self.create_publisher(Float64, 'input', 10)
        self.sub = self.create_subscription(Float64MultiArray, 'state', self.state_callback,10)

        self.Mass = 5; self.Bdamp = 2; self.Kspring = 10
        self.K = np.array([10.0,0.5], dtype=np.float64)
        self.B = np.array([[0.0],[0.0]], dtype=np.float64)

        self.nx = self.B.shape[0]
        self.nu = self.B.shape[1]
        self.dt = 0.01

        self.desired_state = np.array([[-1.0],[0.0]], dtype=np.float64)



    def state_callback(self, sub_msg):
        data=sub_msg.data
        self.state = np.array(data).reshape(self.nx,-1)
        state_error = self.desired_state - self.state
        input = self.K @ state_error

        input = self.CBF_safety_filter(input)

        print(f'input: {input[0]}')   # ← 이 줄 추가!
        pub_msg = Float64()
        pub_msg.data = input[0]
        self.pub.publish(pub_msg)

    def CBF_safety_filter(self, input):
        Q = matrix([1.0])
        p = matrix([0.0])
        G = matrix([-1/(self.Mass)])
        h = matrix([((self.Bdamp)/(self.Mass))*self.state[0,0] + ((self.Kspring)/(self.Mass))*self.state[1,0] - (self.state[1,0] + 1.0) ])
        # A = matrix([0.0])
        # b = matrix([0.0])

        sol = solvers.qp(Q, p, G, h, A= None, b = None)
        u_minus_uref = float(sol['x'][0]) 

        u_safety_filtered = u_minus_uref + input

        return u_safety_filtered

def main(args = None):
    rclpy.init(args= args)

    CTRL = controller()
    rclpy.spin(CTRL)

if __name__ == '__main__':
    main()
