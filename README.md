two_link: manipulator system model (described in joint space) 

controller_twolink: controller of two link manipulator, with feedback linearization and added safety input. It is described in task space. joint space <> task space transfer is also included in this code.
safety input is decoupled as x and y, respectively.(to apply elementary results which are derived with a 2-nd order scalar system.)

graph_two_link_animation: trajectory of manipulator.

graph_input: graph of 2-norm of input. Comparison of FL input and added safety input is also shown.

Onpaper_datas: ROS2 bag file datas of simulation situations on the submitted paper.
