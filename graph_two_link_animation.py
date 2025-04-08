import rosbag2_py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Polygon
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import Point

plt.rcParams.update({
    "text.usetex": True,                   # LaTeX 렌더링 활성화
    "font.family": "serif",                # 글꼴 설정 (원하는 글꼴 선택)
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}"  # 필요한 LaTeX 패키지 추가
})

def read_bag_data(bag_path):
    """
    ROS2 Bag 파일에서 /Crts_ee_point와 /Crts_jnt1_point 토픽 데이터를 읽어와
    각각 x, y 리스트로 반환 (각 리스트의 길이는 같다고 가정)
    """
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions()
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    ee_topic = "/Crts_ee_point"
    jnt1_topic = "/Crts_jnt1_point"

    ee_x_vals, ee_y_vals = [], []
    jnt1_x_vals, jnt1_y_vals = [], []

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == ee_topic:
            msg = deserialize_message(data, Point)
            ee_x_vals.append(msg.x)
            ee_y_vals.append(msg.y)
        elif topic == jnt1_topic:
            msg = deserialize_message(data, Point)
            jnt1_x_vals.append(msg.x)
            jnt1_y_vals.append(msg.y)
            
    return (ee_x_vals, ee_y_vals), (jnt1_x_vals, jnt1_y_vals)

def animate_motion(ee_data_a, jnt1_data_a, ee_data_b, jnt1_data_b,
                   ee_data_c, jnt1_data_c, ee_data_d, jnt1_data_d):
    """
    4개의 bag 파일 데이터를 받아 (0,0), Crts_jnt1_point, 그리고 Crts_ee_point를 잇는 강체의 모션을
    4개의 trajectory로 동시에 애니메이션으로 표시합니다.
      - rosbag_a: 터쿼이즈 (#0B6623)
      - rosbag_b: 파란색 (blue)
      - rosbag_c: 주황색 (orange)
      - rosbag_d: 보라색 (purple)
      
    추가로, unsafe 영역을 빨간색 빗금으로 표시합니다.
    """
    # 데이터 언패킹
    ee_x_a, ee_y_a = ee_data_a
    jnt1_x_a, jnt1_y_a = jnt1_data_a
    ee_x_b, ee_y_b = ee_data_b
    jnt1_x_b, jnt1_y_b = jnt1_data_b
    ee_x_c, ee_y_c = ee_data_c
    jnt1_x_c, jnt1_y_c = jnt1_data_c
    ee_x_d, ee_y_d = ee_data_d
    jnt1_x_d, jnt1_y_d = jnt1_data_d

    # 4개 bag파일 중 짧은 길이에 맞춤
    num_frames = min(len(ee_x_a), len(jnt1_x_a),
                     len(ee_x_b), len(jnt1_x_b),
                     len(ee_x_c), len(jnt1_x_c),
                     len(ee_x_d), len(jnt1_x_d))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("position 1", fontsize=15)
    ax.set_ylabel("position 2", fontsize=15)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal')
    base_point = np.array([0, 0])

    # 색상 및 라벨 설정
    color_a = "blue"   # rosbag_a
    color_b = "#0B6623"       # rosbag_b
    color_c = "orange"     # rosbag_c
    color_d = "purple"     # rosbag_d

    # rosbag_a 관련 객체
    line1_a, = ax.plot([], [], '-', lw=8, color=color_a)
    line2_a, = ax.plot([], [], '-', lw=8, color=color_a)
    base_dot_a, = ax.plot([], [], 'ko', markersize=15)
    jnt1_dot_a, = ax.plot([], [], 'o', color=color_a, markersize=15)
    ee_dot_a, = ax.plot([], [], 'o', color=color_a, markersize=8)
    traj_line_a, = ax.plot([], [], '--', lw=3, color=color_a, label="no safety input", zorder=3)

    # rosbag_b 관련 객체
    line1_b, = ax.plot([], [], '-', lw=8, color=color_b)
    line2_b, = ax.plot([], [], '-', lw=8, color=color_b)
    base_dot_b, = ax.plot([], [], 'ko', markersize=15)
    jnt1_dot_b, = ax.plot([], [], 'o', color=color_b, markersize=15)
    ee_dot_b, = ax.plot([], [], 'o', color=color_b, markersize=8)
    traj_line_b, = ax.plot([], [], '--', lw=3, color=color_b, label="$\mathbf{k}_{\textrm{safe}}$ = 0.2", zorder=3)

    # rosbag_c 관련 객체
    line1_c, = ax.plot([], [], '-', lw=8, color=color_c)
    line2_c, = ax.plot([], [], '-', lw=8, color=color_c)
    base_dot_c, = ax.plot([], [], 'ko', markersize=15)
    jnt1_dot_c, = ax.plot([], [], 'o', color=color_c, markersize=15)
    ee_dot_c, = ax.plot([], [], 'o', color=color_c, markersize=8)
    traj_line_c, = ax.plot([], [], '--', lw=3, color=color_c, label="gain = 0.5", zorder=3)

    # rosbag_d 관련 객체
    line1_d, = ax.plot([], [], '-', lw=8, color=color_d)
    line2_d, = ax.plot([], [], '-', lw=8, color=color_d)
    base_dot_d, = ax.plot([], [], 'ko', markersize=15)
    jnt1_dot_d, = ax.plot([], [], 'o', color=color_d, markersize=15)
    ee_dot_d, = ax.plot([], [], 'o', color=color_d, markersize=8)
    traj_line_d, = ax.plot([], [], '--', lw=3, color=color_d, label="gain = 1.5", zorder=3)

    # 축 범위 설정
    ax.set_xlim(-1.0, 2.0)
    ax.set_ylim(-1.0, 1.5)

    # unsafe 영역 설정 (x > k₁, y < k₂인 L-자 형태)
    k1 = 1.3   # x > k₁ 영역
    k2 = -0.3  # y < k₂ 영역

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    vertices = [(x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (k1, y_max),
                (k1, k2),
                (x_min, k2)]
    
    unsafe_patch = Polygon(vertices, closed=True, hatch='xxxx', fill=False, 
                           edgecolor='red', lw=2, zorder=0, label="Unsafe Region")
    ax.add_patch(unsafe_patch)

    ax.legend(loc='upper right', fontsize=15)
    
    def init():
        # rosbag_a 초기화
        line1_a.set_data([], [])
        line2_a.set_data([], [])
        base_dot_a.set_data([], [])
        jnt1_dot_a.set_data([], [])
        ee_dot_a.set_data([], [])
        traj_line_a.set_data([], [])
        # rosbag_b 초기화
        line1_b.set_data([], [])
        line2_b.set_data([], [])
        base_dot_b.set_data([], [])
        jnt1_dot_b.set_data([], [])
        ee_dot_b.set_data([], [])
        traj_line_b.set_data([], [])
        # rosbag_c 초기화
        line1_c.set_data([], [])
        line2_c.set_data([], [])
        base_dot_c.set_data([], [])
        jnt1_dot_c.set_data([], [])
        ee_dot_c.set_data([], [])
        traj_line_c.set_data([], [])
        # rosbag_d 초기화
        line1_d.set_data([], [])
        line2_d.set_data([], [])
        base_dot_d.set_data([], [])
        jnt1_dot_d.set_data([], [])
        ee_dot_d.set_data([], [])
        traj_line_d.set_data([], [])
        return (line1_a, line2_a, base_dot_a, jnt1_dot_a, ee_dot_a, traj_line_a,
                line1_b, line2_b, base_dot_b, jnt1_dot_b, ee_dot_b, traj_line_b,
                line1_c, line2_c, base_dot_c, jnt1_dot_c, ee_dot_c, traj_line_c,
                line1_d, line2_d, base_dot_d, jnt1_dot_d, ee_dot_d, traj_line_d)

    def update(frame):
        # rosbag_a 업데이트
        jnt1_a = np.array([jnt1_x_a[frame], jnt1_y_a[frame]])
        ee_a = np.array([ee_x_a[frame], ee_y_a[frame]])
        line1_a.set_data([base_point[0], jnt1_a[0]], [base_point[1], jnt1_a[1]])
        line2_a.set_data([jnt1_a[0], ee_a[0]], [jnt1_a[1], ee_a[1]])
        base_dot_a.set_data(base_point[0], base_point[1])
        jnt1_dot_a.set_data(jnt1_a[0], jnt1_a[1])
        ee_dot_a.set_data(ee_a[0], ee_a[1])
        traj_line_a.set_data(ee_x_a[:frame+1], ee_y_a[:frame+1])
        
        # rosbag_b 업데이트
        jnt1_b = np.array([jnt1_x_b[frame], jnt1_y_b[frame]])
        ee_b = np.array([ee_x_b[frame], ee_y_b[frame]])
        line1_b.set_data([base_point[0], jnt1_b[0]], [base_point[1], jnt1_b[1]])
        line2_b.set_data([jnt1_b[0], ee_b[0]], [jnt1_b[1], ee_b[1]])
        base_dot_b.set_data(base_point[0], base_point[1])
        jnt1_dot_b.set_data(jnt1_b[0], jnt1_b[1])
        ee_dot_b.set_data(ee_b[0], ee_b[1])
        traj_line_b.set_data(ee_x_b[:frame+1], ee_y_b[:frame+1])
        
        # rosbag_c 업데이트
        jnt1_c = np.array([jnt1_x_c[frame], jnt1_y_c[frame]])
        ee_c = np.array([ee_x_c[frame], ee_y_c[frame]])
        line1_c.set_data([base_point[0], jnt1_c[0]], [base_point[1], jnt1_c[1]])
        line2_c.set_data([jnt1_c[0], ee_c[0]], [jnt1_c[1], ee_c[1]])
        base_dot_c.set_data(base_point[0], base_point[1])
        jnt1_dot_c.set_data(jnt1_c[0], jnt1_c[1])
        ee_dot_c.set_data(ee_c[0], ee_c[1])
        traj_line_c.set_data(ee_x_c[:frame+1], ee_y_c[:frame+1])
        
        # rosbag_d 업데이트
        jnt1_d = np.array([jnt1_x_d[frame], jnt1_y_d[frame]])
        ee_d = np.array([ee_x_d[frame], ee_y_d[frame]])
        line1_d.set_data([base_point[0], jnt1_d[0]], [base_point[1], jnt1_d[1]])
        line2_d.set_data([jnt1_d[0], ee_d[0]], [jnt1_d[1], ee_d[1]])
        base_dot_d.set_data(base_point[0], base_point[1])
        jnt1_dot_d.set_data(jnt1_d[0], jnt1_d[1])
        ee_dot_d.set_data(ee_d[0], ee_d[1])
        traj_line_d.set_data(ee_x_d[:frame+1], ee_y_d[:frame+1])
        
        return (line1_a, line2_a, base_dot_a, jnt1_dot_a, ee_dot_a, traj_line_a,
                line1_b, line2_b, base_dot_b, jnt1_dot_b, ee_dot_b, traj_line_b,
                line1_c, line2_c, base_dot_c, jnt1_dot_c, ee_dot_c, traj_line_c,
                line1_d, line2_d, base_dot_d, jnt1_dot_d, ee_dot_d, traj_line_d)

    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                         interval=0.5, blit=True, repeat=False)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.show()

if __name__ == "__main__":
    # bag 파일 경로 (필요에 따라 경로 수정)
    bag_path_a = "Nearly_final/nosafety_1"
    bag_path_b = "Nearly_final/safety_t4"
    bag_path_c = "Nearly_final/safety_t5"  # 새로 추가된 bag 파일 1
    bag_path_d = "Nearly_final/safety_t2"  # 새로 추가된 bag 파일 2
    
    # 각 bag파일에서 데이터 읽기
    ee_data_a, jnt1_data_a = read_bag_data(bag_path_a)
    ee_data_b, jnt1_data_b = read_bag_data(bag_path_b)
    ee_data_c, jnt1_data_c = read_bag_data(bag_path_c)
    ee_data_d, jnt1_data_d = read_bag_data(bag_path_d)
    
    animate_motion(ee_data_a, jnt1_data_a,
                   ee_data_b, jnt1_data_b,
                   ee_data_c, jnt1_data_c,
                   ee_data_d, jnt1_data_d)
