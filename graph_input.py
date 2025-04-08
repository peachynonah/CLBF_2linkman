import rosbag2_py
import numpy as np
import matplotlib.pyplot as plt
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import Point
import rclpy

plt.rcParams['text.usetex'] = True

def read_topic_data(bag_path, topic_name):
    """
    주어진 bag 파일에서 topic_name(또는 '/' + topic_name)에 해당하는 메시지(Point 타입)를 읽어와서
    메시지 리스트를 반환합니다.
    """
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions()
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    data_list = []
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == topic_name or topic == "/" + topic_name:
            msg = deserialize_message(data, Point)
            data_list.append(msg)
    print(f"전체 메시지 수: {len(data_list)} for topic '{topic_name}' in {bag_path}")
    return data_list

def plot_subset_overlay(bag_paths, dt=0.001, desired_max=9.5, subset_end=1.0,
                        curve_labels=None, curve_colors=None,
                        curve_linestyles=None, curve_linewidths=None):
    """
    4개의 bag 파일에 대해 전체 데이터는 9.5초로 스케일링되지만, 
    초기 subset_end (예: 1초) 구간만 추출하여 x축이 0 ~ subset_end (여기서는 1초)로 mapping된 그래프를 그립니다.
    
    curve_labels, curve_colors, curve_linestyles, curve_linewidths 인자를 통해
    각 커브의 레전드, 색상, 선 스타일, 선 두께를 임의로 지정할 수 있습니다.
    """
    plt.figure(figsize=(12, 6))
    num_files = len(bag_paths)
    
    # 기본 레이블 설정
    if curve_labels is None:
        curve_labels = []
        for bag in bag_paths:
            if "nosafety" in bag:
                curve_labels.append("Nosafety FL input")
            else:
                curve_labels.append("Safety input")
    
    # 기본 색상 설정
    default_colors = ["blue", "green", "orange", "purple"]
    if curve_colors is None:
        curve_colors = [default_colors[i % len(default_colors)] for i in range(num_files)]
    
    # 기본 선 스타일 설정
    default_linestyles = [':', '-', '--', '-.']
    if curve_linestyles is None:
        curve_linestyles = [default_linestyles[i % len(default_linestyles)] for i in range(num_files)]
    
    # 기본 선 두께 설정
    default_linewidths = [2, 4, 6, 8]
    if curve_linewidths is None:
        curve_linewidths = [default_linewidths[i % len(default_linewidths)] for i in range(num_files)]
    
    for i, bag_path in enumerate(bag_paths):
        data = read_topic_data(bag_path, "Norm")
        if not data:
            print(f"데이터가 없습니다: {bag_path}")
            continue
        
        # 원래 시간 벡터 생성: 0초부터 (샘플 수 * dt)
        t = np.arange(len(data)) * dt
        # 전체 데이터를 desired_max (예: 9.5초)로 스케일링
        if len(t) > 0:
            scale_factor = t[-1] / desired_max
            t_scaled = t / scale_factor
        else:
            t_scaled = t
        
        # "nosafety" 파일은 메시지의 x값, 나머지는 y값 사용
        if "nosafety" in bag_path:
            values = np.array([msg.x for msg in data])
        else:
            values = np.array([msg.y for msg in data])
        
        # 초기 subset_end (예: 1초) 구간만 선택
        subset_mask = t_scaled <= subset_end
        t_subset = t_scaled[subset_mask]
        values_subset = values[subset_mask]
        
        plt.plot(t_subset, values_subset,
                 lw=curve_linewidths[i],
                 linestyle=curve_linestyles[i],
                 color=curve_colors[i],
                 label=curve_labels[i])
    
    plt.xlabel("Time (s)", fontsize=25)
    plt.ylabel("Norm", fontsize=25)
    plt.xlim(0, subset_end)  # x축 0 ~ subset_end (예: 1초)
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    rclpy.init()  # rclpy 초기화
    bag_paths = [
        "Nearly_final/nosafety",
        "Nearly_final/safety_t4",
        "Nearly_final/safety_t5",
        "Nearly_final/safety_t2"
    ]
    custom_labels = ["FL input", "safety input, gain = 0.2", "safety input, gain = 0.5", "safety input, gain = 1.5"]
    custom_colors = ["blue", "#0B6623", "orange", "purple"]
    custom_linestyles = ['-', '--', '--', '--']
    custom_linewidths = [7, 7, 7, 7]
    
    # 전체 데이터는 9.5초로 스케일링되지만, 초기 1초 구간만 추출하여 확인합니다.
    plot_subset_overlay(bag_paths, dt=0.001, desired_max=9.5, subset_end=0.5,
                        curve_labels=custom_labels, curve_colors=custom_colors,
                        curve_linestyles=custom_linestyles, curve_linewidths=custom_linewidths)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
