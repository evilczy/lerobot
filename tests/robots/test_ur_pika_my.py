# from pathlib import Path
# import time

# from lerobot.robots.ur_pika import URPika, URPikaConfig

# robot = URPika(
#     URPikaConfig(
#         robot_ip="192.168.1.15",
#         gripper_port="/dev/ttyUSB0",
#         control_mode="joint",
#         cameras={},
#         calibration_dir=Path("./calibration/ur_pika"),
#     )
# )

# robot.connect()
# try:
#     obs = robot.get_observation()
#     print("keys:", sorted(obs.keys()))
#     print("joints:", [obs[f"joint_{i}.pos"] for i in range(1,
# 7)])
#     print("gripper:", obs["gripper.pos"])

#     action = {f"joint_{i}.pos": obs[f"joint_{i}.pos"] for i
# in range(1, 7)}
#     action["gripper.pos"] = min(obs["gripper.pos"] + 5.0,
# 30.0)
#     print("sending gripper-only action:", action)
#     robot.send_action(action)
#     time.sleep(2.0)
# finally:
#     robot.disconnect()

# """
# 2026-04-07 13:22:26,561 - pika.serial_comm - INFO - 成功连接到串口设备: /dev/ttyUSB0
# 2026-04-07 13:22:26,562 - pika.serial_comm - INFO - 启动串口读取线程
# 2026-04-07 13:22:26,563 - pika.gripper - INFO - 成功连接到Pika Gripper设备: /dev/ttyUSB0
# 2026-04-07 13:22:27,065 - lerobot.robots.ur_pika.robot_ur_pika - INFO - None URPika connected.
# keys: ['gripper.pos', 'joint_1.pos', 'joint_2.pos', 'joint_3.pos', 'joint_4.pos', 'joint_5.pos', 'joint_6.pos']
# joints: [0.12109720706939697, -2.5482126675047816, 0.6939604918109339, -2.5645858250060023, 0.2238302230834961, 0.5881397128105164]
# gripper: 18.966113303440082
# sending gripper-only action: {'joint_1.pos': 0.12109720706939697, 'joint_2.pos': -2.5482126675047816, 'joint_3.pos': 0.6939604918109339, 'joint_4.pos': -2.5645858250060023, 'joint_5.pos': 0.2238302230834961, 'joint_6.pos': 0.5881397128105164, 'gripper.pos': 23.966113303440082}
# 2026-04-07 13:22:27,066 - pika.gripper - INFO - 夹爪已设置为目标距离 23.966113303440082 mm，对应电机角度 0.5042 rad
# 2026-04-07 13:22:29,068 - pika.serial_comm - INFO - 串口读取线程已停止
# 2026-04-07 13:22:29,068 - pika.serial_comm - INFO - 读取线程已停止
# 2026-04-07 13:22:29,075 - pika.serial_comm - INFO - 已断开串口设备连接: /dev/ttyUSB0
# 2026-04-07 13:22:29,075 - pika.gripper - INFO - 已断开Pika Gripper设备连接: /dev/ttyUSB0
# 2026-04-07 13:22:29,075 - lerobot.robots.ur_pika.robot_ur_pika - INFO - None URPika disconnected.
# """



# from pathlib import Path
# import time

# from lerobot.cameras.pika import PikaCameraConfig,PikaCameraSource
# from lerobot.robots.ur_pika import URPika, URPikaConfig

# robot = URPika(
#     URPikaConfig(
#         robot_ip="192.168.1.15",
#         gripper_port="/dev/ttyUSB0",
#         control_mode="joint",
#         calibration_dir=Path("./calibration/ur_pika"),
#         cameras={
#             "front_fisheye": PikaCameraConfig(
#                 source=PikaCameraSource.FISHEYE,
#                 port="/dev/ttyUSB0",
#                 fisheye_index=0,
#                 width=640,
#                 height=480,
#                 fps=30,
#                 warmup_s=3.0,
#             ),
#         },
#     )
# )

# robot.connect()
# try:
#     obs = robot.get_observation()
#     print("keys:", sorted(obs.keys()))
#     print("joints:", [obs[f"joint_{i}.pos"] for i in range(1,
# 7)])
#     print("gripper:", obs["gripper.pos"])
#     print("front_fisheye shape:", obs["front_fisheye"].shape)

#     action = {f"joint_{i}.pos": obs[f"joint_{i}.pos"] for i
# in range(1, 7)}
#     action["gripper.pos"] = min(obs["gripper.pos"] + 5.0,
# 30.0)
#     print("sending gripper-only action:", action)
#     robot.send_action(action)
#     time.sleep(2.0)
# finally:
#     robot.disconnect()

# """
# (lerobot) czy@czy-ROG-Strix-G834JZ-G834JZ:~/code/robot/lerobot$ uv run ./tests/robots/test_ur_pika_my.py 
# 2026-04-07 13:51:44,916 - pika.serial_comm - INFO - 成功连接到串口设备: /dev/ttyUSB0
# 2026-04-07 13:51:44,917 - pika.serial_comm - INFO - 启动串口读取线程
# 2026-04-07 13:51:44,917 - pika.gripper - INFO - 成功连接到Pika Gripper设备: /dev/ttyUSB0
# 2026-04-07 13:51:45,558 - pika.camera.fisheye - INFO - 成功连接到鱼眼相机，设备ID: 0
# 2026-04-07 13:51:45,558 - pika.camera.fisheye - INFO - 启动鱼眼相机高频读取线程
# 2026-04-07 13:51:45,559 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 13:51:45,609 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 13:51:45,660 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 13:51:45,710 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 13:51:45,761 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 13:51:45,811 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 13:51:45,862 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 13:51:45,912 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 13:51:45,962 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 13:51:46,013 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# Exception in thread PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0)_read_loop:
# Traceback (most recent call last):
#   File "/home/czy/code/robot/lerobot/src/lerobot/cameras/pika/camera_pika.py", line 171, in _read_loop
#     frame = self._postprocess_image(self._read_from_hardware())
#                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/czy/code/robot/lerobot/src/lerobot/cameras/pika/camera_pika.py", line 137, in _read_from_hardware
#     raise RuntimeError(f"{self} failed to read a frame from the Pika SDK.")
# RuntimeError: PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.

# The above exception was the direct cause of the following exception:

# Traceback (most recent call last):
#   File "/home/czy/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/threading.py", line 1075, in _bootstrap_inner
#     self.run()
#   File "/home/czy/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/threading.py", line 1012, in run
#     self._target(*self._args, **self._kwargs)
#   File "/home/czy/code/robot/lerobot/src/lerobot/cameras/pika/camera_pika.py", line 191, in _read_loop
#     raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from exc
# RuntimeError: PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) exceeded maximum consecutive read failures.
# 2026-04-07 13:51:48,661 - pika.serial_comm - INFO - 串口读取线程已停止
# 2026-04-07 13:51:48,661 - pika.serial_comm - INFO - 读取线程已停止
# 2026-04-07 13:51:48,664 - pika.serial_comm - INFO - 已断开串口设备连接: /dev/ttyUSB0
# 2026-04-07 13:51:48,664 - pika.gripper - INFO - 已断开Pika Gripper设备连接: /dev/ttyUSB0
# 2026-04-07 13:51:48,682 - pika.camera.fisheye - INFO - 鱼眼相机高频读取线程已停止
# 2026-04-07 13:51:48,682 - pika.camera.fisheye - INFO - 读取线程已停止
# 2026-04-07 13:51:48,686 - pika.camera.fisheye - INFO - 已断开鱼眼相机连接，设备ID: 0
# 2026-04-07 13:51:48,727 - lerobot.robots.ur_pika.robot_ur_pika - INFO - None URPika disconnected.
# Traceback (most recent call last):
#   File "/home/czy/code/robot/lerobot/./tests/robots/test_ur_pika_my.py", line 77, in <module>
#     robot.connect()
#   File "/home/czy/code/robot/lerobot/src/lerobot/utils/decorators.py", line 39, in wrapper
#     return func(self, *args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/czy/code/robot/lerobot/src/lerobot/robots/ur_pika/robot_ur_pika.py", line 120, in connect
#     camera.connect()
#   File "/home/czy/code/robot/lerobot/src/lerobot/utils/decorators.py", line 39, in wrapper
#     return func(self, *args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/czy/code/robot/lerobot/src/lerobot/cameras/pika/camera_pika.py", line 107, in connect
#     self.async_read(timeout_ms=max(self.warmup_s * 1000, 250))
#   File "/home/czy/code/robot/lerobot/src/lerobot/utils/decorators.py", line 29, in wrapper
#     return func(self, *args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/czy/code/robot/lerobot/src/lerobot/cameras/pika/camera_pika.py", line 228, in async_read
#     raise TimeoutError(
# TimeoutError: Timed out waiting for frame from camera PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) after 3000.0 ms. Read thread alive: False.
# (lerobot) czy@czy-ROG-Strix-G834JZ-G834JZ:~/code/robot/lerobot$ uv run ./tests/robots/test_ur_pika_my.py 
# 2026-04-07 14:02:38,761 - pika.serial_comm - INFO - 成功连接到串口设备: /dev/ttyUSB0
# 2026-04-07 14:02:38,762 - pika.serial_comm - INFO - 启动串口读取线程
# 2026-04-07 14:02:38,762 - pika.gripper - INFO - 成功连接到Pika Gripper设备: /dev/ttyUSB0
# 2026-04-07 14:02:39,405 - pika.camera.fisheye - INFO - 成功连接到鱼眼相机，设备ID: 0
# 2026-04-07 14:02:39,405 - pika.camera.fisheye - INFO - 启动鱼眼相机高频读取线程
# 2026-04-07 14:02:39,406 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:02:39,456 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:02:39,507 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:02:39,557 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:02:39,608 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:02:39,658 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:02:39,709 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:02:39,759 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:02:39,810 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:02:39,860 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:02:39,911 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:02:42,525 - lerobot.cameras.pika.camera_pika - INFO - PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) connected.
# 2026-04-07 14:02:42,525 - lerobot.robots.ur_pika.robot_ur_pika - INFO - None URPika connected.
# keys: ['front_fisheye', 'gripper.pos', 'joint_1.pos', 'joint_2.pos', 'joint_3.pos', 'joint_4.pos', 'joint_5.pos', 'joint_6.pos']
# joints: [0.12106948345899582, -2.5482927761473597, 0.6940200964557093, -2.5645467243590296, 0.22381481528282166, 0.5881536602973938]
# gripper: 9.31527090853102
# front_fisheye shape: (480, 640, 3)
# sending gripper-only action: {'joint_1.pos': 0.12106948345899582, 'joint_2.pos': -2.5482927761473597, 'joint_3.pos': 0.6940200964557093, 'joint_4.pos': -2.5645467243590296, 'joint_5.pos': 0.22381481528282166, 'joint_6.pos': 0.5881536602973938, 'gripper.pos': 14.31527090853102}
# 2026-04-07 14:02:42,527 - pika.gripper - INFO - 夹爪已设置为目标距离 14.31527090853102 mm，对应电机角度 0.3234 rad
# 2026-04-07 14:02:44,542 - lerobot.cameras.pika.camera_pika - INFO - PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) disconnected.
# 2026-04-07 14:02:44,543 - pika.serial_comm - INFO - 串口读取线程已停止
# 2026-04-07 14:02:44,544 - pika.serial_comm - INFO - 读取线程已停止
# 2026-04-07 14:02:44,550 - pika.serial_comm - INFO - 已断开串口设备连接: /dev/ttyUSB0
# 2026-04-07 14:02:44,551 - pika.gripper - INFO - 已断开Pika Gripper设备连接: /dev/ttyUSB0
# 2026-04-07 14:02:44,565 - pika.camera.fisheye - INFO - 鱼眼相机高频读取线程已停止
# 2026-04-07 14:02:44,566 - pika.camera.fisheye - INFO - 读取线程已停止
# 2026-04-07 14:02:44,568 - pika.camera.fisheye - INFO - 已断开鱼眼相机连接，设备ID: 0
# 2026-04-07 14:02:44,578 - lerobot.robots.ur_pika.robot_ur_pika - INFO - None URPika disconnected.
# """



# from pathlib import Path
# import time

# from lerobot.cameras.pika import PikaCameraConfig, PikaCameraSource
# from lerobot.robots.ur_pika import URPika, URPikaConfig

# ROBOT_IP = "192.168.1.15"
# GRIPPER_PORT = "/dev/ttyUSB0"
# FISHEYE_INDEX = 0

# JOINT_KEY = "joint_6.pos"
# DELTA_RAD = 0.1  # 很小的动作，先确认方向安全

# robot = URPika(
#     URPikaConfig(
#         robot_ip=ROBOT_IP,
#         gripper_port=GRIPPER_PORT,
#         control_mode="joint",
#         calibration_dir=Path("./calibration/ur_pika"),
#         cameras={
#             "front_fisheye": PikaCameraConfig(
#                 source=PikaCameraSource.FISHEYE,
#                 port=GRIPPER_PORT,
#                 fisheye_index=FISHEYE_INDEX,
#                 width=640,
#                 height=480,
#                 fps=30,
#                 warmup_s=3.0,
#             ),
#         },
#     )
# )

# robot.connect()
# try:
#     obs0 = robot.get_observation()
#     print("keys:", sorted(obs0.keys()))
#     print("joints:", [obs0[f"joint_{i}.pos"] for i in
# range(1, 7)])
#     print("gripper:", obs0["gripper.pos"])
#     print("front_fisheye shape:",
# obs0["front_fisheye"].shape)

#     action1 = {f"joint_{i}.pos": obs0[f"joint_{i}.pos"] for i
# in range(1, 7)}
#     action1["gripper.pos"] = obs0["gripper.pos"]
#     action1[JOINT_KEY] = obs0[JOINT_KEY] + DELTA_RAD

#     print("sending move:", action1)
#     robot.send_action(action1)
#     time.sleep(2.0)

#     obs1 = robot.get_observation()
#     print("after move:", obs1[JOINT_KEY],
# obs1["front_fisheye"].shape)

#     action2 = {f"joint_{i}.pos": obs0[f"joint_{i}.pos"] for i
# in range(1, 7)}
#     action2["gripper.pos"] = obs0["gripper.pos"]

#     print("sending return:", action2)
#     robot.send_action(action2)
#     time.sleep(2.0)

# finally:
#     robot.disconnect()
# """
# (lerobot) czy@czy-ROG-Strix-G834JZ-G834JZ:~/code/robot/lerobot$ uv run ./tests/robots/test_ur_pika_my.py 
# 2026-04-07 14:21:21,202 - pika.serial_comm - INFO - 成功连接到串口设备: /dev/ttyUSB0
# 2026-04-07 14:21:21,203 - pika.serial_comm - INFO - 启动串口读取线程
# 2026-04-07 14:21:21,203 - pika.gripper - INFO - 成功连接到Pika Gripper设备: /dev/ttyUSB0
# 2026-04-07 14:21:21,845 - pika.camera.fisheye - INFO - 成功连接到鱼眼相机，设备ID: 0
# 2026-04-07 14:21:21,845 - pika.camera.fisheye - INFO - 启动鱼眼相机高频读取线程
# 2026-04-07 14:21:21,846 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:21:21,896 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:21:21,947 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:21:21,998 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:21:22,048 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:21:22,099 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:21:22,149 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:21:22,199 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:21:22,250 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:21:22,300 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:21:22,351 - lerobot.cameras.pika.camera_pika - WARNING - Error reading frame in background thread for PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0): PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) failed to read a frame from the Pika SDK.
# 2026-04-07 14:21:24,963 - lerobot.cameras.pika.camera_pika - INFO - PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) connected.
# 2026-04-07 14:21:24,964 - lerobot.robots.ur_pika.robot_ur_pika - INFO - None URPika connected.
# keys: ['front_fisheye', 'gripper.pos', 'joint_1.pos', 'joint_2.pos', 'joint_3.pos', 'joint_4.pos', 'joint_5.pos', 'joint_6.pos']
# joints: [0.1210455596446991, -2.5483013592162074, 0.6940320173846644, -2.564594884912008, 0.2237774133682251, 0.5882206559181213]
# gripper: 13.802250360696732
# front_fisheye shape: (480, 640, 3)
# sending move: {'joint_1.pos': 0.1210455596446991, 'joint_2.pos': -2.5483013592162074, 'joint_3.pos': 0.6940320173846644, 'joint_4.pos': -2.564594884912008, 'joint_5.pos': 0.2237774133682251, 'joint_6.pos': 0.6882206559181213, 'gripper.pos': 13.802250360696732}
# 2026-04-07 14:21:24,965 - pika.gripper - INFO - 夹爪已设置为目标距离 13.802250360696732 mm，对应电机角度 0.3130 rad
# after move: 0.6882384419441223 (480, 640, 3)
# sending return: {'joint_1.pos': 0.1210455596446991, 'joint_2.pos': -2.5483013592162074, 'joint_3.pos': 0.6940320173846644, 'joint_4.pos': -2.564594884912008, 'joint_5.pos': 0.2237774133682251, 'joint_6.pos': 0.5882206559181213, 'gripper.pos': 13.802250360696732}
# 2026-04-07 14:21:26,967 - pika.gripper - INFO - 夹爪已设置为目标距离 13.802250360696732 mm，对应电机角度 0.3130 rad
# 2026-04-07 14:21:28,967 - lerobot.cameras.pika.camera_pika - INFO - PikaCamera(PikaCameraSource.FISHEYE@/dev/ttyUSB0) disconnected.
# 2026-04-07 14:21:28,968 - pika.serial_comm - INFO - 串口读取线程已停止
# 2026-04-07 14:21:28,969 - pika.serial_comm - INFO - 读取线程已停止
# 2026-04-07 14:21:28,973 - pika.serial_comm - INFO - 已断开串口设备连接: /dev/ttyUSB0
# 2026-04-07 14:21:28,974 - pika.gripper - INFO - 已断开Pika Gripper设备连接: /dev/ttyUSB0
# 2026-04-07 14:21:29,005 - pika.camera.fisheye - INFO - 鱼眼相机高频读取线程已停止
# 2026-04-07 14:21:29,006 - pika.camera.fisheye - INFO - 读取线程已停止
# 2026-04-07 14:21:29,008 - pika.camera.fisheye - INFO - 已断开鱼眼相机连接，设备ID: 0
# 2026-04-07 14:21:29,026 - lerobot.robots.ur_pika.robot_ur_pika - INFO - None URPika disconnected.
# """


from pathlib import Path
import time

from lerobot.cameras.pika import PikaCameraConfig, PikaCameraSource
from lerobot.cameras.pika.shared import acquire_shared_pika_device, release_shared_pika_device
from lerobot.robots.ur_pika import URPika, URPikaConfig

ROBOT_IP = "192.168.1.15"
GRIPPER_PORT = "/dev/ttyUSB0"
REALSENSE_SERIAL = "315122272459"  # 改成你的实际 serial

robot = URPika(
    URPikaConfig(
        robot_ip=ROBOT_IP,
        gripper_port=GRIPPER_PORT,
        control_mode="joint",
        calibration_dir=Path("./calibration/ur_pika"),
        cameras={
            "wrist": PikaCameraConfig(
                source=PikaCameraSource.REALSENSE_COLOR,
                port=GRIPPER_PORT,
                realsense_serial_number=REALSENSE_SERIAL,
                width=640,
                height=480,
                fps=30,
                warmup_s=3.0,
            ),
        },
    )
)

shared = None

robot.connect()
try:
    obs = robot.get_observation()
    print("keys:", sorted(obs.keys()))
    print("joints:", [obs[f"joint_{i}.pos"] for i in range(1,
7)])
    print("gripper:", obs["gripper.pos"])
    print("wrist color shape:", obs["wrist"].shape)

    shared = acquire_shared_pika_device(GRIPPER_PORT)
    rs_cam = shared.get_realsense_camera(REALSENSE_SERIAL)

    for i in range(30):
        ok_color, color = rs_cam.get_color_frame()
        ok_depth, depth = rs_cam.get_depth_frame()
        print(
            i,
            "color:", ok_color, None if color is None else color.shape,
            "depth:", ok_depth, None if depth is None else depth.shape,
        )
        if ok_color and color is not None and ok_depth and depth is not None:
            break
        time.sleep(0.2)

finally:
    if shared is not None:
        release_shared_pika_device(shared)
    robot.disconnect()