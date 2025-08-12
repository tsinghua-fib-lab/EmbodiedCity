import sys
import time
import airsim
import pygame
import cv2
import numpy as np
import subprocess  # 用于启动另一个程序

# 启动 record.py 程序
# record_process = subprocess.Popen([sys.executable, "record.py"])  # 启动 record.py

# >------>>>  pygame settings   <<<------< #
pygame.init()
screen = pygame.display.set_mode((320, 240))
pygame.display.set_caption('keyboard ctrl')
screen.fill((0, 0, 0))

# >------>>>  AirSim settings   <<<------< #
# 这里改为你要控制的无人机名称(settings文件里面设置的)
vehicle_name = ""
AirSim_client = airsim.MultirotorClient()
AirSim_client.confirmConnection()
AirSim_client.enableApiControl(True, vehicle_name=vehicle_name)
AirSim_client.armDisarm(True, vehicle_name=vehicle_name)
AirSim_client.takeoffAsync(vehicle_name=vehicle_name).join()

target_position = airsim.Vector3r(7481.66602, -3555.18677, -53.36726)
AirSim_client.simSetVehiclePose(airsim.Pose(target_position, airsim.Quaternionr(0, 0, 0, 1)), True)

# 基础的控制速度(m/s)
vehicle_velocity = 5.0
# 设置临时加速比例
speedup_ratio = 10.0
# 用来设置临时加速
speedup_flag = False

# 基础的偏航速率
vehicle_yaw_rate = 15.0

# 云台初始角度（Pitch, Roll, Yaw）
gimbal_pitch = 0.0  # 向上/向下
gimbal_roll = 0.0   # 左右倾斜
gimbal_yaw = 0.0    # 左右旋转

# 云台角度调整步长
gimbal_step = 0.3  # 每次调整的角度（度）

while True:
    yaw_rate = 0.0
    velocity_x = 0.0
    velocity_y = 0.0
    velocity_z = 0.0

    time.sleep(0.02)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    scan_wrapper = pygame.key.get_pressed()

    # 按下空格键加速10倍
    if scan_wrapper[pygame.K_SPACE]:
        scale_ratio = speedup_ratio
    else:
        scale_ratio = speedup_ratio / speedup_ratio

    # 根据 'A' 和 'D' 按键来设置偏航速率变量
    if scan_wrapper[pygame.K_a] or scan_wrapper[pygame.K_d]:
        yaw_rate = (scan_wrapper[pygame.K_d] - scan_wrapper[pygame.K_a]) * scale_ratio * vehicle_yaw_rate

    # 根据 'UP' 和 'DOWN' 按键来设置pitch轴速度变量(NED坐标系，x为机头向前)
    if scan_wrapper[pygame.K_UP] or scan_wrapper[pygame.K_DOWN]:
        velocity_x = (scan_wrapper[pygame.K_UP] - scan_wrapper[pygame.K_DOWN]) * scale_ratio * vehicle_velocity

    # 根据 'LEFT' 和 'RIGHT' 按键来设置roll轴速度变量(NED坐标系，y为正右方)
    if scan_wrapper[pygame.K_LEFT] or scan_wrapper[pygame.K_RIGHT]:
        velocity_y = -(scan_wrapper[pygame.K_LEFT] - scan_wrapper[pygame.K_RIGHT]) * scale_ratio * vehicle_velocity

    # 根据 'W' 和 'S' 按键来设置z轴速度变量(NED坐标系，z轴向上为负)
    if scan_wrapper[pygame.K_w] or scan_wrapper[pygame.K_s]:
        velocity_z = -(scan_wrapper[pygame.K_w] - scan_wrapper[pygame.K_s]) * scale_ratio * vehicle_velocity

    # 控制云台角度
    if scan_wrapper[pygame.K_f]:  # 向上调整云台
        gimbal_pitch += gimbal_step
    if scan_wrapper[pygame.K_g]:  # 向下调整云台
        gimbal_pitch -= gimbal_step

    # 设置云台角度
    AirSim_client.simSetCameraPose(
        camera_name="0",  # 摄像头 ID 为 "0"
        pose=airsim.Pose(
            airsim.Vector3r(0, 0, 0),  # 摄像头位置保持不变
            airsim.to_quaternion(np.radians(gimbal_pitch), np.radians(gimbal_roll), np.radians(gimbal_yaw))
        )
    )

    # print(f"Gimbal angles: Pitch={gimbal_pitch}, Roll={gimbal_roll}, Yaw={gimbal_yaw}")

    # 设置速度控制以及设置偏航控制
    AirSim_client.moveByVelocityBodyFrameAsync(vx=velocity_x, vy=velocity_y, vz=velocity_z, duration=0.02,
                                               yaw_mode=airsim.YawMode(True, yaw_or_rate=yaw_rate), vehicle_name=vehicle_name)

    if scan_wrapper[pygame.K_ESCAPE]:
        pygame.quit()
        sys.exit()
