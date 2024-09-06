# from vln.env import get_nav_from_actions
# from vln.prompt_builder import get_navigation_lines
import airsim
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import cv2
import sys
import time
sys.path.append('..')
from vln.coord_transformation import quaternion2eularian_angles
from vln.env import get_nav_from_actions
from vln.prompt_builder import get_navigation_lines

AirSimImageType = {
    0: airsim.ImageType.Scene,
    1: airsim.ImageType.DepthPlanar,
    2: airsim.ImageType.DepthPerspective,
    3: airsim.ImageType.DepthVis,
    4: airsim.ImageType.DisparityNormalized,
    5: airsim.ImageType.Segmentation,
    6: airsim.ImageType.SurfaceNormals,
    7: airsim.ImageType.Infrared
}


class Agent:
    def __init__(self, query_func, env, instance, prompt_template):
        self.query_func = query_func
        self.env = env
        self.instance = instance
        self.dataset_name = instance['dataset_name']
        self.landmarks = instance['landmarks']
        self.traffic_flow = instance.get('traffic_flow')
        self.init_prompt = prompt_template.format(instance['navigation_text'])
        self.is_map2seq = instance['is_map2seq']

    def run(self, max_steps, verbatim=False):
        actions = ['init']
        if self.dataset_name == 'map2seq':
            actions.append('forward')

        navigation_lines = list()
        is_actions = list()

        query_count = 0
        nav = get_nav_from_actions(actions, self.instance, self.env)

        step_id = 0
        hints_action = None
        while step_id <= max_steps:
            if verbatim:
                print('Number of Steps:', len(nav.actions))

            new_navigation_lines, new_is_actions = get_navigation_lines(nav,
                                                                        self.env,
                                                                        self.landmarks,
                                                                        self.traffic_flow,
                                                                        step_id=step_id
                                                                        )
            navigation_lines = navigation_lines[:-1] + new_navigation_lines
            is_actions = is_actions[:-1] + new_is_actions
            step_id = len(nav.actions)

            navigation_text = '\n'.join(navigation_lines)
            prompt = self.init_prompt + navigation_text
            # print(navigation_text)

            action, queried_api, hints_action = self.query_next_action(prompt, hints_action, verbatim)
            query_count += queried_api

            action = nav.validate_action(action)

            if action == 'stop':
                nav.step(action)
                prompt += f' {action}\n'
                break

            nav.step(action)
            if verbatim:
                print('Validated action', action)

                # print('actions', actions)
                print('query_count', query_count)

        del hints_action

        new_navigation_lines, new_is_actions = get_navigation_lines(nav,
                                                                    self.env,
                                                                    self.landmarks,
                                                                    self.traffic_flow,
                                                                    step_id=step_id,
                                                                    )
        navigation_lines = navigation_lines[:-1] + new_navigation_lines
        is_actions = is_actions[:-1] + new_is_actions

        return nav, navigation_lines, is_actions, query_count

    def query_next_action(self, prompt, hints=None, verbatim=True):
        output, queried_api, hints = self.query_func(prompt, hints)
        try:
            predicted = self.extract_next_action(output, prompt)
        except Exception as e:
            print('extract_next_action error: ', e)
            print('returned "forward" instead')
            predicted_sequence = output[len(prompt):]
            predicted = 'forward'
            print('predicted_sequence', predicted_sequence)
        if verbatim:
            print('Predicted Action:', predicted)
        return predicted, queried_api, hints

    @staticmethod
    def extract_next_action(output, prompt):
        assert output.startswith(prompt)
        predicted_sequence = output[len(prompt):]
        predicted = predicted_sequence.strip().split()[0]
        predicted = predicted.lower()
        if predicted in {'forward', 'left', 'right', 'turn_around', 'stop'}:
            return predicted

        predicted = ''.join([i for i in predicted if i.isalpha()])
        if predicted == 'turn':
            next_words = predicted_sequence.strip().split()[1:]
            next_predicted = next_words[0]
            next_predicted = ''.join([i for i in next_predicted if i.isalpha()])
            next_predicted = next_predicted.lower()
            predicted += ' ' + next_predicted
        return predicted


class LLMAgent(Agent):

    def __init__(self, llm, env, instance, prompt_template):
        self.llm = llm
        self.env = env
        self.instance = instance
        self.dataset_name = instance['dataset_name']

        self.landmarks = instance['landmarks']
        self.traffic_flow = instance.get('traffic_flow')

        self.init_prompt = prompt_template.format(instance['navigation_text'])

        cache_key = f'{self.dataset_name}_{instance["idx"]}'

        def query_func(prompt, hints=None):
            queried_api = 0
            output = self.llm.get_cache(prompt, cache_key)
            if output is None:
                print('query API')
                output = self.llm.query_api(prompt)
                queried_api += 1
                self.llm.add_to_cache(output, cache_key)
                print('api sequence')
            return output, queried_api, dict()

        super().__init__(query_func, env, instance, prompt_template)


class AirsimAgent:
    def __init__(self, cfg, query_func, prompt_template):
        self.query_func = query_func
        self.prompt_template = prompt_template
        self.landmarks = None
        self.client = airsim.MultirotorClient()
        self.actions = []
        self.states = []
        self.cfg = cfg
        self.rotation = R.from_euler("X", -np.pi).as_matrix()
        self.gt_height = 0.0
        self.velocity = 3
        self.panoid_yaws = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]

        self.init_config()

    def init_config(self):
        print("Initializing - init_config()")
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # self.client.moveToZAsync(300, 6).join()  # 上升到3m高度
        # # self.client.moveToPositionAsync(10, 0, -9, 7).join()
        # self.client.moveByRollPitchYawZAsync(0, 0, 0, -9, 2).join()
        time.sleep(2)

        cur_pos, cur_rot = self.get_current_state()
        print("initial position: {}, initial rotation: {}".format(cur_pos, cur_rot))

    def setVehiclePose(self, pose: np.ndarray) -> None:
        '''
        pose为[pos, rot]
        rot接受欧拉角或者四元数，
        如果len(pose) == 6,则认为rot为欧拉角,单位为弧度, [pitch, roll, yaw]
        如果len(pose) == 7,则认为rot为四元数, [x, y, z, w]
        '''
        pos = pose[:3]
        rot = pose[3:]

        if len(rot) == 3:
            rot = np.deg2rad(rot)
            air_rot = airsim.to_quaternion(rot[0], rot[1], rot[2])
        elif len(rot) == 4:
            air_rot = airsim.Quaternionr()
            air_rot.x_val = rot[0]
            air_rot.y_val = rot[1]
            air_rot.z_val = rot[2]
            air_rot.w_val = rot[3]
        else:
            raise ValueError(f"Expected rotation shape is (4,) or (3, ), got ({len(rot)},)")

        air_pos = airsim.Vector3r(pos[0], pos[1], pos[2])
        air_pose = airsim.Pose(air_pos, air_rot)
        self.client.simSetVehiclePose(air_pose, True)
        self.gt_height = float(air_pos.z_val)
        print(f"gt z:{self.gt_height}")
        print(f"set pose: {pos}")

    def global2body_rotation(self, global_rot, body_rot):
        # todo: assert shape
        global2body_rot = global_rot.dot(body_rot)
        return global2body_rot

    def bodyframe2worldframe(self, bodyframe):
        if type(bodyframe) is not np.ndarray:
            bodyframe = np.array(bodyframe)

        cur_pos, cur_rot = self.get_current_state()
        cur_rot = R.from_euler("XYZ", cur_rot).as_matrix()
        global2body_rot = self.global2body_rotation(cur_rot, self.rotation)
        worldframe = global2body_rot.dot(bodyframe) + cur_pos

        return worldframe

    # position is in current body frame
    def moveToPosition(self, position):
        pos_world = self.bodyframe2worldframe(position)
        print(f"next position in world coords: {pos_world}")
        self.client.moveToPositionAsync(float(pos_world[0]), float(pos_world[1]), float(pos_world[2]),
                                        self.velocity).join()

    def moveBackForth(self, distance):
        pos = [distance, 0, 0]
        self.moveToPosition(pos)

    def moveHorizontal(self, distance):
        pos = [0, distance, 0]
        self.moveToPosition(pos)

    def moveVertical(self, distance):
        pos = [0, 0, distance]
        self.moveToPosition(pos)

    def makeAction(self, act_enum):
        new_pose = self.getPoseAfterAction(act_enum)
        self.setVehiclePose(new_pose)
        # time.sleep(1)
        # self.client.moveToPositionAsync(float(new_pose[0]), float(new_pose[1]), float(new_pose[2]), 1).join()

    def getPoseAfterAction(self, act_enum):
        cur_pos, cur_rot = self.get_current_state()

        cur_pos[2] = self.gt_height

        new_pos = cur_pos
        new_rot = np.rad2deg(cur_rot)

        if act_enum == 1:           # forward 10 m
            new_pos[0] = cur_pos[0] + 10 * np.cos(cur_rot[2])
            new_pos[1] = cur_pos[1] + 10 * np.sin(cur_rot[2])
        elif act_enum == 2:         # turn left by 45 degrees
            new_rot[2] = cur_rot[2] - 90
        elif act_enum == 3:         # turn right by 45 degrees
            new_rot[2] = cur_rot[2] + 90
        elif act_enum == 4:         # go up by 5 m
            new_pos[2] = cur_pos[2] - 10
        elif act_enum == 5:         # go down by 5 m
            new_pos[2] = cur_pos[2] + 10
        else:
            print(f"Unknown action {act_enum}, keep still.")

        return np.concatenate((new_pos, new_rot), axis=0)

    """
    下面是zbn写的移动函数
    """

    def moveUp(self, distance=20):
        pos = [0, 0, distance]
        self.moveToPosition(pos)

    def moveDown(self, distance=-20):
        pos = [0, 0, distance]
        self.moveToPosition(pos)

    def moveLeft(self, distance=10):
        pos = [0, distance, 0]
        self.moveToPosition(pos)

    def moveRight(self, distance=-10):
        pos = [0, distance, 0]
        self.moveToPosition(pos)

    def moveForth(self, distance=10):
        pos = [distance, 0, 0]
        self.moveToPosition(pos)

    def moveBack(self, distance=-10):
        pos = [distance, 0, 0]
        self.moveToPosition(pos)

    def turnLeft(self, yaw=np.pi/4):
        cur_pos, cur_rot = self.get_current_state()  # get rotation in world frame
        cur_yaw_body = -cur_rot[2]  # current yaw in local body frame
        new_yaw_body = cur_yaw_body + yaw

        # print("new yaw body: {}, current yaw: {}".format(new_yaw_body, cur_yaw_body))

        # moveByRollPitchYaw is on current body frame
        self.client.moveByRollPitchYawZAsync(0, 0, float(new_yaw_body), float(cur_pos[2]), 2).join()

    def turnRight(self, yaw=-np.pi / 4):
        cur_pos, cur_rot = self.get_current_state()  # get rotation in world frame
        cur_yaw_body = -cur_rot[2]  # current yaw in local body frame
        new_yaw_body = cur_yaw_body + yaw

        # print("new yaw body: {}, current yaw: {}".format(new_yaw_body, cur_yaw_body))

        # moveByRollPitchYaw is on current body frame
        self.client.moveByRollPitchYawZAsync(0, 0, float(new_yaw_body), float(cur_pos[2]), 2).join()

    # yaw is in current body frame, radian unit
    def moveByYaw(self, yaw):
        cur_pos, cur_rot = self.get_current_state()  # get rotation in world frame
        cur_yaw_body = -cur_rot[2]  # current yaw in local body frame
        new_yaw_body = cur_yaw_body + yaw

        # print("new yaw body: {}, current yaw: {}".format(new_yaw_body, cur_yaw_body))

        # moveByRollPitchYaw is on current body frame
        self.client.moveByRollPitchYawZAsync(0, 0, float(new_yaw_body), float(cur_pos[2]), 2).join()

    def get_panorama_images(self, image_type=0):
        panorama_images = []
        new_yaws = []
        cur_pos, cur_rot = self.get_current_state()
        cur_yaw_body = -cur_rot[2]  # current yaw in body frame

        for angle in self.panoid_yaws:
            yaw = cur_yaw_body + angle
            self.client.moveByRollPitchYawZAsync(0, 0, float(yaw), float(cur_pos[2]), 2).join()
            image = self.get_front_image(image_type)
            panorama_images.append(image)

        self.client.moveByRollPitchYawZAsync(0, 0, float(cur_yaw_body), float(cur_pos[2]), 2).join()

        return panorama_images

    def get_front_image(self, image_type=0):
        # todo
        responses = self.client.simGetImages(
            [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])  # if image_type == 0:
        response = responses[0]
        if image_type == 0:
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_out = img1d.reshape(response.height, response.width, 3)
            img_out = img_out[:, :, [2, 1, 0]]
        else:
            # todo: add other image type
            img_out = None
        return img_out

    def get_xyg_image(self, image_type, cameraID):
        # todo
        # "3"地面 “4”后面 “2”前面
        if image_type == 0:
            responses = self.client.simGetImages(
                [airsim.ImageRequest(cameraID, airsim.ImageType.Scene, False, False)])  # if image_type == 0:
            response = responses[0]

            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_out = img1d.reshape(response.height, response.width, 3)
            img_out = img_out[:, :, [2, 1, 0]]
        elif image_type == 1:
            # todo: add other image type

            # 获取DepthVis深度可视图
            responses = self.client.simGetImages([
                airsim.ImageRequest(cameraID, airsim.ImageType.DepthPlanar, True, False)])
            img_depth_planar = np.array(responses[0].image_data_float).reshape(responses[0].height, responses[0].width)
            # 2. 距离100米以上的像素设为白色（此距离阈值可以根据自己的需求来更改）
            img_depth_vis = img_depth_planar / 100
            img_depth_vis[img_depth_vis > 1] = 1.
            # 3. 转换为整形
            img_out = (img_depth_vis * 255).astype(np.uint8)

            # responses = self.client.simGetImages([
            #     airsim.ImageRequest('front_center', airsim.ImageType.DepthVis, False, False)])
            # img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            # img_depthvis_bgr = img_1d.reshape(responses[0].height, responses[0].width, 3)

            # responses = self.client.simGetImages(
            #     [airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False)])  # if image_type == 0:
            # response = responses[0]

            # img1d = (np.array(response.image_data_float)*255).astype(int)
            # img_out = img1d.reshape(response.height, response.width)

        elif image_type == 11:
            # todo: add other image type

            # 获取DepthVis深度可视图
            responses = self.client.simGetImages([
                airsim.ImageRequest(cameraID, airsim.ImageType.DepthPlanar, True, False)])
            img_out = np.array(responses[0].image_data_float).reshape(responses[0].height, responses[0].width)
            # # 2. 距离100米以上的像素设为白色（此距离阈值可以根据自己的需求来更改）
            # img_depth_vis = img_depth_planar / 100
            # img_depth_vis[img_depth_vis > 1] = 1.
            # # 3. 转换为整形
            # img_out = (img_depth_vis * 255).astype(np.uint8)
        elif image_type == 2:
            # responses = self.client.simGetImages([airsim.ImageRequest(cameraID, airsim.ImageType.Segmentation, pixels_as_float=False, compress=True)])
            # # img_depth_planar = np.array(responses[0].image_data_float).reshape(responses[0].height, responses[0].width)
            #
            #
            # responses = self.client.simGetImages([airsim.ImageRequest(cameraID, airsim.ImageType.Segmentation, False, False)])
            # img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)  # get numpy array
            # img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)  # reshape array to 3 channel image array H X W X 3

            responses = self.client.simGetImages([
                airsim.ImageRequest(cameraID, airsim.ImageType.Segmentation, pixels_as_float=False, compress=True),
                airsim.ImageRequest(cameraID, airsim.ImageType.Segmentation, pixels_as_float=False, compress=False), ])

            f = open('imgs/seg.png', 'wb')
            f.write(responses[0].image_data_uint8)
            f.close()

            img1d = np.fromstring(responses[1].image_data_uint8, dtype=np.uint8)  # get numpy array
            img_out = img1d.reshape(responses[1].height, responses[1].width, 3)

            # img_out = None

        else:
            return None
        return img_out

    def get_current_state(self):
        # get world frame pos and orientation
        # orientation is in roll, pitch, yaw format
        state = self.client.simGetGroundTruthKinematics()
        pos = state.position.to_numpy_array()
        ori = quaternion2eularian_angles(state.orientation)

        return pos, ori

    # def update_coord_rot(self, axis, angle, intrinsic_rot=True):
    #     if intrinsic_rot:
    #         assert axis in ["X", "Y", "Z"]
    #         rot_mat = R.from_euler(axis, angle).as_matrix()
    #         self.coord_rot = self.coord_rot.dot(rot_mat)
    #     else:
    #         assert axis in ["x", "y", "z"]
    #         rot_mat = R.from_euler(axis, angle).as_matrix()
    #         self.coord_rot = rot_mat.dot(self.coord_rot)

#################################
    """
    下面是zdy写的一些避障和导航的函数
    _p是相对玩家坐标,_w是Airsim世界坐标
    """
    def get_image_by_yaw(self,yaw=0,image_type=0):
        """往左转是正，往右转是负"""
        cur_pos, cur_rot = self.get_current_state()
        cur_yaw_body = -cur_rot[2]   # current yaw in body frame
        target_yaw = yaw+cur_yaw_body
        # print("current_yaw:",cur_yaw_body,"target_yaw:",target_yaw)
        self.client.moveByRollPitchYawZAsync(0, 0, float(target_yaw), float(cur_pos[2]), 1).join()
        time.sleep(1)
        # self.client.rotateToYawAsync(-yaw).join()
        image = self.get_front_image(image_type)
        self.client.moveByRollPitchYawZAsync(0, 0, float(cur_yaw_body), float(cur_pos[2]), 1).join()
        # self.client.rotateToYawAsync(yaw).join()
        return image

    def get_obstacle_dis(self):
        '''
        获取正前方障碍物距离
        '''
        depth = self.get_xyg_image(image_type=1, cameraID="0")
        time.sleep(0.1)
        width,height = depth.shape
        distance = depth[2*width//5:3*width//5,height//2+1:3*height//5]/255*100
        # print("center:",depth[width//2,height//2]/255*100)
        return np.min(distance)

    def get_z_obstacle_dis(self):
        '''
        获取下方障碍物距离
        '''
        depth = self.get_xyg_image(image_type=1, cameraID="3")
        time.sleep(0.1)
        width,height = depth.shape
        distance = depth[2*width//5:3*width//5,2*height//5:3*height//5]/255*100
        return np.min(distance)

    def fly_to_p(self,goal,obstruct_dis=10,step_dis=10,target_dis=10,yaw_bias=30):
        '''
        goal是目标点坐标(相对玩家坐标)
        obstruct_dis是障碍物距离阈值
        step_dis是每次移动距离
        target_dis是目标点距离阈值
        yaw_bias是避障时的偏航角,此时正角度适用于俯视逆时针
        '''
        start_w = self.bodyframe2worldframe([0,0,0])
        # 目标世界坐标
        goal_w = self.bodyframe2worldframe(goal)
        print("target world coordinate:",goal_w)
        current_w = start_w
        yaw = math.atan2(goal_w[1]-current_w[1], goal_w[0]-current_w[0]) * 180 / math.pi
        self.client.rotateToYawAsync(yaw).join()
        #水平移动
        while np.linalg.norm(np.array(current_w[:-1]) - np.array(goal_w[:-1])) > target_dis:
            distance = self.get_obstacle_dis()
            print("obstruct_dis:",distance)
            while distance < obstruct_dis:
                yaw = yaw + yaw_bias
                self.client.rotateToYawAsync(yaw).join()
                time.sleep(0.5)
                distance = self.get_obstacle_dis()
            self.moveBackForth(step_dis)
            current_w = self.bodyframe2worldframe([0,0,0])
            yaw = math.atan2(goal_w[1]-current_w[1], goal_w[0]-current_w[0]) * 180 / math.pi
            self.client.rotateToYawAsync(yaw).join()
            #修正z坐标
            self.moveVertical(-(start_w[2]-current_w[2]))
            # if np.abs(current_w[2]-start_w[2]) > 5:
            #     self.moveVertical(start_w[2]-current_w[2])

        #竖直移动
        current_w = self.bodyframe2worldframe([0,0,0])
        z_bias = goal_w[2] - current_w[2]
        self.moveVertical(-z_bias)

    def fly_to_w(self,goal,obstruct_dis=10,step_dis=10,target_dis=10,yaw_bias=30):
        '''
        goal是目标点坐标(Airsim世界坐标)
        obstruct_dis是障碍物距离阈值
        step_dis是每次移动距离
        target_dis是目标点距离阈值
        yaw_bias是避障时的偏航角,此时正角度适用于俯视顺时针
        '''
        goal_p = self.worldframe2bodyframe(goal)
        self.fly_to_p(goal_p,obstruct_dis,step_dis,target_dis,yaw_bias)

    def move_to_ground_above(self):
        """新版:根据语义判断是否在建筑物上方"""
        """TODO 根据初始高度改变判断范围，现在如果出生点太高会导致判断错误"""
        z = self.get_z_obstacle_dis()
        if z >= 100:
            z = 100
        t = -0.0013*z+0.13
        bird_eye_view = self.get_xyg_image(image_type=2, cameraID="3")
        width,height = bird_eye_view.shape[:2]
        # 建筑物颜色是[28,67,121]BGR
        while np.equal(bird_eye_view[int(width/2-t*width):int(width/2+t*width),int(height/2-t*height):int(height/2+t*height)],[28,67,121]).any():
            self.moveBackForth(10)
            time.sleep(1)
            bird_eye_view = self.get_xyg_image(image_type=2, cameraID="3")
        if z > 50:
            self.moveBackForth(20)
        else:
            self.moveBackForth(10)
        print("moved to ground above")

    def land(self,height):
        '''配合上面的move_to_ground_above使用'''
        z_obstrucle_dis = self.get_z_obstacle_dis()
        while z_obstrucle_dis > 90:
            self.moveVertical(-80+height)
            z_obstrucle_dis = self.get_z_obstacle_dis()
        self.moveVertical(-(z_obstrucle_dis-height))
        z_obstrucle_dis = self.get_z_obstacle_dis()
        if z_obstrucle_dis > height:
            self.moveVertical(height-z_obstrucle_dis)
        print("landed")

    # """下面的环绕函数仅用于验证可行性，需要进一步修改"""
    # """TODO 将环绕到每一面的结果交给GPT"""
    # def encircle_p(self,center,radius=50,target_dis=10,step_dis=10):
    #     # center是环绕圆心，radius是半径(大概，差了建筑物的半径)
    #     center_w = self.bodyframe2worldframe(center)
    #     # 1先将无人机移动到圆周上
    #     self.fly_to_w([center_w[0]+radius,center_w[1],center_w[2]],target_dis=target_dis,step_dis=step_dis)
    #     #拍一张照
    #     current_w = self.bodyframe2worldframe([0,0,0])
    #     yaw = math.atan2(center_w[1]-current_w[1], center_w[0]-current_w[0]) * 180 / math.pi
    #     self.client.rotateToYawAsync(yaw).join()
    #     image = self.get_xyg_image(image_type=0, cameraID="0")
    #     cv2.imwrite('imgs/encircle1.jpg', image)
    #     # 2开始绕圆周飞行
    #     self.fly_to_w([center_w[0],center_w[1]-radius,center_w[2]],target_dis=target_dis,step_dis=step_dis)
    #     #拍一张照
    #     current_w = self.bodyframe2worldframe([0,0,0])
    #     yaw = math.atan2(center_w[1]-current_w[1], center_w[0]-current_w[0]) * 180 / math.pi
    #     self.client.rotateToYawAsync(yaw).join()
    #     image = self.get_xyg_image(image_type=0, cameraID="0")
    #     cv2.imwrite('imgs/encircle2.jpg', image)
    #     # 3继续绕圆周飞行
    #     self.fly_to_w([center_w[0]-radius,center_w[1],center_w[2]],target_dis=target_dis,step_dis=step_dis)
    #     #拍一张照
    #     current_w = self.bodyframe2worldframe([0,0,0])
    #     yaw = math.atan2(center_w[1]-current_w[1], center_w[0]-current_w[0]) * 180 / math.pi
    #     self.client.rotateToYawAsync(yaw).join()
    #     image = self.get_xyg_image(image_type=0, cameraID="0")
    #     cv2.imwrite('imgs/encircle3.jpg', image)
    #     # 4继续绕圆周飞行
    #     self.fly_to_w([center_w[0],center_w[1]+radius,center_w[2]],target_dis=target_dis,step_dis=step_dis)
    #     #拍一张照
    #     current_w = self.bodyframe2worldframe([0,0,0])
    #     yaw = math.atan2(center_w[1]-current_w[1], center_w[0]-current_w[0]) * 180 / math.pi
    #     self.client.rotateToYawAsync(yaw).join()
    #     image = self.get_xyg_image(image_type=0, cameraID="0")
    #     cv2.imwrite('imgs/encircle4.jpg', image)

    def align_yaw_w(self,target):
        # target是目标点坐标(Airsim世界坐标)
        target_w = target
        current_w = self.bodyframe2worldframe([0,0,0])
        yaw = math.atan2(target_w[1]-current_w[1], target_w[0]-current_w[0]) * 180 / math.pi
        self.client.rotateToYawAsync(yaw).join()

    def align_yaw_p(self,target):
        # target是目标点坐标(相对玩家坐标)
        target_w = self.bodyframe2worldframe(target)
        current_w = self.bodyframe2worldframe([0,0,0])
        yaw = math.atan2(target_w[1]-current_w[1], target_w[0]-current_w[0]) * 180 / math.pi
        self.client.rotateToYawAsync(yaw).join()
    # def update_coord_rot(self, axis, angle, intrinsic_rot=True):
    #     if intrinsic_rot:
    #         assert axis in ["X", "Y", "Z"]
    #         rot_mat = R.from_euler(axis, angle).as_matrix()
    #         self.coord_rot = self.coord_rot.dot(rot_mat)
    #     else:
    #         assert axis in ["x", "y", "z"]
    #         rot_mat = R.from_euler(axis, angle).as_matrix()
    #         self.coord_rot = rot_mat.dot(self.coord_rot)


if __name__ == "__main__":
    drone = AirsimAgent(None, None, None)
    drone.get_panorama_images()
    # drone.moveByYaw(np.pi/4)
    # img1 = drone.get_front_image()
    #
    # drone.moveBackForth(5)
    # img2 = drone.get_front_image()
    #
    # drone.moveHorizontal(5)
    # img3 = drone.get_front_image()
    #
    # drone.moveBackForth(-5)
    # drone.get_front_image()
    #
    # drone.moveHorizontal(-5)
    # drone.get_front_image()

    # import matplotlib.pyplot as plt
    # img = plt.imread("../figures/scene.png")
    # # img = cv2.imread("../figures/scene.png")
    # plt.imshow(img)
    # plt.show()

    # cv2.imshow("img", img)
    # cv2.waitKey()
