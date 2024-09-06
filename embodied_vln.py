import os
import cv2
import anthropic
import numpy as np
from PIL import Image
from vln.agent import AirsimAgent
import base64
from openai import OpenAI
from prompts.prompt2 import build_prompt
from utils import encode_image, LM_VLN




def parse_llm_action(llm_output: str):
    command_str = llm_output.split(":")[-1]
    command_str = command_str.strip(" ")
    command_str = command_str.lower()

    act_enum = -1
    if 'stop' in command_str:
        return 0
    elif 'forth' in command_str:
        return 1
    elif 'left' in command_str:
        return 2
    elif 'right' in command_str:
        return 3
    elif 'up' in command_str:
        return 4
    elif 'down' in command_str:
        return 5
    else:
        return -1

class VLN_evaluator:
    def __init__(self, root_dir, eval_model, api_key):
        self.root_dir = root_dir
        self.eval_model = eval_model
        self.agent = AirsimAgent(None, None, None)
        self.navi_tasks = self.load_navi_task()
        self.client = LM_VLN(eval_model, api_key)

    def load_navi_task(self):
        navi_data = []

        task_file = os.path.join(self.root_dir, 'start_loc.txt')
        gt_traj_dir = os.path.join(self.root_dir, 'label')
        if not os.path.isfile(task_file):
            raise ValueError(f"Task file not found in {task_file}")

        with open(task_file, 'r') as f:
            task_data = f.readlines()

        traj_files = os.listdir(gt_traj_dir)
        traj_files = sorted(traj_files, key=lambda x: int(x.split('.')[0]))

        assert len(task_data) == len(traj_files)

        for i in range(len(task_data)):
            task_line = task_data[i]
            traj_file = os.path.join(self.root_dir, 'label', traj_files[i])

            init_pos, init_rot, task_desc = self.parse_task_line(task_line)
            gt_traj = self.parse_traj_file(traj_file)
            target_pos = init_pos + gt_traj[len(gt_traj)-1]

            gt_traj_len = 0.0
            last_pos = np.zeros(3)
            for j in range(len(gt_traj)):
                step_len = np.linalg.norm(gt_traj[j] - last_pos)
                gt_traj_len += step_len
                last_pos = gt_traj[j]

            navi_data.append({
                "start_pos": init_pos,
                "target_pos": target_pos,
                "start_rot": init_rot,
                "gt_traj": gt_traj,
                "gt_traj_len": gt_traj_len,
                "task_desc": task_desc
            })

        return navi_data

    def parse_task_line(self, task_line: str):
        task_line = task_line.strip('\n')
        items = task_line.split(';')

        pos_corp = items[0].strip(' ')
        rot_corp = items[1].strip(' ')
        desc = items[2].strip(' ')

        pos_str_items = pos_corp.split(' ')[1:]
        rot_str_items = rot_corp.split(':')[1].split(', ')

        for i in range(len(pos_str_items)):
            pos_str_items[i] = pos_str_items[i].strip(',')

        for i in range(len(rot_str_items)):
            rot_str_items[i] = rot_str_items[i].strip(' ')

        pos = list(map(float, pos_str_items))
        rot = list(map(float, rot_str_items))

        pos = np.array(pos) / 100   # cm to m
        rot = np.array(rot)

        return pos, rot, desc

    def parse_traj_file(self, traj_file: str):
        if not os.path.isfile(traj_file):
            raise ValueError(f"Trajectory file is not found in {traj_file}")
        with open(traj_file, 'r') as f:
            traj_lines = f.readlines()

        traj = []
        traj_lines = traj_lines[1:]
        for i in range(len(traj_lines)):
            traj_line = traj_lines[i].strip('\n')

            pos_str_items = traj_line.split(',')[1:]
            pos = list(map(float, pos_str_items))
            traj.append(pos)

        return np.array(traj)

    def evaluation(self):
        navi_data = self.navi_tasks
        SR_count = 0.0
        SPL = 0.0
        traj_len = 0.0
        ne_count = 0.0
        SR_short_idx = []
        SR_long_idx = []
        for idx, navi_task in enumerate(navi_data):
            if idx > 10:
                break

            traj_len = 0.0

            start_pos = navi_task["start_pos"]
            start_rot = navi_task["start_rot"]
            gt_traj = navi_task["gt_traj"]
            target_pos = navi_task["target_pos"]
            gt_traj_len = navi_task["gt_traj_len"]
            task_desc = navi_task["task_desc"]

            start_pos[2] = -start_pos[2]    # unreal coords to airsim coords

            start_pose = np.concatenate((start_pos, start_rot))
            # print(f"start pose: {start_pose}")

            self.agent.setVehiclePose(start_pose)
            # time.sleep(1)
            # self.agent.client.moveToPositionAsync(float(start_pose[0]), float(start_pose[1]), float(start_pose[2]), 1).join()
            pos, rot = self.agent.get_current_state()
            print(f"pos: {pos}, rot: {rot}")

            messages = []

            step_size = 0
            while step_size < 30:

                answer = self.client.query(self.agent, messages, task_desc)
                # print(answer)

                act = parse_llm_action(answer)
                print("action: ", act)
                if act == 0:
                    break

                self.agent.makeAction(act)
                cur_pos, cur_rot = self.agent.get_current_state()

                if act in [1, 4, 5]:
                    traj_len += 10.0

                step_size += 1

                dist = np.linalg.norm(cur_pos - target_pos)
                print(f"Task idx: {idx}, current step size: {step_size}, current dist: {dist}")

                if dist < 20:
                    break
                elif dist > 300:
                    break

            print(f"Max step size reached or target reached. step size: {step_size}")
            final_pos, final_rot = self.agent.get_current_state()
            dist = np.linalg.norm(final_pos - target_pos)
            if dist < 20:
                if gt_traj_len > 100:
                    SR_long_idx.append(idx)
                else:
                    SR_short_idx.append(idx)
                SR_count += 1
                SPL_count = gt_traj_len / max(gt_traj_len, traj_len)
                SPL += SPL_count

            ne_count += dist
            print(f"####### SR count: {SR_count}, SPL: {SPL}, NE: {ne_count}")
            # time.sleep(10)

        SR = SR_count / len(navi_data)
        NE = ne_count / len(navi_data)
        print(f"SR: {SR}, SPL: {SPL}, NE: {NE}")
        print(SR_short_idx)
        print(SR_long_idx)


if __name__ == "__main__":

    model = "xxxxx"  # LM models, for example: "claude-3-haiku-20240307", "gpt-4o"
    api_key = "xxxxxxxxx"  # Fill in API key

    vln_eval = VLN_evaluator("dataset/vln", model, api_key)
    navi_data = vln_eval.navi_tasks
    vln_eval.evaluation()
    # print(navi_data[0])

