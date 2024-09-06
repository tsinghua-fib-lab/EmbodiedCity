import pandas as pd
import traceback
from natsort import natsorted
from utils import encode_image, get_image_paths, LM_client, evaluate_texts
import os
import time
import ast


class EmbodiedTasks:
    def __init__(self, query_dict, model, api_key):
        self.query_dict = query_dict
        self.model = model
        self.api_key = api_key

    def run(self):
        """
        Derive the responses from the LLM API
        """
        query_dict = self.query_dict

        prompt_ = open("prompts/prompt1.txt", "r").read()

        # Initializing large model
        client = LM_client(self.model, self.api_key)

        for qa in query_dict.keys():

            # Path of images in dataset
            path = 'dataset/imgs'

            # Subfolder names corresponding to samples
            list_dir = os.listdir(path)
            list_dir = natsorted(list_dir)

            if isinstance(query_dict[qa], str):
                # Perform only single round interaction with the large model

                # Initializing prompt
                prompt = prompt_ + query_dict[qa]

                # Output from LM
                label = pd.DataFrame(columns=['folder_name', 'claude3'])

                # Run each sample one by one and collect the output of LM
                for count in range(len(list_dir)):
                    error_time = 0
                    while True:
                        try:
                            # Read the embodied image observation
                            folder_name = list_dir[count]
                            folder_path = os.path.join(path, folder_name)
                            imgs = get_image_paths(folder_path)

                            # Store historical information of large model dialogues
                            messages = []

                            # Run LM to obtain output
                            messages, answer = client.query(messages, prompt, imgs)
                            print(answer)

                            # Concat Output from LM
                            label.loc[label.shape[0], :] = [folder_name, answer]
                            break
                        except Exception as e:
                            # Dealing with abnormal situations caused by frequent LM access
                            print(f"An error occurred: {e}")
                            traceback.print_exc()
                            error_time += 1
                            time.sleep(10)
                            if error_time == 3:
                                label.loc[label.shape[0], :] = [folder_name, str(e)]
                                break
                            else:
                                continue

            elif isinstance(query_dict[qa], list):
                # Perform multiple rounds of interaction with the large model

                # Initializing prompt
                prompt1 = prompt_ + query_dict[qa][0]
                prompt2 = query_dict[qa][1]

                # Output from LM
                label = pd.DataFrame(columns=['folder_name', '1', '2'])

                # Run each sample one by one and collect the output of LM
                for count in range(len(list_dir)):
                    error_time = 0
                    while True:
                        try:
                            # Read the embodied image observation
                            folder_name = list_dir[count]
                            folder_path = os.path.join(path, folder_name)
                            imgs = get_image_paths(folder_path)

                            # Store historical information of large model dialogues
                            messages = []

                            # Run LM to obtain output
                            messages, answer1 = client.query(messages, prompt1, imgs)
                            messages, answer2 = client.query(messages, prompt2)

                            # Concat Output from LM
                            label.loc[label.shape[0], :] = [folder_name, answer1, answer2]
                            break
                        except Exception as e:
                            # Dealing with abnormal situations caused by frequent LM access
                            print(f"An error occurred: {e}")
                            traceback.print_exc()
                            error_time += 1
                            time.sleep(10)
                            if error_time == 3:
                                label.loc[label.shape[0], :] = [folder_name, None, None]
                                break
                            else:
                                continue

            # Save the results
            save_path = 'results/%s_%s.csv' % (qa, model)
            label.to_csv(save_path)

    def evaluate(self):
        """
        Evaluate the model's performance
        """
        for qa in self.query_dict.keys():
            if qa == 'scene' or qa[:2] == 'qa':
                groundtruth_df = pd.read_csv('dataset/imgs_label/%s.csv' % qa, index_col=0)
                label = pd.read_csv('results/%s_%s.csv' % (qa, model), index_col=0)
                label = label.iloc[:, 1]
                groundtruth = groundtruth_df.apply(lambda row: [row[0], row[1], row[2]], axis=1)

                results_df = evaluate_texts(groundtruth, label)

            elif qa[:6] == 'dialog':

                groundtruth_df = pd.read_csv('dataset/imgs_label/%s.csv' % qa, index_col=0)
                label = pd.read_csv('results/%s_%s.csv' % (qa, model), index_col=0)
                df_extract = label.apply(lambda row: str(row[1]) + ' ' + str(row[2]), axis=1)
                label = df_extract

                # 将 'items' 列转换为列表
                groundtruth_df.iloc[:, 2] = groundtruth_df.iloc[:, 2].apply(ast.literal_eval)
                groundtruth_df.iloc[:, 3] = groundtruth_df.iloc[:, 3].apply(ast.literal_eval)

                # 合并第二列和第三列成一个列表，并生成一个新的 Series
                groundtruth = groundtruth_df.apply(lambda row: [row[0] + ' ' + row[1]] + [row[2][0] + ' ' + row[2][1]] + [row[3][0] + ' ' + row[3][1]], axis=1)

                results_df = evaluate_texts(groundtruth, label)

            elif qa[:2] == 'tp':
                groundtruth_df = pd.read_csv('dataset/imgs_label/%s.csv' % qa, index_col=0)
                label = pd.read_csv('results/%s_%s.csv' % (qa, model), index_col=0)
                label = label.iloc[:, 1]
                groundtruth = groundtruth_df.apply(lambda row: [row[1], row[2], row[3]], axis=1)

                results_df = evaluate_texts(groundtruth, label)

            print(qa, ': ')
            print(results_df['mean'])




if __name__ == '__main__':
    # Part of prompt for LM
    # Embodied first-view scene understanding ('situation')
    # Embodied question answering ('qa1', ..., 'qa10')
    # Embodied dialogue ('dialog1', 'dialog2', 'dialog3')
    # Embodied task planning ('tp1', 'tp2', 'tp3')
    query_dict = {
        'scene': 'please describe your current location, including the surrounding environment, your relationship to the environment, and any relevant spatial information.',
        'qa1': 'How many traffic lights can be observed around in total?',
        'qa2': 'Is there a building on the left side? What color is it?',
        'qa3': 'Are you facing the road, the building, or the greenery?',
        'dialog1': ['May I ask if there are any prominent waypoints around?',
                    'Where are they located respectively?'],
        'dialog2': ['May I ask what color the building on the left is?',
                    'Where is it located relative to the road ahead'],
        'dialog3': ['How many trees are there in the rear view?', 'What colors are they respectively'],
        'tp1': 'I want to have a cup of coffee at ALL-Star coffee shop, but I have not brought any money. What should I do? Please give a chain-like plan.',
        'tp2': 'I need to get an emergency medicine from the pharmacy, but I do not know the way. What should I do? Please give a chain-like plan.',
    }

    model = "xxxx"  # LM models, for example: "claude-3-haiku-20240307", "gpt-4o"
    api_key = "xxxxxxx"  # Fill in API key

    embodied_tasks = EmbodiedTasks(query_dict, model, api_key)
    embodied_tasks.run()
    embodied_tasks.evaluate()






