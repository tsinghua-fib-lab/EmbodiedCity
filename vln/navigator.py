import numpy as np
from prompt_builder import get_observations_str
from clip import PanoCLIP
from agent import AirsimAgent
import sys
sys.path.append('..')
from llm.query_llm import OpenAI_LLM

class MultirotorNavigator:
    def __init__(self, agent, llm, vlm, prompt_template):
        self.agent = agent
        self.llm = llm
        self.vlm = vlm
        self.prompt_template = prompt_template
        self.image_threshold = 0

    def run(self, max_steps, verbatim=False):
        step_id = 0
        observations = dict()

        while step_id < max_steps:
            if verbatim:
                print("Number of steps:", len(self.agent.get_actions()))

            # get observation scores
            vis_obs = self.agent.get_panorama_images()

            landmarks = ["trees", "tall building", "long white building", "black fence", "green wall"]
            scores = self.vlm.get_scores_v2(vis_obs, landmarks, prompt="picture of {}", ignore_cache=True)
            mean_stds = self.vlm.get_landmarks_mean_std_v2(scores)

            print("scores:", scores)

            observed_landmarks = [list() for _ in range(5)]
            for landmark in landmarks:

                score = max(scores[landmark])
                max_id = np.argmax(scores[landmark])

                mean, std = mean_stds[landmark]
                score = (score - mean) / std

                if score >= self.image_threshold:
                    observed_landmarks[max_id].append(landmark)

            print("observed_landmarks:", observed_landmarks)

            observations['landmarks'] = observed_landmarks
            observation_str = get_observations_str(observations)
            print(observation_str)

            print("based on the current observation, the next action is {}".format("Move forward"))
            print()
            agent.moveBackForth(5)



if __name__ == "__main__":
    agent = AirsimAgent(None, None, None)

    llm = None  # OpenAI_LLM()
    vlm = PanoCLIP(model_name="openclip", device="cpu", cache_dir="E:\\py-pro\\AirVelma\\features")
    prompt_template = None
    nav = MultirotorNavigator(agent, llm, vlm, prompt_template)
    nav.run(5)
