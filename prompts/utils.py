import gc
import math
import time
import random

import torch
from vln.evaluate import get_metrics_from_results
from vln.agent import Agent
from vln.env import get_gold_nav
from vln.prompt_builder import get_navigation_lines

from tqdm import tqdm
from http import HTTPStatus
import dashscope

import os
import cv2
import anthropic
import numpy as np
from PIL import Image
import base64
from openai import OpenAI
from prompts.prompt2 import build_prompt

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from scipy.stats import bootstrap
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_prompt_template():
    text = 'Navigate to the described target location!\n'
    text += 'Action Space: forward, left, right, turn_around, stop\n'
    text += 'Navigation Instructions: "{}"\n'
    instructions_prompt = text + 'Action Sequence:\n'
    return instructions_prompt


def run_navigation(model, tokenizer, instances, env, max_steps):
    model.eval()

    prompt_template = get_prompt_template()

    results = dict()
    results['prompt_template'] = prompt_template
    results['time'] = int(time.time())
    results['num_novel'] = 0
    results['instances'] = dict()
    for instance in tqdm(instances):
        torch.cuda.empty_cache()
        gc.collect()
        nav, navigation_lines, is_actions = run_navigation_instance(model,
                                                                    tokenizer,
                                                                    env,
                                                                    max_steps,
                                                                    instance,
                                                                    prompt_template,
                                                                    sample=False)

        target_panoid = instance['target_panoid']
        target_list = env.graph.get_target_neighbors(target_panoid) + [target_panoid]
        is_novel = False
        if nav.pano_path[-1] in target_list and len(nav.pano_path) - len(instance['route_panoids']) >= 2:
            is_novel = True
            results['num_novel'] += 1

        gold_nav = get_gold_nav(instance, env)
        gold_navigation_lines, gold_is_actions = get_navigation_lines(gold_nav,
                                                                      env,
                                                                      instance['landmarks'],
                                                                      instance.get('traffic_flow'))
        result = dict(idx=instance['idx'],
                      start_heading=instance['start_heading'],
                      gold_actions=gold_nav.actions,
                      gold_states=gold_nav.states,
                      gold_pano_path=instance['route_panoids'],
                      gold_navigation_lines=gold_navigation_lines,
                      gold_is_actions=gold_is_actions,
                      agent_actions=nav.actions,
                      agent_states=nav.states,
                      agent_pano_path=nav.pano_path,
                      agent_navigation_lines=navigation_lines,
                      agent_is_actions=is_actions,
                      is_novel=is_novel)

        results['instances'][result['idx']] = result

    correct, tc, spd, kpa, results = get_metrics_from_results(results, env.graph)
    return tc, spd, kpa, results


def run_navigation_instance(model, tokenizer, env, max_steps, instance, prompt_template, sample=False, sample_token_ids=None):

    def query_func(prompt, hints):
        with torch.autocast("cuda"):
            inputs = tokenizer([prompt], padding=True, return_tensors="pt").to(model.device)
            new_hints = dict(input_ids=inputs['input_ids'])

            past_key_values = None
            if hints:
                past_key_values = hints['past']
                past_input_ids = hints['input_ids']

                new_input_ids = inputs['input_ids'][0][len(past_input_ids[0]):]
                new_input_ids = torch.unsqueeze(new_input_ids, dim=0)

                inputs['input_ids'] = new_input_ids.to(model.device)

            with torch.no_grad():
                raw_outputs = model(**inputs,
                                    return_dict=True,
                                    output_hidden_states=False,
                                    output_attentions=False,
                                    use_cache=True,
                                    past_key_values=past_key_values
                                    )
                past = raw_outputs.past_key_values
                new_hints['past'] = past

            generated_logits = raw_outputs.logits.detach()[:, -1, :]
            generated_id_argmax = torch.argmax(generated_logits, dim=-1)[0].item()
            if sample:
                logits_sample_token_ids = generated_logits[0][sample_token_ids]
                m = torch.distributions.Categorical(logits=logits_sample_token_ids)
                sampled_action_id = m.sample()
                generated_id = sample_token_ids[sampled_action_id]
            else:
                generated_id = generated_id_argmax
            token = tokenizer.sp_model.IdToPiece(int(generated_id))
            output = tokenizer.sp_model.decode(token)

            if len(output) == 0:
                print('empty token generated')
                output = ' forward'

            if output[0] != ' ':
                output = ' ' + output

            if output == ' turn':
                output = ' turn_around'

            return prompt + output, 0, new_hints

    agent = Agent(query_func, env, instance, prompt_template)
    nav, navigation_lines, is_actions, _ = agent.run(max_steps, verbatim=False)
    return nav, navigation_lines, is_actions


def rl_ratio_decay(current_step, max_steps, start, end, strategy='linear'):
    start_step = start * max_steps
    end_step = end * max_steps

    if current_step <= start_step:
        return 0
    elif current_step >= end_step:
        return 1
    else:
        decay_range = end_step - start_step
        decay_step = current_step - start_step
        decay_ratio = decay_step / decay_range

        if strategy == 'cosine':
            return 1 - (0.5 * (1 + math.cos(math.pi * decay_ratio)))
        else:
            return decay_ratio



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_image_paths(folder_path):
    # Define common image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    # Retrieve all files in the folder and sort them by name
    all_files = sorted(os.listdir(folder_path))

    # Filter out image files and obtain absolute paths
    image_paths = [
        os.path.join(folder_path, f)
        for f in all_files
        if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in image_extensions
    ]

    return image_paths

class LM_client:
    """
    Large model API interaction
    """
    def __init__(self, model, api_key):
        """

        :param model: LM model
        :param api_key: api key corresponding to the LM model
        """
        self.model = model
        self.model_class = model.split('-')[0]
        if self.model_class == 'claude':
            self.client = anthropic.Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key=api_key,
            )
        elif self.model_class == 'gpt':
            self.client = OpenAI(
                api_key=api_key,
            )
        elif self.model_class == 'qwen':
            dashscope.api_key = api_key

    def query(self, messages, prompt, imgs=None):
        """

        :param messages: Historical dialogue information
        :param prompt: The prompt for the current conversation
        :param imgs: images (if exists)
        :return: updated messages, answer
        """

        # Access according to the official API input format of different models
        if self.model_class == 'claude':
            if imgs == None:
                messages.append({"role": "user", "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                ]})
            else:
                inputGPT = [
                    {
                        "type": "text",
                        "text": prompt
                    }, ]
                image1_media_type = "image/png"
                for img1 in imgs:
                    base64_image1 = encode_image(img1)
                    inputGPT += [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image1_media_type,
                                "data": base64_image1
                            }
                        }, ]

                messages.append({"role": "user", "content": inputGPT})

            try:

                chat_response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=messages,
                )

            except:
                time.sleep(20)
                chat_response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=messages,
                )

            answer = chat_response.content[0].text
            messages.append({"role": "assistant", "content": chat_response.content})

        elif self.model_class == 'gpt':
            if imgs == None:
                messages.append({"role": "user", "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                ]})
            else:
                inputGPT = [
                    {
                        "type": "text",
                        "text": prompt
                    }, ]
                for img1 in imgs:
                    base64_image1 = encode_image(img1)
                    inputGPT += [{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image1}"
                        }
                    }, ]

                messages.append({"role": "user", "content": inputGPT})

            try:

                chat_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=4000,
                )

            except:
                time.sleep(20)
                chat_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=4000,
                )

            answer = chat_response.choices[0].message.content
            # print(f'ChatGPT: {answer}')
            messages.append({"role": "assistant", "content": answer})

            return messages, answer

        elif self.model_class == 'qwen':
            if imgs == None:
                content = [{'text': prompt}]
                if len(messages) == 0:

                    messages = [{
                        'role': 'system',
                        'content': [{
                            'text': 'You are a helpful assistant.'
                        }]
                    }, {
                        'role':
                            'user',
                        'content': content
                    }]
                else:
                    messages += [{
                        'role':
                            'user',
                        'content': content
                    }]

            else:
                content = []
                for k in imgs:
                    content.append({'image': 'file://' + k})
                content.append({'text': prompt})

                if len(messages) == 0:
                    messages = [{
                        'role': 'system',
                        'content': [{
                            'text': 'You are a helpful assistant.'
                        }]
                    }, {
                        'role':
                            'user',
                        'content': content
                    }]
                else:
                    messages += [{
                        'role':
                            'user',
                        'content': content
                    }]

            try:
                response = dashscope.MultiModalConversation.call(model=self.model, messages=messages)
                answer = response.output.choices[0]['message']['content'][0]['text']

                if response.status_code == HTTPStatus.OK:
                    messages.append({'role': response.output.choices[0]['message']['role'],
                                     'content': response.output.choices[0]['message']['content']})
                else:
                    print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                        response.request_id, response.status_code,
                        response.code, response.message
                    ))
                    # If the response fails, remove the last user message from the messages list and ensure that the user/assistant messages alternate
                    messages = messages[:-1]

                return messages, answer

            except:
                return messages, 'error'

        return messages, answer


class LM_VLN:
    """
    Large model API interaction for VLN task
    """
    def __init__(self, model, api_key):
        """

        :param model: LM model
        :param api_key: api key corresponding to the LM model
        """
        self.model = model
        self.model_class = model.split('-')[0]
        if self.model_class == 'claude':
            self.llm_client = anthropic.Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key=api_key,
            )
        elif self.model_class == 'gpt':
            self.llm_client = OpenAI(
                api_key=api_key,
            )
        else:
            raise ValueError(f"Unknown evaluation model type {self.eval_model}")

    def query(self, agent_handler, messages, task_desc):

        # Access according to the official API input format of different models
        if self.model_class == 'claude':
            img1 = agent_handler.get_xyg_image(image_type=0, cameraID="0")  # 获取前景图
            img1 = Image.fromarray(img1, 'RGB')
            img1.save('imgs/1.jpg', format="jpeg")
            img2 = agent_handler.get_xyg_image(image_type=1, cameraID="0")  # 获取前景深度图
            cv2.imwrite('imgs/2.jpg', img2)
            img1 = agent_handler.get_xyg_image(image_type=0, cameraID="3")  # 获取俯视图
            img1 = Image.fromarray(img1, 'RGB')
            img1.save('imgs/3.jpg', format="jpeg")
            img2 = agent_handler.get_xyg_image(image_type=1, cameraID="3")  # 获取俯视深度图
            cv2.imwrite('imgs/4.jpg', img2)

            # Path to your image
            image1 = "imgs/1.jpg"
            image2 = "imgs/2.jpg"
            image3 = "imgs/3.jpg"
            image4 = "imgs/4.jpg"

            # Getting the base64 string
            base64_image1 = encode_image(image1)
            base64_image2 = encode_image(image2)
            base64_image3 = encode_image(image3)
            base64_image4 = encode_image(image4)

            encoded_imgs = [base64_image1, base64_image2, base64_image3, base64_image4]

            UserContent = build_prompt(task_desc)

            inputGPT = [
                {
                    "type": "text",
                    "text": UserContent
                }
            ]
            image1_media_type = "image/jpg"
            for encoded_img in encoded_imgs:
                inputGPT += [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image1_media_type,
                            "data": encoded_img
                        }
                    }
                ]

            messages.append({"role": "user", "content": inputGPT})

            try:
                chat_response = self.llm_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1000,
                    messages=messages,
                )
                answer = chat_response.content[0].text
            except:
                print('Error: LM response')
                answer = "moveForth"

            # answer = chat_response.content[0].text
            # # print(f'ChatGPT: {answer}')
            # messages.append({"role": "assistant", "content": chat_response.content})

        elif self.model_class == 'gpt':
            img1 = agent_handler.get_xyg_image(image_type=0, cameraID="0")  # 获取前景图
            img1 = Image.fromarray(img1, 'RGB')
            img1.save('imgs/1.jpg', format="jpeg")
            img2 = agent_handler.get_xyg_image(image_type=1, cameraID="0")  # 获取前景深度图
            cv2.imwrite('imgs/2.jpg', img2)
            img1 = agent_handler.get_xyg_image(image_type=0, cameraID="3")  # 获取俯视图
            img1 = Image.fromarray(img1, 'RGB')
            img1.save('imgs/3.jpg', format="jpeg")
            img2 = agent_handler.get_xyg_image(image_type=1, cameraID="3")  # 获取俯视深度图
            cv2.imwrite('imgs/4.jpg', img2)

            # Path to your image
            image1 = "imgs/1.jpg"
            image2 = "imgs/2.jpg"
            image3 = "imgs/3.jpg"
            image4 = "imgs/4.jpg"

            # Getting the base64 string
            base64_image1 = encode_image(image1)
            base64_image2 = encode_image(image2)
            base64_image3 = encode_image(image3)
            base64_image4 = encode_image(image4)

            UserContent = build_prompt(task_desc)
            # else:
            #     UserContent = input("请输入（输入 'quit' 结束）：")
            #     if UserContent == 'quit':
            #         break
            #     UserContent += "Current status:\nObservation of drones: Pan tilt angle  and 90 degrees.\nCommand:"
            # # What’s in this image?
            # What’s the similarity between this image and the previous image?

            messages.append({"role": "user", "content": [
                {
                    "type": "text",
                    "text": UserContent
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image1}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image2}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image3}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image4}"
                    }
                },
            ]})

            try:
                # time.sleep(5)
                chat_response = self.llm_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=4000,
                )
                answer = chat_response.choices[0].message.content
                # print(f'ChatGPT: {answer}')
                messages.append({"role": "assistant", "content": answer})

                print(answer)
            except:
                print('Error: LM response')
                answer = "action: moveForth"

        return answer




def calculate_cider_score(reference_texts, generated_text):
    # 创建TF-IDF向量器
    vectorizer = TfidfVectorizer(ngram_range=(1, 4))

    # 将参考描述和生成描述组合在一起进行向量化
    all_texts = reference_texts + [generated_text]
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # 计算生成描述与每个参考描述的余弦相似度
    generated_vector = tfidf_matrix[-1]
    reference_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(generated_vector, reference_vectors)

    # 计算平均相似度
    average_similarity = np.mean(similarities)

    return average_similarity

# 下载wordnet数据
nltk.download('wordnet')


def evaluate_texts(groundtruth, df1):
    # Ensure the dataframes have the same shape
    assert groundtruth.shape == df1.shape, "Shape of groundtruth and df1 must be the same."

    # Initialize lists to store scores
    cider_scores = []
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    meteor_scores = []
    rouge_scores = []
    exact_match_scores = []
    # spice_scores = []

    # Initialize CIDEr, ROUGE, and SPICE scorers
    cider_scorer = Cider()
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # spice_scorer = Spice()

    for gt, pred in zip(groundtruth, df1):
        try:
            reference = [k.split() for k in gt]
        except:
            reference = [str(k).split() for k in gt]

        try:
            candidate = pred.split()
        except:
            pred = str(pred)
            candidate = pred.split()

        # Calculate BLEU scores
        bleu1_scores.append(sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
        bleu2_scores.append(sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
        bleu3_scores.append(sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
        bleu4_scores.append(sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))

        # Calculate METEOR score
        meteor_scores.append(meteor_score(reference, candidate))

        # Calculate ROUGE score

        temp_scores = 0
        for ref in gt:
            try:
                temp_scores = np.maximum(rouge.score(ref, pred)['rougeL'].fmeasure, temp_scores)
            except:
                continue

        rouge_scores.append(temp_scores)

        # # Calculate CIDEr score
        # if len(gt.replace('.', '')) == 1 and len(pred.replace('.', '')) == 1:
        #     temp_cider = 1 if gt == pred else 0
        #     cider_scores.append(temp_cider)
        # else:
        #     cider_scores.append(calculate_cider_score([gt], pred))

        temp_cider = 0
        for ref in gt:
            try:
                temp_cider = np.maximum(calculate_cider_score([ref], pred), temp_cider)
            except:
                continue

        cider_scores.append(temp_cider)

        # Calculate SPICE score
        # spice_scores.append(spice_scorer.compute_score({0:[gt]}, {0:[pred]})[0])

        # Calculate Exact Match
        exact_match_scores.append(1 if (gt[0] == pred or gt[1] == pred) else 0)

        # Convert lists to numpy arrays for easier statistical calculations
    scores = {
        'CIDEr': np.array(cider_scores),
        'BLEU-1': np.array(bleu1_scores),
        'BLEU-2': np.array(bleu2_scores),
        'BLEU-3': np.array(bleu3_scores),
        'BLEU-4': np.array(bleu4_scores),
        'METEOR': np.array(meteor_scores),
        'ROUGE': np.array(rouge_scores),
        'Exact Match': np.array(exact_match_scores),
    }

    # Calculate mean, 2.5% percentile, and 97.5% percentile for each score
    results = {}
    for metric, values in scores.items():
        mean = np.mean(values)
        try:
            ci_lower, ci_upper = np.percentile(bootstrap((values,), np.mean, confidence_level=0.95).confidence_interval,
                                               [2.5, 97.5])
        except:
            ci_lower, ci_upper = None, None
        results[metric] = {'mean': mean, '2.5%': ci_lower, '97.5%': ci_upper}
        # results[metric] = {'mean': mean, '2.5%': 0, '97.5%': 0}

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results).transpose()
    return results_df