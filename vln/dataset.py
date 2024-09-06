import os
import json

from vln.landmarks import filter_landmarks_5shot
from vln.base_navigator import get_closest_heading

from tqdm import tqdm


def load_dataset(split, env, dataset_dir, dataset_name, landmarks_file=None, size=-1):
    print('load ' + dataset_name + ' ' + split)

    with open(landmarks_file) as f:
        landmarks = json.load(f)['instances']

    instances = list()
    with open(os.path.join(dataset_dir, 'data', f'{split}.json')) as f:
        for line in tqdm(list(f)):
            instance = dict(json.loads(line))

            if dataset_name == 'touchdown':
                instance = preprocess_touchdown_instance(env, instance)

            idx = str(instance['id'])
            instance['idx'] = idx
            instance['dataset_name'] = dataset_name

            if idx not in landmarks:
                unfiltered = []
            else:
                unfiltered = landmarks[idx]['unfiltered']
            instance['landmarks'] = filter_landmarks_5shot(unfiltered)

            instance['orig_route_panoids'] = instance['route_panoids']  # route_panoids can be overwritten by Dagger
            instance['is_novel'] = False
            instance['target_panoid'] = instance['route_panoids'][-1]
            instance['is_map2seq'] = dataset_name == 'map2seq'
            instances.append(instance)

            if size > 0 and len(instances) == size:
                break

    return instances


def preprocess_touchdown_instance(env, instance):
    def get_score(image_embeds, text):
        text_embeds = env.panoCLIP.get_text_embeds([text], prompt='{}', ignore_cache=False)
        scores = env.panoCLIP.get_logits_per_landmark(text_embeds, image_embeds)
        return scores[0].item()

    instance['id'] = instance['route_id']
    start_pano = instance['route_panoids'][0]
    start_node = env.graph.nodes[start_pano]
    start_heading = get_closest_heading(instance['start_heading'], start_node.neighbors.keys())
    instance['start_heading'] = start_heading

    if env.panoCLIP is None:
        return instance

    heading_behind = (start_heading - 180) % 360
    heading_behind = get_closest_heading(heading_behind, list(start_node.neighbors.keys()))

    flow_of_traffic_images = env.panoCLIP.flow_of_traffic_images

    # predict flow of traffic by orientation of vehicles when looking ahead and looking back
    score_with_flow = 0
    score_against_flow = 0
    for vehicle in ['car', 'truck', 'bus']:
        image_ahead_name = start_pano + '_' + str(round(start_heading)) + '_' + vehicle
        if image_ahead_name in flow_of_traffic_images:
            img = flow_of_traffic_images[image_ahead_name]
            score_rear = get_score(img, text=f'the rear view of a {vehicle}')
            score_front = get_score(img, text=f'the front view of a {vehicle}')
            score_side = get_score(img, text=f'a {vehicle} from the side')

            if score_side < max([score_rear, score_front]):
                score_with_flow += score_rear
                score_against_flow += score_front

        image_behind_name = start_pano + '_' + str(round(heading_behind)) + '_' + vehicle
        if image_behind_name in flow_of_traffic_images:
            img = flow_of_traffic_images[image_behind_name]
            score_rear = get_score(img, text=f'the rear view of a {vehicle}')
            score_front = get_score(img, text=f'the front view of a {vehicle}')
            score_side = get_score(img, text=f'a {vehicle} from the side')

            if score_side < max([score_rear, score_front]):
                score_against_flow += score_rear
                score_with_flow += score_front

    pred_flow = 'against'
    if score_with_flow > score_against_flow:
        pred_flow = 'with'

    instance['traffic_flow'] = pred_flow

    return instance
