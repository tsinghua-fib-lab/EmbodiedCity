import numpy as np

from vln.base_navigator import BaseNavigator, get_relative_angle, get_closest_heading
from vln.graph_loader import GraphLoader

from tqdm import tqdm


class ClipEnv:

    def __init__(self, graph_dir, panoCLIP=None, image_threshold=3.5, image_prompt='{}', position_alignment=False):
        self.graph = GraphLoader(graph_dir).construct_graph()

        self.panoCLIP = panoCLIP
        self.image_threshold = image_threshold
        self.image_prompt = image_prompt
        self.position_alignment = position_alignment
        self.seen_panoids = {n.panoid for n in self.graph.nodes.values() if n.partition == 'seen'}

    def get_observations(self, states, step_id, landmarks, traffic_flow=None):
        observations = dict()

        avg_steps = 40  # map2seq
        if traffic_flow:
            avg_steps = 37  # touchdown
        step_pos = min(1.0, step_id / avg_steps)

        state = states[step_id]
        prev_state = None, None
        if step_id > 0:
            prev_state = states[step_id-1]

        panoid, heading = state
        prev_panoid, prev_heading = prev_state

        # intersection
        num_neighbors = self.graph.get_num_neighbors(panoid)
        if num_neighbors > 2 and panoid != prev_panoid:
            observations['intersection'] = num_neighbors

        # image
        if self.panoCLIP is not None:
            pano_yaw = self.graph.nodes[panoid].pano_yaw_angle
            scores = self.panoCLIP.get_scores(panoid, pano_yaw, round(heading), landmarks, prompt=self.image_prompt)
            mean_stds = self.panoCLIP.get_landmarks_mean_std(landmarks,
                                                             prompt=self.image_prompt,
                                                             panoids=self.seen_panoids)

            observed_landmarks = [list() for _ in range(5)]
            for landmark in landmarks:

                score = max(scores[landmark])
                max_id = np.argmax(scores[landmark])

                mean, std = mean_stds[landmark]
                score = (score - mean) / std

                if score >= self.image_threshold:
                    observed_landmarks[max_id].append(landmark)

            observations['landmarks'] = observed_landmarks

        if step_id == 0 and traffic_flow is not None:
            observations['traffic_flow'] = traffic_flow

        return observations


def get_nav_from_actions(actions, instance, env):
    nav = BaseNavigator(env)

    start_heading = instance['start_heading']
    start_panoid = instance['route_panoids'][0]
    nav.init_state(panoid=start_panoid,
                   heading=start_heading)

    for i, action in enumerate(actions):
        nav.step(action)

    assert nav.actions == actions
    return nav


def get_gold_nav(instance, env):
    nav = BaseNavigator(env)

    start_heading = instance['start_heading']
    start_panoid = instance['route_panoids'][0]
    nav.init_state(panoid=start_panoid,
                   heading=start_heading)

    gt_action = None
    while gt_action != 'stop':
        gt_action = get_gt_action(nav, gt_path=instance['route_panoids'])
        nav.step(gt_action)

    test_nav = BaseNavigator(env)
    test_nav.init_state(panoid=start_panoid,
                        heading=start_heading)

    for action in nav.actions:
        test_nav.step(action)

    assert nav.pano_path == instance['route_panoids']
    assert nav.get_state() == test_nav.get_state()
    assert nav.get_state()[0] == instance['route_panoids'][-1]
    return nav


def get_gt_action(nav, gt_path):
    target_panoid = gt_path[-1]
    curr_panoid, curr_heading = nav.get_state()
    curr_node = nav.env.graph.nodes[curr_panoid]

    if curr_panoid in gt_path:
        num_occurrences = gt_path.count(curr_panoid)
        if num_occurrences == 1:
            pano_index = gt_path.index(curr_panoid)
        else:  # if novel gold path visits panoid twice then select the correct one based on the current trajectory
            num_occurrences_nav = nav.pano_path.count(curr_panoid)
            nth_occurrence = min(num_occurrences, num_occurrences_nav)-1
            pano_index = [i for i, p in enumerate(gt_path) if p == curr_panoid][nth_occurrence]

        if pano_index == len(gt_path)-1:
            assert gt_path[pano_index] == target_panoid
            return 'stop'

        gt_next_panoid = gt_path[pano_index + 1]
        gt_next_heading = curr_node.get_neighbor_heading(gt_next_panoid)
    else:
        shortest_path = nav.env.graph.get_shortest_path(curr_panoid, target_panoid)
        if len(shortest_path) <= 1:
            return 'stop'
        gt_next_panoid = shortest_path[1]
        gt_next_heading = curr_node.get_neighbor_heading(gt_next_panoid)

    next_panoid, next_heading = nav.get_next_state('forward')
    if gt_next_panoid == next_panoid:
        # at 3-way intersection, "forward" AND "left"/"right" can be correct. Only chose forward as gold action
        # if it doesn't imply a rotation of over 45 degrees.
        if len(curr_node.neighbors) != 3 or abs(get_relative_angle(next_heading, gt_next_heading)) < 45:
            return 'forward'

    next_panoid, next_heading = nav.get_next_state('turn_around')
    if gt_next_heading == next_heading:
        return 'turn_around'

    next_panoid, next_heading_left = nav.get_next_state('left')
    if gt_next_heading == next_heading_left:
        return 'left'

    next_panoid, next_heading_right = nav.get_next_state('right')
    if gt_next_heading == next_heading_right:
        return 'right'

    # if multiple rotations are needed, choose direction which brings the agent closer to the correct next heading
    next_heading = get_closest_heading(gt_next_heading, [next_heading_left, next_heading_right])
    if next_heading == next_heading_left:
        return 'left'
    if next_heading == next_heading_right:
        return 'right'

    raise ValueError('gt_action_found not found')


if __name__ == '__main__':
    import os
    import json
    graph_dir = "../datasets/map2seq_seen/graph"

    import torch
    from vln.clip import PanoCLIP
    from vln.prompt_builder import get_navigation_lines
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panoCLIP = PanoCLIP(model_name='clip', device="cpu", cache_dir='../features/features_5shot')

    env = ClipEnv(graph_dir, panoCLIP)
    nav = BaseNavigator(env)

    nav.init_state('_iWkTonHpxW63zgQta8yTg', 27)
    observations = env.get_observations(nav.states, 0, ['brown storefront'])
    #print(observations)
    #exit()

    #p = ['j5TvGvp4ncLqUHiLtgagKA', 'FToLbV-pS6-u12k6ekh2Dg', 'AlU5ovqjCgYgf9BmGFQxUw', 'gzchW-t7bzuOeADmVSrGyw', '4JsURI01132DEYS6bK6LZw', 'fd9_fO1NCKVbcD2bZaE5zQ', '8GMxbWFudC4ZpyWLQQ0uEA', 'HTrP6swGla90fx0WMTbayQ', 'mtqh7TQF0s3yeyhgPKA9cA', '86TdkpRWc5kaBMCCcKZgcg', 'MKFImk35Ifwgn7KVzPIRDw', '6BsKCKLnUJAjuQX5_qGgmg', 'vfAQAtfoQICeR9av5tjV9A', 's1-VzZ0oNROqP6mN3Qh3-g', 'H1rg1AtptxKUvwmb59jL9g', 'bKhOMCzI2dmLfmoxAMyASg', 'KQvCIrC7rYsFocVQqGB9ZA', 'HkvvN9SdShpOELntOeJRCg', 'MVAw1OWDoQV-96R6qcp5qA', 'xgmG0_rRwSlO5DXxYzMJvw', 'lVRC3fU5zBgjVJlrQszcQg', 'Y-ZuQeSLzTEIyq2UpcPjNQ', 'XFrlErWxSaZosBmKFk3Y-w', 'T9DR4U41l3haC5IoK1Llow', 'DUTo4R-EZzcGtbcG8FB6qQ', 'mx4jRmehEcxtAYQ2DG7FSA', 'sDJAWBgjIGb_KGjcDrQxEQ', 'DlKTom37Qv922fsYdagH9A', 'zNn209X7rojwH3rDLbyinQ', 'FOxraV3D2aNRhdn1JnLecw', 'Q9v8lC8r8YqQH3fxutu9TA', 'VSiQGEMAXo1qCG0Mb8d_nA', 'aCNntOtaF6pWi8uN8mEQhg', 'R7x4BHPN9zNMZrEGhGIzxQ', 'a1EETrFYRGL62-kLjhbehA', '0Ig3hirmcj_3e_gEMg6lJw', 'l5oGRSyZ1frhCQHCZ0Usvw', '-hc_1OE0iQ1LDU4_wcyUCg', 'db70mr73mQ3UsJehW5trqg', 'sGSHG27e7me1XG5H_TO5cA', 'pfq1SXr08RYgU_tSbKLL2Q', 'NU5qANElYuqJ8h01ftMo_w', 'gVBtNG-rcMIZP-ZjFbD2SA', 'oRGpoKQHehHvltNC6HppNg', 'gVBtNG-rcMIZP-ZjFbD2SA', 'NU5qANElYuqJ8h01ftMo_w', 'pfq1SXr08RYgU_tSbKLL2Q', 'sGSHG27e7me1XG5H_TO5cA', 'db70mr73mQ3UsJehW5trqg', '-hc_1OE0iQ1LDU4_wcyUCg']
    #print('\n'.join(get_gold_nav(dict(start_heading=299, route_panoids=p), env).actions))
    #exit()

    for dataset_name in ['touchdown', 'map2seq']:
        data_dir = f"../datasets/{dataset_name}_seen/data"
        for split in ['dev', 'train', 'test']:
            path = os.path.join(data_dir, f'{split}.json')
            with open(path) as f:
                for line in tqdm(f):
                    instance = json.loads(line)
                    start_node = env.graph.nodes[instance['route_panoids'][0]]
                    instance['start_heading'] = get_closest_heading(instance['start_heading'],
                                                                    start_node.neighbors.keys())


                    #print(start_node.panoid)
                    #if instance['route_id'] != 11772:
                    #    continue
                    #print(instance['route_id'])
                    #print(instance['id'])
                    #print(instance['route_panoids'])
                    nav = get_gold_nav(instance, env)
                    navigation_lines = get_navigation_lines(nav, env, ['Starbucks', 'bike rental'], is_map2seq=instance['is_map2seq'])
                    print(instance)
                    print('\n'.join(navigation_lines))
                    print()
                    #print(nav.states)
                    #exit()


# zkvuAtd0LeY1fO_EU4lcMw", "ARRlZyTkG-pDfJj_Iq-CqQ", "zkvuAtd0LeY1fO_EU4lcMw