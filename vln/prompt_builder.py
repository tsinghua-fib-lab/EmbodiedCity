action_space = 'Action Space:\nforward (go straight), left (rotate left), right (rotate right), stop (end navigation)\n\n'
prompt_template = 'Navigation Instructions:\n"{}"\nAction Sequence:\n'


def build_prompt(instructions):
    prompt = action_space
    #prompt += prompt_template.format(instructions, action_space, "1. forward\n2.")
    prompt += prompt_template.format(instructions, action_space)
    return prompt


def get_navigation_lines(nav, env, landmarks, traffic_flow, step_id=0):
    actions = nav.actions
    states = nav.states

    assert len(actions) == len(states)

    lines = list()
    is_action = list()
    while step_id < len(actions):
        action = actions[step_id]

        # print step number and action
        line = f'{step_id}. {action}'
        if action != 'init':
            lines.append(line)
            is_action.append(True)

        # print current env observations
        observations = env.get_observations(states, step_id, landmarks, traffic_flow)
        observations_str = get_observations_str(observations)

        if observations_str:
            line = observations_str
            lines.append(line)
            is_action.append(False)

        step_id += 1

    # print number of input step if sequence not finished
    if actions[-1] != 'stop':
        line = f'{len(actions)}.'
        lines.append(line)
        is_action.append(False)

    assert len(lines) == len(is_action)
    return lines, is_action


def get_observations_str(observations):
    observations_strs = list()

    if 'traffic_flow' in observations:
        traffic_flow = observations["traffic_flow"]
        observations_strs.append(f'You are aligned {traffic_flow} the flow of the traffic.')

    if 'intersection' in observations:
        num_ways = observations['intersection']
        observations_strs.append(f'There is a {num_ways}-way intersection.')

    if 'landmarks' in observations:
        directions = ['on your left', 'slightly left', 'ahead', 'slightly right', 'on your right']
        for direction, landmarks in zip(directions, observations['landmarks']):
            if len(landmarks) > 0:
                landmarks = ' and '.join(landmarks)
                landmarks = landmarks[0].upper() + landmarks[1:]
                observations_strs.append(f'{landmarks} {direction}.')

    return ' '.join(observations_strs)

