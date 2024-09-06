class BaseNavigator:
    action_list = ["forward", "left", "right", "stop", "turn_around"]
    action_mapping = {'turnaround': 'turn_around', 'turn around': 'turn_around', 'turn_left': 'left',
                      'turn left': 'left', 'turn_right': 'right', 'turn right': 'right'}

    def __init__(self, env):
        self.env = env

        self.states = list()
        self.actions = list()
        self.pano_path = list()

    def init_state(self, panoid, heading):
        self._set_state(panoid, heading)
        self.actions.append('init')

    def _set_state(self, panoid, heading):
        heading = round(heading)
        state = (panoid, heading)

        prev_pano, _ = self.get_state()
        if panoid != prev_pano:
            self.pano_path.append(panoid)

        self.states.append(state)

    def get_state(self):
        if len(self.states) == 0:
            return None, None
        return self.states[-1]

    def get_prev_state(self):
        if len(self.states) < 2:
            return None, None
        return self.states[-2]

    def validate_action(self, action):
        curr_panoid, cur_heading = self.get_state()
        num_neighbors = self.env.graph.get_num_neighbors(curr_panoid)

        action = self.action_mapping.get(action, action)

        if num_neighbors <= 1:
            if action not in ['stop', 'turn_around']:
                return 'stop'
        if num_neighbors == 2:  # can only stop, turn_around or forward on regular street segment
            if action in ['left', 'right']:
                return 'forward'

        if action not in self.action_list:
            print('action that caused error:', action)
            #raise ValueError()
            action = 'forward'
        return action

    def step(self, action):
        if action == 'init':
            return
        assert action in self.action_list

        next_panoid, next_heading = self.get_next_state(action)
        self.actions.append(action)
        self._set_state(next_panoid, next_heading)

    def get_next_state(self, action):
        curr_pano, curr_heading = self.get_state()
        prev_pano, prev_heading = self.get_prev_state()

        neighbors = self.env.graph.nodes[curr_pano].neighbors
        num_neighbors = len(neighbors)
        out_headings = list(neighbors.keys())

        if action == "stop":
            return curr_pano, curr_heading

        if action == 'forward' and curr_heading in neighbors:
            return neighbors[curr_heading].panoid, curr_heading

        if action == 'turn_around':
            out_heading = (curr_heading - 180) % 360
            next_heading = get_closest_heading(out_heading, out_headings)
            return curr_pano, next_heading

        if num_neighbors <= 1 and prev_pano not in [None, curr_pano]:
            # don't move if end of graph reached; but only if agent is not at starting point.
            return curr_pano, curr_heading

        # regular street segment
        if num_neighbors == 2:
            next_heading = get_closest_heading(curr_heading, out_headings)

        # handle intersection
        if num_neighbors > 2:
            next_heading = self._get_next_heading_intersection(action)

        if action == 'forward':
            next_pano = neighbors[next_heading].panoid
        else:  # "left" or "right" rotates the agent but does not move panos
            next_pano = curr_pano

        return next_pano, next_heading

    def _get_next_heading_intersection(self, action):
        curr_pano, curr_heading = self.get_state()
        prev_pano, prev_heading = self.get_prev_state()
        curr_node = self.env.graph.nodes[curr_pano]
        neighbors = curr_node.neighbors

        # heading of all outgoing edges
        out_headings = list(neighbors.keys())

        forward_heading = curr_heading
        if curr_pano != prev_pano and prev_pano is not None:
            out_headings.remove(curr_node.get_neighbor_heading(prev_pano))

            # select forward_heading relative to other outgoing edges
            out_headings_sorted = list(sorted(out_headings, key=lambda h: get_relative_angle(curr_heading, h)))

            # forward_heading is the middle direction of all outgoing edges
            n = len(out_headings_sorted)
            if len(out_headings_sorted) % 2 == 0:
                forward_heading_1 = out_headings_sorted[n // 2 - 1]
                forward_heading_2 = out_headings_sorted[n // 2]
                forward_heading = get_closest_heading(curr_heading, [forward_heading_1, forward_heading_2])
            else:
                forward_heading = out_headings_sorted[n // 2]

            if len(neighbors) == 3:
                if action == 'left':
                    return out_headings_sorted[0]
                if action == 'right':
                    return out_headings_sorted[-1]

        if action == 'forward':
            return forward_heading

        candidate_headings = set(out_headings) - {forward_heading}
        if action == 'left':
            return min(candidate_headings, key=lambda h: (forward_heading - h) % 360)
        if action == 'right':
            return min(candidate_headings, key=lambda h: (h - forward_heading) % 360)


def get_closest_heading(heading, headings):
    closest = min(headings, key=lambda h: 180 - abs(abs(heading - h) - 180))
    return closest


def get_relative_angle(curr_heading, heading):
    angle = heading - curr_heading
    if angle > 180:
        angle = angle - 360
    if angle <= -180:
        angle = angle + 360
    return angle

