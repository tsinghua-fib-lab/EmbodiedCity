def get_metrics(instances, graph):
    tc = 0
    spd = 0
    kpa = 0
    total = 0.0

    all_metrics = list()
    for instance in instances:
        metrics = dict(tc=0.0, spd=0.0, kpa=0.0)

        agent_path = instance['agent_pano_path']
        gold_path = instance['gold_pano_path']
        target_panoid = gold_path[-1]
        total += 1

        def _get_key_points(pano_path):
            if len(pano_path) <= 1:
                return []
            key_points = list((pano_path[0], pano_path[1]))
            for i in range(len(pano_path)):
                pano = pano_path[i]
                if graph.get_num_neighbors(pano) > 2:
                    if i+1 < len(pano_path):
                        next_pano = pano_path[i+1]
                        key_points.append((pano, next_pano))
            return key_points

        gold_key_points = _get_key_points(gold_path)
        agent_key_points = _get_key_points(agent_path)

        kp_correct = len(set(gold_key_points) & set(agent_key_points))

        if agent_path[-1] in graph.get_target_neighbors(gold_path[-1]) + [gold_path[-1]]:
            kp_correct += 1

        _kpa = kp_correct / (len(gold_key_points) + 1)
        metrics['kpa'] = _kpa
        kpa += _kpa

        target_list = graph.get_target_neighbors(target_panoid) + [target_panoid]
        if agent_path[-1] in target_list:
            tc += 1
            metrics['tc'] = 1
        _spd = graph.get_shortest_path_length(agent_path[-1], target_panoid)
        spd += _spd
        metrics['spd'] = _spd

        all_metrics.append(metrics)

    correct = tc
    tc = tc / total * 100
    spd = spd / total
    kpa = kpa / total * 100
    return correct, tc, spd, kpa, all_metrics


def get_metrics_from_results(results, graph):

    instances = list(results['instances'].values())
    correct, tc, spd, kpa, all_metrics = get_metrics(instances, graph)

    assert len(instances) == len(all_metrics)
    for instance, metrics in zip(instances, all_metrics):
        instance['metrics'] = metrics

    results['metrics'] = dict(correct=correct, tc=round(tc, 2), spd=round(spd, 2), sed=round(kpa, 2))

    return correct, tc, spd, kpa, results
