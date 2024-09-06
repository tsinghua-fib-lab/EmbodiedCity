def filter_by_words(elements):
    filtered_elements = list()
    for element in elements:
        if 'last building' in element or 'first building' in element or 'second building' in element:
            continue

        if 'intersection' in element.lower():
            continue
        if 'intersecting' in element.lower():
            continue
        if 'destination' in element.lower():
            continue
        if element.lower() == 'street':
            continue
        if 'side street' in element.lower():
            continue
        if 'left turn' in element.lower():
            continue
        if 'right turn' in element.lower():
            continue
        filtered_elements.append(element)

    return filtered_elements


def filter_landmarks(elements, instructions):
    landmarks = list()

    elements = filter_by_words(elements)

    for element in elements:

        no_article_element = element
        if element.lower().startswith('the '):
            no_article_element = element[4:]

        if no_article_element.lower() not in instructions.lower():
            continue
        if not element:
            continue

        con = False
        for banned in ['Corner', 'Light', 'Block']:
            # skip banned, except it is written in uppercase in list and instructions
            if banned.lower() in element.lower() and not (banned in element and banned in instructions):
                con = True
        if con:
            continue

        if element in landmarks:
            continue
        landmarks.append(element)

    return landmarks


def filter_landmarks_5shot(unfiltered):
    landmarks = list()

    for element in unfiltered:
        element = element.strip()

        if 'intersection' in element or 'side street' in element:
            continue
        if element == 'traffic lights' or element == 'double lights':
            continue
        if element == 'a building' or element == 'a large building':
            continue
        if not element:
            continue

        if element in landmarks:
            continue
        landmarks.append(element)

    return landmarks
