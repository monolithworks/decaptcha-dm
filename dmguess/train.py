import cv2
import logging
from dm.segm import Segmenter
from dm.cnn import trained_model_on_data, prepared, charset
import numpy as np

log = logging.getLogger(__name__)

class CaptchaReader:
    def __init__(self, map_):
        self._map = map_

    def read(self, nr):
        answer = self._map[str(nr)]['solution']['text']
        if len(answer) == 4:
            return answer
        else:
            log.warning('{}: malformed solution ({}), ignoring'.format(nr, answer))

if __name__ == '__main__':
    import sys
    import json
    import os
    import random
    import getopt

    model_fn = None
    output_model_fn = None
    answer_map = sys.argv[1]

    opts,corpses = getopt.gnu_getopt(sys.argv[1:], 'm:o:a:', ['model=', 'output=', 'answers='])
    for o,a in opts:
        if o in ['-m', '--model']:
            model_fn = a
        if o in ['-o', '--output']:
            output_model_fn = a
        if o in ['-a', '--answers']:
            answer_map = a

    valid_dataset = []

    logging.basicConfig(level=logging.INFO, format='%(msg)s')

    with open(answer_map, 'r') as f:
        reader = CaptchaReader(json.load(f))
        for nr in range(1000):
            answer = reader.read(nr)
            if answer is not None:
                image = cv2.imread(os.path.join(corpses[0], 'dm-{}.jpg'.format(nr)), cv2.IMREAD_GRAYSCALE)
                digits = Segmenter().segment(image)
                if len(digits) == len(answer):
                    for img,digit in zip(digits, answer):
                        valid_dataset.append(dict(grayscale=img, digit=digit))

    log.info('valid datasets: {} samples'.format(len(valid_dataset)))
    cnn_model = trained_model_on_data(valid_dataset, load_model_from=model_fn)
    if output_model_fn:
        cnn_model.save(output_model_fn)
