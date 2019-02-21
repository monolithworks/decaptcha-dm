import cv2
import logging
from dm.segm import Segmenter
from dm.cnn import loaded_model, charset, prepared

log = logging.getLogger(__name__)

if __name__ == '__main__':
    import sys
    import json
    import os
    import random
    import getopt
    import numpy as np

    model_fn = None
    output_model_fn = None

    opts,captchas = getopt.gnu_getopt(sys.argv[1:], 'm:', ['model='])
    for o,a in opts:
        if o in ['-m', '--model']:
            model_fn = a

    valid_dataset = []

    logging.basicConfig(level=logging.INFO, format='%(msg)s')

    model = loaded_model(model_fn)

    for captcha in captchas:
        image = cv2.imread(captcha, cv2.IMREAD_GRAYSCALE)
        digits = Segmenter().segment(image)
        if len(digits) == 4:
            X = prepared(digits)
            y_pred = model.predict(X)
            print('{}: {}'.format(captcha, ''.join(charset[np.argmax(y_pred[s])] for s in range(4))))
