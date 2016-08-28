import os
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,floatX=float32,device=cpu'

import os
import os.path as osp
import json
import numpy as np

from tools import Logger
from reader import Reader
from beam_search import BeamSearch
from tools import evaluate
from config import DATA_ROOT, SAVE_ROOT
from model.ra import Model
import time


def validate(beam_searcher, dataset, logger=None, res_file=None):
    if logger is None:
        logger = Logger(None)
    # generating captions
    all_candidates = []
    tic = time.clock()
    for i in xrange(dataset.n_image):
        data = dataset.iterate_batch()  # data: id, img, scene...
        sent = beam_searcher.generate(data[1:])
        cap = ' '.join([dataset.vocab[word] for word in sent])
        print '[{}], id={}, \t\t {}'.format(i, data[0], cap)
        all_candidates.append({'image_id': data[0], 'caption': cap})
    toc = time.clock() - tic
    running_time = toc / 5000.0

    if res_file is None:
        res_file = 'tmp.json'
    json.dump(all_candidates, open(res_file, 'w'))
    gt_file = osp.join(dataset.data_dir, 'captions_'+dataset.data_split+'.json')
    scores = evaluate(gt_file, res_file, logger)
    if res_file == 'tmp.json':
        os.system('rm -rf %s' % res_file)

    return scores, running_time

if __name__ == '__main__':
    # compile model
    data_dir = str(np.load('data_dir.npy'))
    save_dir = str(np.load('save_dir.npy'))

    # save_dir, data_dir,
    models = [Model(model_file=osp.join(save_dir, 'ra.h5.merge'))]
    valid_set = Reader(batch_size=1, data_split='test', vocab_freq='freq5', stage='test',
                       data_dir=data_dir, feature_file='features_30res.h5',
                       caption_switch='off', topic_switch='off', head=0, tail=1)
    bs = BeamSearch(models, beam_size=3, num_cadidates=500, max_length=20)
    scores, running_time = validate(bs, valid_set) # compile

    # again
    valid_set = Reader(batch_size=1, data_split='test', vocab_freq='freq5', stage='test',
                       data_dir=data_dir, feature_file='features_30res.h5',
                       caption_switch='off', topic_switch='off') # 5000
    scores, running_time = validate(bs, valid_set)

    # scores: B1-4, M, R, C
    np.save('scores', scores)
    np.save('running_time', running_time)
