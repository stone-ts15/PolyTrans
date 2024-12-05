
import json
from config import *

total_opt = None

_questions = []

def all_questions():
    global _questions
    if _questions == []:
        with open(DATASET_PATH, 'r') as f:
            ecql = json.load(f)
            _questions = [q['question'] for q in ecql]
    return _questions


_gts = None
def gts():
    global _gts
    if not _gts:
        with open(DATASET_PATH, 'r') as f:
            _gts = json.load(f)
    return _gts


_schema = None
def imdb_schema():
    global _schema
    if not _schema:
        with open(SCHEMA_PATH, 'r') as f:
            _schema = json.load(f)
    return _schema

_datasets = {
    'IMDB':                  ['imdb', None],
    'IMDBDense':             ['imdb-dense', None]
}

def dataset():
    fn, data = _datasets[total_opt.dataset]
    if not data:
        with open(f'{DATASET_DIR}/{fn}-dataset.json', 'r') as f:
            data = json.load(f)
        _datasets[total_opt.dataset][1] = data
    return data
    

def gt_of(question_or_id):
    if isinstance(question_or_id, str):
        question = question_or_id
        for sample in dataset()['samples']:
            if sample['question'] == question:
                return [t for t in dataset()['template'] if t['template'] == sample['template']][0]
        print(f'Could not find gt for "{question}".')
    elif isinstance(question_or_id, int):
        qid = question_or_id
        for sample in dataset()['samples']:
            if sample['id'] == qid:
                return [t for t in dataset()['template'] if t['template'] == sample['template']][0]

    return None


def sample_of(question):
    for sample in dataset()['samples']:
        if sample['question'] == question:
            return sample
    print(f'Could not find sample for "{question}".')
    return None

MAX_CMP_RESULT_COUNT = 256

QL_TYPENAMES = {
    'AQL': 'aql',
    'ECQL': 'ecql'
}
