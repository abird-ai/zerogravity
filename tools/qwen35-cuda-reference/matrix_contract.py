"""Fresh-prompt successor of the original matrix data contract; same validators."""
import hashlib
import json
from pathlib import Path
from reference_request import read_json, rendered, request

_REQUEST = request(read_json(Path(__file__).parent / 'request.json'), tokenized=True)
MODEL = '0.8b'
PROMPT_ROWS = _REQUEST['prompt_tokens']
HIDDEN, INTERMEDIATE, ALL_TENSORS = 1024, 3584, 488
VOCAB, MAX_OUTPUT, EOS = 248320, 8, 248046
LAYER_LIMIT, READ_LIMIT = 16 * 1024**2, 1024**2
PROMPT = _REQUEST['prompt']
PROMPT_IDS = _REQUEST['prompt_token_ids']
RENDERED = rendered(_REQUEST)
CASE = f'qwen35-cuda-prefill-tokens{PROMPT_ROWS}-batch1-max8-eos248046'
GENERATION = {'eos_token_id': EOS, 'bos_token_id': None, 'pad_token_id': 248044,
              'temperature': 0.0, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0,
              'repetition_penalty': 1.0, 'presence_penalty': 0.0, 'frequency_penalty': 0.0}

def require(ok, why):
    if not ok:
        raise ValueError(why)


def count(value):
    require(type(value) is int and 1 <= value <= MAX_OUTPUT, 'one-to-eight natural samples')
    return value


def rows_for(outputs):
    return (PROMPT_ROWS,) + (1,) * (count(outputs)-1)


def sequences_for(outputs):
    return tuple(range(PROMPT_ROWS, PROMPT_ROWS+count(outputs)))


def all42_bytes(outputs):
    rows, ends = rows_for(outputs), sequences_for(outputs)
    return 18*len(rows)*(36864+1048576) + 6*sum(ends)*2048 + 24*sum(rows)*HIDDEN*4


def prior_bytes(outputs):
    return 18*(count(outputs)-1)*(36864+1048576) + 6*sum(sequences_for(outputs)[:-1])*2048


def expected_prior_records(layer, outputs):
    require(type(layer) is int and layer in range(24), 'actual original layer')
    records = []
    for call in range(1, count(outputs)):
        if layer % 4 != 3:
            records += [(call, 'state.conv.before', 'torch.bfloat16', [3,6144], 36864),
                        (call, 'state.recurrent.before', 'torch.float32', [16,128,128], 1048576)]
        else:
            records += [(call, f'kv.before.position.{p}', 'torch.bfloat16', [2,512], 2048)
                        for p in range(PROMPT_ROWS+call-1)]
    return records


def validate_completion(item):
    require(item['case'] == CASE and item['prompt_ids'] == PROMPT_IDS, 'exact genuine tokenized prompt')
    ids = item['completion_ids']; count(len(ids))
    require(type(ids) is list and all(type(x) is int and 0 <= x < VOCAB for x in ids)
            and EOS not in ids[:-1], 'original natural IDs stop at first EOS')
    if ids[-1] == EOS:
        require(item['finish_reason'] == 'stop' and item['stop_reason'] in (None, EOS), 'actual terminal EOS')
    else:
        require(len(ids) == 8 and item['finish_reason'] == 'length' and item['stop_reason'] is None,
                'eight reached naturally; no forced teachers')
    return len(ids)


def validate_effective_policy(document):
    require(document['complete'] is True and document['failures'] == []
            and len(document['records']) == 1, 'one original InputProcessor return')
    row = document['records'][0]
    require(row['source_observer'] == 'actual-pinned-InputProcessor.process_inputs-return'
            and row['unchanged_return'] is True and row['prompt_token_ids'] == PROMPT_IDS
            and row['tokenizer_eos_token_id'] == EOS
            and row['generation_config_fields'].get('eos_token_id') == EOS
            and row['model_eos_metadata'] == 248044 and row['max_tokens'] == 8
            and row['enforce_eager'] is True, 'actual common generation/source EOS distinction')
    wanted = {'temperature':0.0,'top_p':1.0,'n':1,'repetition_penalty':1.0,
              'presence_penalty':0.0,'frequency_penalty':0.0,'ignore_eos':False,
              'eos_token_id':EOS,'all_stop_token_ids':[EOS],'stop_token_ids':[],
              'stop':[],'allowed_token_ids':None,'min_tokens':0}
    require(row['sampling'] == wanted, 'actual unprocessed greedy EOS-only policy')
    return row

