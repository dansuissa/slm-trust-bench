"""Serial, checkpoint-pinned inference. GPU execution is verified separately from local checks."""
import ast
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

from .io import completed_run_records, digest, read_jsonl, validate_items, write_json
from .metrics import entropy_score, exact_match, parse_value_choice


def draw_seed(seed, item_id, repetition, phase):
    value = f'{seed}:{item_id}:{repetition}:{phase}'
    return int(sha256(value.encode()).hexdigest()[:8], 16)


def extract_factual(text, rule):
    if rule == 'released_period_slice':
        return text[:text.find('.')]
    if rule == 'preserve_missing_period':
        return text.split('.', 1)[0].strip()
    raise ValueError(f'Unknown answer extraction: {rule}')


class HFGenerator:
    def __init__(self, config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        model = config['model']
        if len(model['revision']) != 40:
            raise ValueError('A full checkpoint revision is required')
        if config['device'] == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; select a CPU pilot explicitly or use the allocated GPU')
        self.device = config['device']
        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[config['dtype']]
        self.tokenizer = AutoTokenizer.from_pretrained(model['id'], revision=model['revision'], trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(model['id'], revision=model['revision'],
                                                        torch_dtype=dtype, trust_remote_code=False).to(self.device)
        self.model.eval()
        self.parameters = sum(p.numel() for p in self.model.parameters())
        if self.parameters >= 10_000_000_000:
            raise ValueError('Checkpoint exceeds the total-parameter inclusion limit')
        self.generation_defaults = self.model.generation_config.to_dict()
        self.config = config

    def generate(self, messages, settings, seed, plain=False):
        import torch
        from transformers import set_seed
        set_seed(seed)
        if plain:
            text = messages[-1]['content']
        else:
            options = {'tokenize': False, 'add_generation_prompt': self.config.get('add_generation_prompt', True)}
            if 'enable_thinking' in self.config:
                options['enable_thinking'] = self.config['enable_thinking']
            text = self.tokenizer.apply_chat_template(messages, **options)
        inputs = self.tokenizer(text, return_tensors='pt', add_special_tokens=self.config.get('add_special_tokens', False)).to(self.device)
        input_length = inputs['input_ids'].shape[1]
        if input_length + settings['max_new_tokens'] > self.config['max_context_tokens']:
            raise ValueError('Context limit exceeded; silent truncation is disabled')
        kwargs = {key: value for key, value in settings.items() if value is not None}
        kwargs['return_dict_in_generate'] = True
        kwargs['output_scores'] = False
        if kwargs.get('do_sample') is False:
            kwargs.pop('temperature', None)
            kwargs.pop('top_p', None)
        if self.tokenizer.pad_token_id is None:
            kwargs['pad_token_id'] = self.tokenizer.eos_token_id
        if self.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            result = self.model.generate(**inputs, **kwargs)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        new_tokens = result.sequences[0][input_length:]
        return {'text': self.tokenizer.decode(new_tokens, skip_special_tokens=True), 'input_tokens': input_length,
                'output_tokens': len(new_tokens), 'seconds': elapsed, 'seed': seed,
                'peak_cuda_bytes': torch.cuda.max_memory_allocated() if self.device == 'cuda' else None,
                'rendered_prompt_sha256': sha256(text.encode()).hexdigest()}


def run(root, config_path, output, resume=False):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['task'] not in {'factual', 'abstention', 'value'}:
        raise ValueError('Unknown task')
    if config['repetitions'] < 1 or (config['max_items'] is not None and config['max_items'] < 1):
        raise ValueError('Repetitions and item limits must be positive')
    input_path = root / config['data']
    items = validate_items(read_jsonl(input_path))
    if any(item['task'] != config['task'] for item in items):
        raise ValueError('Dataset and experiment task differ')
    if config.get('research_split'):
        items = [x for x in items if x.get('research_split') == config['research_split']]
    items.sort(key=lambda x: key_for_selection(x['item_id'], config['seed']))
    items = items[:config['max_items']] if config['max_items'] is not None else items
    if not items:
        raise ValueError('No selected items')
    output = Path(output)
    fingerprint = {'config_sha256': digest(config_path), 'data_sha256': digest(input_path),
                   'code_sha256': sha256(''.join(digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))).encode()).hexdigest()}
    result_path = output / 'responses.jsonl'
    done = set()
    if output.exists():
        if not resume:
            raise ValueError('Run directory exists; use a new directory or --resume')
        previous = json.loads((output / 'run.json').read_text(encoding='utf-8'))
        if previous['fingerprint'] != fingerprint:
            raise ValueError('Resume rejected: code, configuration or data changed')
        if result_path.exists():
            for record in read_jsonl(result_path):
                identity = (record['item_id'], record['repetition'])
                if identity in done:
                    raise ValueError('Duplicate saved response')
                done.add(identity)
    else:
        output.mkdir(parents=True)
    starting_metadata = {'fingerprint': fingerprint, 'config': config,
                         'selected_items': [x['item_id'] for x in items],
                         'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'loading_model'}
    if resume:
        starting_metadata['started_at_utc'] = previous['started_at_utc']
        starting_metadata['resumed_at_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(output / 'run.json', starting_metadata)
    generator = HFGenerator(config)
    versions = {package: importlib.metadata.version(package) for package in ['torch', 'transformers', 'accelerate']}
    write_json(output / 'run.json', {'fingerprint': fingerprint, 'config': config, 'versions': versions,
        'python': platform.python_version(), 'platform': platform.platform(), 'parameter_count_loaded': generator.parameters,
        'generation_defaults': generator.generation_defaults, 'selected_items': [x['item_id'] for x in items],
        'started_at_utc': starting_metadata['started_at_utc'], 'status': 'running', 'resume': resume})
    with result_path.open('a', encoding='utf-8', newline='\n') as stream:
        for item in items:
            for repetition in range(config['repetitions']):
                if (item['item_id'], repetition) in done:
                    continue
                record = dict(item_id=item['item_id'], group_id=item['group_id'], dataset=item['dataset'],
                              repetition=repetition, should_abstain=item['should_abstain'], variant='original', status='ok')
                try:
                    plain = config['task'] == 'factual'
                    messages = item['messages']
                    if plain:
                        messages = [{'role': 'user', 'content': config['candidate_prompt'].format(question=messages[-1]['content'])}]
                    candidate = generator.generate(messages, config['candidate'],
                        draw_seed(config['seed'], item['item_id'], repetition, 'candidate'), plain)
                    record['generation'] = candidate
                    record['response'] = candidate['text']
                    if config['task'] == 'factual':
                        answer = extract_factual(candidate['text'], config['answer_extraction'])
                        record['answer'] = answer
                        record['correct'] = (any(ref in answer for ref in item['references'])
                                             if config['answer_rule'] == 'substring' else exact_match(answer, item['references']))
                        samples = []
                        score_messages = [{'role': 'user', 'content': config['score_prompt'].format(question=item['messages'][-1]['content'])}]
                        for number in range(config['score_samples']):
                            sample = generator.generate(score_messages, config['score_decoding'],
                                draw_seed(config['seed'], item['item_id'], repetition, f'sample-{number}'), True)
                            samples.append(sample)
                        record['samples'] = samples
                        record['score'] = entropy_score([extract_factual(s['text'], config['answer_extraction']) for s in samples])
                    elif config['task'] == 'value':
                        record['choice'] = parse_value_choice(candidate['text'], item['option_map'])
                        record['parser_valid'] = record['choice'] is not None
                except Exception as error:
                    record['status'] = 'error'
                    record['error_type'] = type(error).__name__
                    record['error'] = str(error)
                    # An allocation failure can poison subsequent work; stop after recording it.
                    if isinstance(error, generator.torch.cuda.OutOfMemoryError):
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                        stream.flush()
                        raise
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
    metadata = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    saved = read_jsonl(result_path)
    errors = sum(r['status'] != 'ok' for r in saved)
    metadata.update(status='generation_complete_with_errors' if errors else 'generation_complete',
                    failed_responses=errors, finished_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / 'run.json', metadata)
    return {'responses': len(read_jsonl(result_path)), 'path': str(result_path)}


def key_for_selection(item_id, seed):
    return sha256(f'{seed}:{item_id}'.encode()).hexdigest()


def judge(root, config_path, run_directory):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    source = root / 'data/sources/facebookresearch__AbstentionBench/files/recipe/evaluation_judge_prompts.py'
    templates = {}
    for node in ast.parse(source.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            templates[node.targets[0].id] = ast.literal_eval(node.value)
    items = {x['item_id']: x for x in read_jsonl(root / config['data'])}
    run_directory = Path(run_directory)
    responses = completed_run_records(run_directory / 'responses.jsonl')
    if (run_directory / 'judged.jsonl').exists():
        raise ValueError('Judge results exist; use a separate run for a changed judge')
    generation = HFGenerator(config)
    from .io import write_jsonl
    judged = []
    for record in responses:
        record = dict(record)
        if record['status'] != 'ok':
            judged.append(record)
            continue
        item = items[record['item_id']]
        question = item['messages'][-1]['content']
        abstention_prompt = templates['LLM_JUDGE_ABSTENTION_COCONOT_STYLE_PROMPT'].format(
            question=question, ref_answer='[' + '; '.join(item['references']) + ']',
            abstention_label=item['should_abstain'], model_answer=record['response'])
        try:
            result = generation.generate([{'role': 'user', 'content': abstention_prompt}], config['candidate'],
                draw_seed(config['seed'], item['item_id'], record['repetition'], 'judge-abstention'))
            label = result['text'].lower().strip(' .,\n')
            record['abstained'] = {'yes': True, 'no': False}.get(label)
            record['abstention_judge'] = result
            record['correct'] = None
            if item['references']:
                prompt = templates['LLM_JUDGE_CORRECTNESS_PROMPT'].format(
                    question=question, ref_answer='\n'.join(item['references']), model_answer=record['response'])
                result = generation.generate([{'role': 'user', 'content': prompt}], config['candidate'],
                    draw_seed(config['seed'], item['item_id'], record['repetition'], 'judge-correctness'))
                record['correct'] = {'correct': True, 'incorrect': False}.get(result['text'].lower().strip(' .,\n'))
                record['correctness_judge'] = result
        except Exception as error:
            record.update(status='judge_error', error_type=type(error).__name__, error=str(error))
        judged.append(record)
    write_jsonl(run_directory / 'judged.jsonl', judged)
    write_json(run_directory / 'judge.json', {'config': config, 'config_sha256': digest(config_path),
        'prompt_source_sha256': digest(source), 'input_responses_sha256': digest(run_directory / 'responses.jsonl'),
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers']},
        'parameter_count_loaded': generation.parameters, 'generation_defaults': generation.generation_defaults})
    return {'judged': len(judged), 'invalid_abstention_labels': sum(x.get('abstained') is None for x in judged)}
