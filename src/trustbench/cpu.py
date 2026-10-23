"""Checkpointed serial CPU experiments; each completed draw is persisted before proceeding."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import threading
import time
from uuid import uuid4

from .io import digest, read_jsonl, validate_items
from .metrics import entropy_score, exact_match, parse_value_choice
from .runner import draw_seed

ROOT = Path.cwd().resolve()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def journal_records(path, recover=False):
    if not path.exists():
        return []
    content = path.read_bytes()
    records, kept = [], bytearray()
    lines = content.splitlines(keepends=True)
    for number, line in enumerate(lines):
        try:
            records.append(json.loads(line))
            kept.extend(line if line.endswith(b'\n') else line + b'\n')
        except (json.JSONDecodeError, UnicodeError):
            if not recover or number != len(lines) - 1:
                raise ValueError('Damaged journal record; only a trailing partial record can be recovered')
            break
    if bytes(kept) != content:
        if not recover:
            raise ValueError('Journal has an incomplete trailing record')
        backup = path.with_name(path.name + '.before_recovery.' + uuid4().hex)
        backup.write_bytes(content)
        temporary = path.with_name(path.name + '.recovered')
        temporary.write_bytes(kept)
        temporary.replace(path)
        atomic_json(path.parent / 'journal_recovery.json', {'original_preserved': backup.name,
                    'original_bytes': len(content), 'retained_complete_records': len(records)})
    return records


def environment(config):
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {name: importlib.metadata.version(name) for name in
                         ['torch', 'transformers', 'accelerate', 'tokenizers', 'safetensors', 'numpy', 'psutil']},
            'cpu_threads': config['cpu_threads'], 'dtype': config['dtype']}


class MemoryMonitor:
    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self.stop.wait(.05):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def likelihood_summary(logits, tokens, eos_ids):
    """Raw next-token log-probabilities; EOS excluded from answer-only baselines."""
    import torch
    if len(logits) != len(tokens):
        raise ValueError('Each generated token must have one raw logit vector')
    values = [float(torch.log_softmax(row[0].float(), dim=-1)[int(token)])
              for row, token in zip(logits, tokens)]
    answer_values = [value for value, token in zip(values, tokens) if int(token) not in eos_ids]
    return {'raw_token_log_probabilities': values,
            'answer_log_likelihood_sum': sum(answer_values) if answer_values else None,
            'answer_log_likelihood_mean': sum(answer_values) / len(answer_values) if answer_values else None,
            'answer_likelihood_token_count': len(answer_values), 'likelihood_eos_policy': 'exclude_eos_only'}


class CPUGenerator:
    def __init__(self, config, monitor):
        import psutil
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(config['cpu_threads'])
        torch.set_num_interop_threads(1)
        self.torch = torch
        self.config = config
        self.monitor = monitor
        model = config['model']
        if not re.fullmatch(r'[0-9a-f]{40}', model['revision']):
            raise ValueError('Checkpoint revision must be an immutable commit')
        self.tokenizer = AutoTokenizer.from_pretrained(model['id'], revision=model['revision'], trust_remote_code=False)
        started = time.monotonic()
        self.model = AutoModelForCausalLM.from_pretrained(model['id'], revision=model['revision'],
            torch_dtype={'float32': torch.float32, 'bfloat16': torch.bfloat16}[config['dtype']],
            low_cpu_mem_usage=True, use_safetensors=True, trust_remote_code=False)
        self.model.eval()
        self.load_seconds = time.monotonic() - started
        self.parameters = sum(p.numel() for p in self.model.parameters())
        if self.parameters >= 10_000_000_000:
            raise ValueError('Total parameter limit exceeded')
        self.defaults = self.model.generation_config.to_dict()
        self.check_memory()

    def check_memory(self):
        import psutil
        rss = psutil.Process().memory_info().rss
        if max(rss, self.monitor.peak) > self.config['max_process_rss_gib'] * 2**30:
            raise MemoryError('Process RSS exceeded the configured budget')
        if psutil.virtual_memory().available < self.config['minimum_available_memory_gib'] * 2**30:
            raise MemoryError('Available host memory is below the configured reserve')

    def generate(self, messages, settings, seed, likelihood=False):
        from transformers import set_seed
        self.check_memory()
        set_seed(seed)
        full_started = time.monotonic()
        if self.config['prompt_rendering'] == 'native_chat':
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            special = False
        else:
            rendered = '\n'.join(message['content'] for message in messages)
            special = self.config.get('raw_add_special_tokens', False)
        inputs = self.tokenizer(rendered, return_tensors='pt', add_special_tokens=special)
        count = int(inputs['input_ids'].shape[1])
        if count + settings['max_new_tokens'] > self.config['max_context_tokens']:
            raise ValueError('Context budget exceeded; no truncation was applied')
        generation_settings = dict(settings)
        if not generation_settings['do_sample']:
            generation_settings.pop('temperature', None)
            generation_settings.pop('top_p', None)
            generation_settings.pop('top_k', None)
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        started = time.monotonic()
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_settings, pad_token_id=pad,
                return_dict_in_generate=True, output_logits=likelihood, output_scores=False)
        generation_seconds = time.monotonic() - started
        ids = generated.sequences[0][count:]
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        ended_eos = bool(len(ids) and int(ids[-1]) in eos_ids)
        quality = 'empty' if not text.strip() else ('length_capped' if len(ids) >= settings['max_new_tokens'] and not ended_eos else 'complete')
        result = {'text': text, 'input_tokens': count, 'output_tokens': len(ids), 'seed': seed,
                  'generation_seconds': generation_seconds, 'end_to_end_seconds': time.monotonic() - full_started,
                  'response_quality': quality, 'finish_reason': 'eos' if ended_eos else 'length_or_other_stop',
                  'peak_rss_bytes_observed': self.monitor.peak,
                  'rendered_prompt_sha256': sha256(rendered.encode()).hexdigest()}
        if likelihood:
            result.update(likelihood_summary(generated.logits, ids, eos_ids))
        self.check_memory()
        return result


def messages_for(item, config):
    if item['task'] != 'factual':
        return item['messages']
    return [{'role': 'user', 'content': config['question_instruction'] + '\nQuestion: ' + item['messages'][-1]['content']}]


def run(config_path, output, resume=False):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['cpu_threads'] < 1 or config['prompt_rendering'] not in {'native_chat', 'raw_content'}:
        raise ValueError('Invalid CPU configuration')
    source = ROOT / config['data']
    items = validate_items(read_jsonl(source))
    if not items or any(item['task'] != config['task'] for item in items):
        raise ValueError('Empty or incompatible input data')
    env = environment(config)
    identity = {'config': digest(config_path), 'data': digest(source), 'environment': env,
                'code': {path.name: digest(path) for path in sorted(Path(__file__).parent.glob('*.py'))}}
    if output.exists():
        if not resume:
            raise ValueError('Output exists; use explicit resume for an unchanged run')
        previous = json.loads((output / 'run.json').read_text(encoding='utf-8'))
        if previous['fingerprint'] != identity:
            raise ValueError('Resume rejected: model/configuration/data/code/environment changed')
    else:
        output.mkdir(parents=True)
    metadata = {'fingerprint': identity, 'config': config, 'environment': env,
                'selected_items': [item['item_id'] for item in items], 'status': 'loading_model',
                'started_at_utc': previous['started_at_utc'] if resume else datetime.now(timezone.utc).isoformat()}
    atomic_json(output / 'run.json', metadata)
    journal = output / 'draws.jsonl'
    events = journal_records(journal, recover=resume)
    saved = {}
    for event in events:
        key = (event['item_id'], event['repetition'], event['phase'])
        if key in saved:
            raise ValueError('Duplicate generation checkpoint')
        saved[key] = event['generation']
    phases_expected = ['candidate'] + ([f'sample-{i}' for i in range(config['score_samples'])] if config['task'] == 'factual' else [])
    expected_keys = {(item['item_id'], repetition, phase) for item in items
                     for repetition in range(config['repetitions']) for phase in phases_expected}
    if not set(saved).issubset(expected_keys):
        raise ValueError('Checkpoint contains a draw outside the configured experiment')
    expected_draws = sum(config['repetitions'] * (1 + (config['score_samples'] if config['task'] == 'factual' else 0)) for _ in items)
    outputs = []
    try:
        with MemoryMonitor() as monitor:
            generator = CPUGenerator(config, monitor)
            metadata.update(status='running', parameter_count_loaded=generator.parameters,
                            generation_defaults=generator.defaults, model_load_seconds=generator.load_seconds)
            atomic_json(output / 'run.json', metadata)
            with journal.open('a', encoding='utf-8', newline='\n') as stream:
                for item in items:
                    messages = messages_for(item, config)
                    for repetition in range(config['repetitions']):
                        phases = ['candidate'] + ([f'sample-{i}' for i in range(config['score_samples'])] if config['task'] == 'factual' else [])
                        for phase in phases:
                            key = (item['item_id'], repetition, phase)
                            if key in saved:
                                continue
                            generation = generator.generate(messages, config['candidate'] if phase == 'candidate' else config['score_decoding'],
                                draw_seed(config['seed'], item['item_id'], repetition, phase),
                                likelihood=phase == 'candidate' and config['task'] == 'factual')
                            event = {'item_id': item['item_id'], 'repetition': repetition, 'phase': phase, 'generation': generation}
                            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
                            stream.flush()
                            os.fsync(stream.fileno())
                            saved[key] = generation
                            atomic_json(output / 'progress.json', {'completed_draws': len(saved), 'expected_draws': expected_draws,
                                        'last_item': item['item_id'], 'last_phase': phase, 'peak_rss_bytes_observed': monitor.peak,
                                        'updated_at_utc': datetime.now(timezone.utc).isoformat()})
                        candidate = saved[(item['item_id'], repetition, 'candidate')]
                        record = {'item_id': item['item_id'], 'group_id': item['group_id'], 'dataset': item['dataset'],
                                  'repetition': repetition, 'status': 'ok', 'response': candidate['text'],
                                  'generation': candidate, 'should_abstain': item['should_abstain'],
                                  'candidate_nonempty': bool(candidate['text'].strip())}
                        if config['task'] == 'factual':
                            samples = [saved[(item['item_id'], repetition, phase)] for phase in phases[1:]]
                            answers = [sample['text'].strip() for sample in samples]
                            valid = record['candidate_nonempty'] and all(answers)
                            raw_score = entropy_score(answers)
                            record.update(samples=samples, answer=candidate['text'].strip(),
                                          correct=exact_match(candidate['text'], item['references']),
                                          correct_substring_release_rule=any(ref in candidate['text'] for ref in item['references']),
                                          score_raw_entropy=raw_score, confidence_inputs_nonempty=bool(valid),
                                          score=raw_score if valid else -math.log(len(samples)) - 1.)
                        elif config['task'] == 'value':
                            record['choice'] = parse_value_choice(candidate['text'], item['option_map'])
                            record['parser_valid'] = record['choice'] is not None
                        outputs.append(record)
            response_path = output / 'responses.jsonl'
            temporary = response_path.with_name('responses.writing')
            with temporary.open('w', encoding='utf-8', newline='\n') as stream:
                for record in outputs:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            temporary.replace(response_path)
            quality = Counter(record['generation']['response_quality'] for record in outputs)
            summary = {'responses': len(outputs), 'candidate_quality': dict(quality),
                       'draws': len(saved), 'peak_rss_bytes_observed': monitor.peak,
                       'all_generation_end_to_end_seconds': sum(event['end_to_end_seconds'] for event in saved.values()),
                       'purpose': config['purpose']}
            if config['task'] == 'factual':
                summary.update(exact_correct=sum(record['correct'] for record in outputs),
                               substring_correct=sum(record['correct_substring_release_rule'] for record in outputs),
                               empty_scoring_samples=sum(not sample['text'].strip() for record in outputs for sample in record['samples']))
            elif config['task'] == 'value':
                summary['valid_choices'] = sum(record['parser_valid'] for record in outputs)
            atomic_json(output / 'summary.json', summary)
            metadata.update(status='generation_complete', finished_at_utc=datetime.now(timezone.utc).isoformat(),
                            peak_rss_bytes_observed=monitor.peak)
            atomic_json(output / 'run.json', metadata)
            print(json.dumps(summary, indent=2))
    except BaseException as error:
        metadata.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        error_type=type(error).__name__, error=str(error),
                        last_update_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(output / 'run.json', metadata)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.config, args.out, args.resume)
