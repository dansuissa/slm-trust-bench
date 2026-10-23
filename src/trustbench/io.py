"""Strict records and provenance utilities shared by preparation and evaluation."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank record at line {number}")
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at line {number}") from error
    return records


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def validate_items(items):
    seen = set()
    for item in items:
        required = {"item_id", "group_id", "dataset", "task", "split", "messages", "references", "should_abstain"}
        if required - item.keys():
            raise ValueError(f"Missing item fields: {required - item.keys()}")
        if not isinstance(item["item_id"], str) or item["item_id"] in seen:
            raise ValueError("Item IDs must be distinct strings")
        seen.add(item["item_id"])
        if not isinstance(item["group_id"], str) or not item["group_id"]:
            raise ValueError("Missing independent grouping unit")
        if item["task"] not in {"factual", "abstention", "value"}:
            raise ValueError("Unknown task")
        if not isinstance(item["references"], list) or any(not isinstance(x, str) for x in item["references"]):
            raise ValueError("References must be a list of strings")
        if item["should_abstain"] is not None and type(item["should_abstain"]) is not bool:
            raise ValueError("Answerability label must be boolean or null")
        if not item["messages"] or any(m.get("role") not in {"system", "user", "assistant"}
                                        or not isinstance(m.get("content"), str) for m in item["messages"]):
            raise ValueError("Invalid chat messages")
    return items


def assert_disjoint(*partitions):
    seen = set()
    for partition in partitions:
        groups = {item["group_id"] for item in partition}
        if seen & groups:
            raise ValueError(f"Group leakage across splits: {len(seen & groups)} groups")
        seen |= groups


def completed_run_records(path):
    path = Path(path)
    metadata = json.loads((path.parent / 'run.json').read_text(encoding='utf-8'))
    if metadata['status'] not in {'generation_complete', 'generation_complete_with_errors'}:
        raise ValueError('Inference run is incomplete')
    expected = {(identifier, repetition) for identifier in metadata['selected_items']
                for repetition in range(metadata['config']['repetitions'])}
    records = read_jsonl(path)
    identities = [(r['item_id'], r['repetition']) for r in records]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError('Missing, duplicate, or unexpected responses')
    return records
