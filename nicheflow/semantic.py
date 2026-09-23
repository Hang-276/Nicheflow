"""Frozen CPU MiniLM, masked mean pooling and fixed data-independent projection.

Only public text enters the encoder. No label fitting or network calls at runtime.
The projection is hash-defined so a restart cannot choose a different basis.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from .spec import EnvironmentBlocked, IntegrityError, file_hash, digest


def verify_encoder(spec):
    path = Path(spec['path'])
    manifest = path / 'manifest.json'
    if not manifest.is_file() or file_hash(manifest) != spec['manifest_sha256']:
        raise EnvironmentBlocked('frozen semantic encoder manifest missing/changed')
    value = json.loads(manifest.read_text())
    for name, expected in value['files'].items():
        if Path(name).name != name or not (path / name).is_file() or file_hash(path / name) != expected:
            raise IntegrityError('semantic encoder file missing/changed: ' + name)
    return value


class FrozenSemantic:
    def __init__(self, spec):
        self.spec, self.dimension = dict(spec), spec['dimension']
        self.identity = verify_encoder(spec)
        self.model = self.tokenizer = None
        self.projection = np.array([[1. if hashlib.sha256(f"{spec['projection_seed']}:{i}:{j}".encode()).digest()[0] & 1 else -1.
                                    for j in range(self.dimension)] for i in range(384)]) / np.sqrt(self.dimension)

    @lru_cache(maxsize=4096)
    def encode(self, text):
        import torch
        if self.model is None:
            from transformers import AutoModel, AutoTokenizer
            torch.set_num_threads(self.spec.get('cpu_threads', 2))
            self.tokenizer = AutoTokenizer.from_pretrained(self.spec['path'], local_files_only=True)
            self.model = AutoModel.from_pretrained(self.spec['path'], local_files_only=True,
                                                 use_safetensors=True).cpu().eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        # Average all deterministic chunks, including late nodes/prompts of a DAG.
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = [ids[i:i + 254] for i in range(0, len(ids), 254)] or [[]]
        vectors, counts = [], []
        with torch.inference_mode():
            for chunk in chunks:
                tokens = self.tokenizer.prepare_for_model(chunk, return_tensors='pt')
                tokens = {k: v.unsqueeze(0) for k, v in tokens.items()}
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens['attention_mask'].unsqueeze(-1)
                vectors.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).squeeze(0).numpy())
                counts.append(max(1, len(chunk)))
        vector = np.average(vectors, axis=0, weights=counts) @ self.projection
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm <= 0:
            raise IntegrityError('non-finite/zero semantic feature')
        return tuple((vector / norm).tolist())


def unit(values):
    a = np.asarray(values, float)
    return (a / max(1., np.linalg.norm(a))).tolist()


class SemanticFeatures:
    def __init__(self, config, encoder=None):
        # JSON object key order is not part of the protocol digest. Feature
        # columns must therefore use a canonical role order, never dict order.
        self.spec = config['semantic']
        self.models = [m for m in ('local', 'middle', 'strong') if m in config['models']]
        self.encoder = encoder or FrozenSemantic(self.spec)
        self.dimension = self.spec['dimension']
        self.workflow_dimension = 8 + self.dimension + len(self.models)
        self.router_dimension = 3 + self.dimension + self.workflow_dimension

    def workflow(self, graph):
        from .policy import structural_features
        # Omit lineage IDs: ancestry does not change workflow semantics.
        public = {k: v for k, v in graph.definition().items() if k != 'parents'}
        semantic = self.encoder.encode(json.dumps(public, sort_keys=True, ensure_ascii=False))
        mixture = [sum(n.model == m for n in graph.nodes) / max(1, graph.model_calls) for m in self.models]
        return unit([*structural_features(graph), *semantic, *mixture])

    def route(self, public, graph):
        # Explicit allow-list also protects callers that accidentally pass extra fields.
        text = json.dumps({k: public[k] for k in ('question', 'context', 'options', 'public_tests', 'output_contract') if k in public},
                          sort_keys=True, ensure_ascii=False)
        question = public['question']
        return unit([1., min(len(question) / 2048, 1), min(sum(c.isdigit() for c in question) / 128, 1),
                     *self.encoder.encode(text), *self.workflow(graph)])
