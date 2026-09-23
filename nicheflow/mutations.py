"""Auditable bounded mutation grammar; graph validity remains a separate gate."""
from .spec import digest


def validate_mutation(parents, candidate):
    if len(parents) != 1:
        raise ValueError("bounded mutation requires exactly one parent")
    parent = parents[0]
    old, new = ({n.id: n for n in g.nodes} for g in (parent, candidate))
    shared = old.keys() & new.keys()
    if not shared:
        raise ValueError("mutation must retain at least one parent node ID")
    replaced = [i for i in sorted(shared) if any(getattr(old[i], k) != getattr(new[i], k)
                for k in ('operator', 'prompt', 'model', 'temperature', 'source', 'signature', 'subgroup'))]
    rewired = [i for i in sorted(shared) if old[i].inputs != new[i].inputs]
    inserted, deleted = sorted(new.keys() - old.keys()), sorted(old.keys() - new.keys())
    topology = parent.topology_primitive != candidate.topology_primitive or parent.output != candidate.output
    if not (replaced or rewired or inserted or deleted or topology):
        raise ValueError("unchanged candidate; lineage alone is not a mutation")
    return {'grammar': 'retained_node_dag_edits_v1', 'parent': parent.version,
            'replaced': replaced, 'rewired': rewired, 'inserted': inserted,
            'deleted': deleted, 'topology_changed': topology,
            'validation': 'post_generation_typed_DAG_not_token_constrained_decoding'}
