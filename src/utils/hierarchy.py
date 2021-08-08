import json
import torch
import yaml
from itertools import groupby
from operator import itemgetter
from collections import OrderedDict
from networkx.readwrite.json_graph import tree_graph
from src.datamodules.transforms import MapLabelsToHeads
import networkx as nx


def load_yaml(config_file):
    return yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)


class ComputeHPredictions:
    full_graph: nx.DiGraph

    def __init__(
            self, labels_to_heads_mapping,
            hierarchy_json="/code/mappings/hierarchy_0k.json"):
        self.hierarchy_json = hierarchy_json
        h0 = self._get_h_from_json()
        self.head_positions = []
        self.mappings = []
        for _lod, _sids in labels_to_heads_mapping.lod_and_set_id.items():
            for _nc, (_i, _mapping) in _sids.items():
                self.head_positions.append((_lod, _nc))
                self.mappings.append(_mapping)
        h_updated = self.add_metric_to_hierarchy(h0)
        self.results = []
        self.recursive_get(h_updated)
        self.full_graph = tree_graph(
            h_updated, attrs=dict(children="children", id="node_id"))

    def _get_h_from_json(self):
        with open(self.hierarchy_json, 'r') as fh:
            h0 = json.load(fh)
        return {
            'set_id': 0,
            'name': 'ROOT',
            'children': h0['ROOT']
        }

    def get_head_id_from_set_id(self, head_id):
        _lod, _nc = self.head_positions[head_id]
        mapping = self.mappings[head_id]
        sids = torch.where(mapping > 0)[0]
        return {
            _cid.item(): (_sid.item(), _lod, self.full_graph.nodes[f'{_lod + 1}_{_sid}']['head_id'])
            for _cid, _sid in zip(mapping[sids], sids)
        }

    @classmethod
    def from_eval_conf(cls, eval_config_path):
        # eval_config_path = '/code/configs/test/semantic/eval_config_semantic_seg_hierarchical_v2.yaml'
        # metric_results = 'metrics_seg_hierarchical_v3_version_0_epoch_77.yaml'
        # metric_results = '/logs/metrics_seg_hierarchical_v2_version_0_epoch_134.yaml'

        eval_config = load_yaml(eval_config_path)
        metrics_file = eval_config['metrics_file']
        mapping_file = eval_config['mapping_file']
        selected_lods = eval_config['selected_lods']
        hierarchy_file = eval_config['hierarchy_file']
        head_hierarchy = get_output_hierarchy(
            hierarchy_file, selected_lods=selected_lods)
        labels_to_heads_mapping = MapLabelsToHeads(
            head_hierarchy, mapping_file=mapping_file)
        return cls(labels_to_heads_mapping, hierarchy_json=hierarchy_file)

    def recursive_get(self, data):
        node = self.check_head_id(data)
        if node:
            self.results.append(node)
        if 'children' in data:
            for _child in data['children']:
                self.recursive_get(_child)

    @staticmethod
    def check_head_id(x):
        if x['head_id'] is not None:
            return {
                _k: _v for _k, _v in x.items() if _k != 'children'
            }
        else:
            return False

    @staticmethod
    def _get_child_ids(node):
        if 'children' in node and node['children']:
            return frozenset(_child['set_id'] for _child in node['children'])
        else:
            return frozenset()

    def add_metric_to_hierarchy(self, data, lod=1):
        data['lod'] = lod
        data['node_id'] = f"{lod}_{data['set_id']}"
        key = (lod, self._get_child_ids(data))
        if key in self.head_positions:
            data['head_id'] = self.head_positions.index(key)
        else:
            data['head_id'] = None
        if 'children' in data and data['children']:
            children = [
                self.add_metric_to_hierarchy(_child, lod=lod + 1)
                for _child in data['children']
            ]
            data['children'] = children
        return data


def get_output_hierarchy(hierarchy_file, selected_nodes=None, selected_lods=None):
    with open(hierarchy_file) as fh:
        hierarchy_0k = json.loads(fh.read())
    branchings = compute_branchings({
        'name': 'ROOT',
        'set_id': None,
        'children': hierarchy_0k['ROOT']
    }, lod=1)
    if selected_nodes is not None:
        selected_nodes = [tuple(_e) for _e in selected_nodes]
    projections_to_branchings = {}
    for _key, _group_iter in groupby(sorted(branchings, key=itemgetter(0)), itemgetter(0)):
        if (selected_lods is not None) and (_key not in selected_lods):
            continue
        projections_to_branchings[_key] = {}
        for _br in _group_iter:
            _lod, _children_num, _root_set_id, _children_set_ids = _br
            if selected_nodes is None or ((_lod, _root_set_id) in selected_nodes):
                projections_to_branchings[_key][(_root_set_id, _children_set_ids)] = _children_num
    return OrderedDict(sorted(projections_to_branchings.items()))


def compute_branchings(obj, struct=None, lod=0):
    if struct is None:
        struct = []
    if 'children' in obj and obj['children']:
        if len(obj['children']) > 1:
            struct.append((
                lod,
                len(obj['children']),
                obj['set_id'],
                tuple(_e['set_id'] for _e in obj['children'])
            ))
        for _child in obj['children']:
            compute_branchings(_child, struct=struct, lod=lod + 1)
    return struct


def normalize_attribute(data, attr='count'):
    if 'children' in data:
        if data['children']:
            children = [
                normalize_attribute(_child)
                for _child in data['children']
                if _child.get(attr, 0) > 0
            ]
            if children:
                flat_sum = float(sum(_c.get(attr, 0) for _c in children))
                for _c in children:
                    if attr in _c:
                        _c[f'norm_{attr}'] = _c[attr] / flat_sum
                    else:
                        _c[f'norm_{attr}'] = 0.0
                data['children'] = children
            else:
                del data['children']
        else:
            del data['children']
    return data


def convert(to_file=None, hierarchy_json="mappings/hierarchy_0k.json"):
    with open(hierarchy_json, 'r') as fh:
        h0 = json.load(fh)
    data = {'name': 'ROOT', 'children': h0['ROOT']}
    proc_data = normalize_attribute(data)
    if to_file is not None:
        with open("mappings/norm_hierarchy_0k.json", 'w+') as fh:
            fh.write(json.dumps(proc_data))
    return proc_data


if __name__ == '__main__':
    convert(to_file="mappings/norm_hierarchy_0k.json")
