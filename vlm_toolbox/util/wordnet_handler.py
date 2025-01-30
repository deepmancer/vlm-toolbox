import nltk
from nltk.corpus import wordnet as wn


nltk.download('wordnet')

class WordNetHandler:
    def __init__(self):
        pass

    def _synset_info(self, synset):
        wn_id = f"{synset.pos()}{synset.offset():08d}"
        depth = synset.min_depth()
        return {
            'wn_id': wn_id,
            'synset': synset,
            'labels': synset.lemma_names(),
            'definition': synset.definition(),
            'examples': synset.examples(),
            'lemmas': [lemma.name() for lemma in synset.lemmas()],
            'antonyms': [lemma.antonyms()[0].name() for lemma in synset.lemmas() if lemma.antonyms()],
            'depth': depth
        }
    def get_info_from_synset(self, synset):
        return self._synset_info(synset)

    def get_info_from_wn_id(self, wn_id):
        synset = self.get_synset_from_wn_id(wn_id)
        return self._synset_info(synset)

    def get_synset_from_wn_id(self, wn_id):
        synset = wn.synset_from_pos_and_offset(wn_id[0], int(wn_id[1:]))
        return synset
    
    def get_info(self, input_):
        if isinstance(input_, str):
            return self.get_info_from_wn_id(input_)
        else:
            return self.get_info_from_synset(input_)

    def get_parent_from_synset(self, synset):
        hypernyms = synset.hypernyms()
        if hypernyms:
            return self._synset_info(hypernyms[0])
        return {}

    def get_all_parents_from_synset(self, synset):
        parents = []
        while synset.hypernyms():
            synset = synset.hypernyms()[0]
            parents.append(self._synset_info(synset))
        return parents

    def get_children_from_synset(self, synset):
        return [self._synset_info(child) for child in synset.hyponyms()]

    def get_parent_from_wn_id(self, wn_id):
        return self.get_parent_from_synset(wn.synset_from_pos_and_offset(wn_id[0], int(wn_id[1:])))

    def get_all_parents_from_wn_id(self, wn_id):
        return self.get_all_parents_from_synset(wn.synset_from_pos_and_offset(wn_id[0], int(wn_id[1:])))

    def get_children_from_wn_id(self, wn_id):
        return self.get_children_from_synset(wn.synset_from_pos_and_offset(wn_id[0], int(wn_id[1:])))

    def get_parent(self, input_):
        return self.get_parent_from_wn_id(input_) if isinstance(input_, str) else self.get_parent_from_synset(input_)

    def get_all_parents(self, input_):
        return self.get_all_parents_from_wn_id(input_) if isinstance(input_, str) else self.get_all_parents_from_synset(input_)

    def get_children(self, input_):
        return self.get_children_from_wn_id(input_) if isinstance(input_, str) else self.get_children_from_synset(input_)

    def complement_hierarchy(self, wn_ids, per_node_up_depth=1, per_node_down_depth=1, min_depth=None, max_depth=None):
        hierarchy = {}
        
        for wn_id in wn_ids:
            current_synset = self.get_synset_from_wn_id(wn_id)
            current_info = self._synset_info(current_synset)
            current_depth = current_info['depth']
            # If min_depth is not defined, start from the root node
            if min_depth is None:
                # Navigate upwards to the root, then start the descent
                ancestors = self.get_all_parents_from_synset(current_synset)
                if ancestors:
                    root_synset = self.get_synset_from_wn_id(ancestors[-1]['wn_id'])
                else:
                    root_synset = current_synset
                self._traverse_and_build(root_synset, hierarchy, 0, max_depth, per_node_min_depth, per_node_max_depth)
            else:
                # Ensure we're within the desired depth range before proceeding
                if min_depth <= current_depth <= (max_depth if max_depth is not None else current_depth):
                    self._traverse_and_build(current_synset, hierarchy, current_depth, max_depth, per_node_min_depth, per_node_max_depth)
                
        return hierarchy

    def get_synset_neighbors_adj_dict(self, synset):
        adj_dict = {}
        synset_info = self._synset_info(synset)
        parent = self.get_parent_from_synset(synset)
        children = self.get_children_from_synset(synset)
        adj_dict[synset_info['wn_id']] = parent.get(['wn_id'], None)
        for child in children:
            adj_dict[child['wn_id']] = synset_info['wn_id']
        return adj_dict

    def _traverse_and_build(self, synset, hierarchy, current_depth, max_depth, per_node_min_depth, per_node_max_depth):
        wn_id = f"{synset.pos()}{synset.offset():08d}"
        
        # Traverse downwards until the maximum depth or leaf nodes
        if max_depth is None or current_depth < max_depth:
            children = self.get_children_from_synset(synset)
            for child_info in children:
                child_synset = wn.synset_from_pos_and_offset(child_info['wn_id'][0], int(child_info['wn_id'][1:]))
                child_depth = self._synset_info(child_synset)['depth']
                
                # Check per node depth constraints
                if per_node_min_depth <= child_depth - current_depth <= (per_node_max_depth if per_node_max_depth is not None else child_depth - current_depth):
                    hierarchy[child_info['wn_id']] = wn_id
                    self._traverse_and_build(child_synset, hierarchy, child_depth, max_depth, per_node_min_depth, per_node_max_depth)
        
        # Also, consider navigating upwards if needed, similar logic can be applied
        
        return hierarchy
