"""Auditable one-to-one annotation matching, with field and diagnosis gates."""
from src.data.labels import normalize_diagnosis
from src.pipeline.schema import validate_tree


def matching(matrix,threshold):
    """Maximum-cardinality bipartite matching; deterministic similarity ordering."""
    adjacency=[sorted((j for j,s in enumerate(row) if s>=threshold),key=lambda j:(-row[j],j)) for row in matrix]
    owners={}
    def augment(i,seen):
        for j in adjacency[i]:
            if j in seen:continue
            seen.add(j)
            if j not in owners or augment(owners[j],seen):
                owners[j]=i;return True
        return False
    for i in range(len(matrix)):augment(i,set())
    return sorted((i,j,matrix[i][j]) for j,i in owners.items())


def prf(matched,predicted,reference):
    # Empty denominators score zero; counts preserve the distinction in reports.
    precision=matched/predicted if predicted else 0.
    recall=matched/reference if reference else 0.
    return {'matched':matched,'predicted':predicted,'reference':reference,
        'precision':precision,'recall':recall,
        'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}


def unique_observations(items):
    seen=set();result=[]
    for item in items:
        key=(item['source_field'],' '.join(item['text'].split()).casefold())
        if key not in seen:seen.add(key);result.append(item)
    return result


def reference_parts(sample):
    observations=unique_observations([{'text':o.content,'source_field':o.source_field} for o in sample.observations])
    steps=[]
    for node in sample.rationales:
        parent=sample.record_nodes.get(node.parent_id)
        evidence=[sample.record_nodes[c] for c in node.child_ids if sample.record_nodes[c].is_observation]
        if parent and parent.is_diagnosis and evidence:
            steps.append({'diagnosis':parent.content,'rationale':node.content,
                'evidence':unique_observations([{'text':o.content,'source_field':o.source_field} for o in evidence])})
    return observations,steps


def predicted_steps(nodes):
    steps=[]
    for node in nodes.values():
        if node['type']!='Cause':continue
        parent=nodes.get(node['parent'])
        if not parent or parent['type']!='Intermedia':continue
        steps.append({'diagnosis':parent['content'],'rationale':node['content'],
            'evidence':unique_observations([{'text':nodes[c]['content'],'source_field':nodes[c]['source_field']}
                for c in node['children'] if nodes[c]['type']=='Input'])})
    return steps


def grounded(item,inputs):
    field=item.get('source_field')
    return type(field) is int and 1<=field<=6 and bool(item.get('text','').strip()) and item['text'] in inputs.get_field(field)


def score_case(result,sample,encoder,threshold=.8):
    reference,ref_steps=reference_parts(sample)
    observations=unique_observations(result.get('observations',[]))
    errors=validate_tree(result.get('nodes',{}),sample.inputs) if result.get('nodes') else []
    steps=predicted_steps(result.get('nodes',{})) if not errors else []
    texts=[o['text'] for o in observations+reference]
    texts += [s['rationale'] for s in steps+ref_steps]
    texts += [o['text'] for s in steps+ref_steps for o in s['evidence']]
    encoder.prepare(texts)
    def obs_score(a,b):
        if a['source_field']!=b['source_field'] or not grounded(a,sample.inputs):return -1.
        if ' '.join(a['text'].split()).casefold()==' '.join(b['text'].split()).casefold():return 1.
        return encoder.similarity(a['text'],b['text'])
    obs_matrix=[[obs_score(a,b) for b in reference] for a in observations]
    obs_pairs=matching(obs_matrix,threshold)
    exact_pairs=matching([[1. if a['source_field']==b['source_field'] and
        ' '.join(a['text'].split()).casefold()==' '.join(b['text'].split()).casefold() and grounded(a,sample.inputs)
        else -1. for b in reference] for a in observations],1.)
    def step_score(a,b):
        if normalize_diagnosis(a['diagnosis'])!=normalize_diagnosis(b['diagnosis']):return -1.
        evidence=max((obs_score(x,y) for x in a['evidence'] for y in b['evidence']),default=-1.)
        return min(evidence,encoder.similarity(a['rationale'],b['rationale']))
    step_matrix=[[step_score(a,b) for b in ref_steps] for a in steps]
    step_pairs=matching(step_matrix,threshold)
    predicted_labels={normalize_diagnosis(n['content']) for n in result.get('nodes',{}).values() if n['type']=='Intermedia'} if not errors else set()
    reference_labels={normalize_diagnosis(n.content) for n in sample.diagnoses}
    root_label=sample.root_node.content if sample.root_node else sample.pdd
    return {'sample_id':sample.sample_id,'status':result['status'],
        'category_correct':result.get('category')==sample.disease_category,
        'diagnosis_correct':normalize_diagnosis(result.get('diagnosis') or '')==normalize_diagnosis(sample.pdd),
        'diagnosis_annotation_root_correct':normalize_diagnosis(result.get('diagnosis') or '')==normalize_diagnosis(root_label),
        'reference_metadata_pdd':sample.pdd,'reference_annotation_root':root_label,
        'observations_semantic':prf(len(obs_pairs),len(observations),len(reference)),
        'observations_exact':prf(len(exact_pairs),len(observations),len(reference)),
        'explanation_steps':prf(len(step_pairs),len(steps),len(ref_steps)),
        'diagnostic_milestones':prf(len(predicted_labels&reference_labels),len(predicted_labels),len(reference_labels)),
        'grounded_observations':sum(grounded(o,sample.inputs) for o in observations),
        'tree_errors':errors,
        'observation_matches':[{'predicted':observations[i],'reference':reference[j],'similarity':v} for i,j,v in obs_pairs],
        'explanation_matches':[{'predicted':steps[i],'reference':ref_steps[j],'similarity':v} for i,j,v in step_pairs]}
