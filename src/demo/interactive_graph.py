"""Standalone interactive playback of the generated, validated graph."""
import html,json,shutil
from pathlib import Path
from xml.etree import ElementTree as ET

ASSETS=Path(__file__).with_name('web')
NS='http://www.w3.org/2000/svg'


def graph_data(result):
    nodes=result.get('nodes',{})
    keys=list(nodes);ids={key:f'n{i}' for i,key in enumerate(keys)}
    milestones=[]
    def visit(key):
        node=nodes[key]
        for child in node['children']:
            if nodes[child]['type']=='Intermedia':visit(child)
        milestones.append(key)
    for key,node in nodes.items():
        if node['parent'] is None:visit(key)
    phases=[];phase_for={}
    for step,key in enumerate(milestones):
        causes=[c for c in nodes[key]['children'] if nodes[c]['type']=='Cause']
        evidence=[c for cause in causes for c in nodes[cause]['children'] if nodes[c]['type']=='Input']
        for kind,members in [('Evidence',evidence),('Rationale',causes),('Milestone',[key])]:
            phases.append({'kind':kind,'milestone':nodes[key]['content'],'step':step,'nodes':[ids[k] for k in members]})
            for member in members:phase_for[member]=len(phases)
    from src.demo.education import explain
    return {'status':result.get('status'),'diagnosis':result.get('diagnosis'),
        'education':result.get('education') or explain(result),
        'source_inputs':result.get('source_inputs',{}),'replay':result.get('replay',False),
        'case_id':result.get('case_id','Patient case'),
        'model':result.get('model_choice','d4'),'phases':phases,
        'milestones':[{'id':ids[k],'name':nodes[k]['content'],'phase':phase_for[k]} for k in milestones],
        'nodes':{ids[k]:{'content':n['content'],'type':n['type'],'source_field':n.get('source_field'),
            'criterion':n.get('kg_premise'),'parent':ids.get(n['parent']),
            'children':[ids[c] for c in n['children']], 'phase':phase_for.get(k,0)} for k,n in nodes.items()}}


def write_interactive_graph(result,folder):
    folder=Path(folder);data=graph_data(result)
    ET.register_namespace('',NS)
    root=ET.parse(folder/'graph.svg').getroot()
    root.set('aria-label','Interactive diagnostic reasoning chain')
    for group in root.iter(f'{{{NS}}}g'):
        element_id=group.get('id','')
        if element_id.startswith('node-n'):
            node=data['nodes'][element_id[5:]]
            group.set('tabindex','0');group.set('role','button')
            group.set('aria-label',f"{node['type']}: {node['content']}")
    template=(ASSETS/'graph.html').read_text(encoding='utf-8')
    page=template.replace('<!--GRAPH_SVG-->',ET.tostring(root,encoding='unicode')).replace(
        '<!--GRAPH_DATA-->',html.escape(json.dumps(data,ensure_ascii=False)))
    (folder/'graph.html').write_text(page,encoding='utf-8')
    for name in ['graph.js','graph.css']:shutil.copyfile(ASSETS/name,folder/name)
