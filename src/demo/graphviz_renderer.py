"""Render validated reasoning trees using project-contained Graphviz."""
import json,subprocess,textwrap
from pathlib import Path
from src.pipeline.schema import validate_tree

ROOT=Path(__file__).resolve().parents[2]


def dot_executable():
    manifest=ROOT/'tools'/'graphviz'/'manifest.json'
    if not manifest.exists():raise RuntimeError('Graphviz is missing. Run scripts/setup_graphviz.py first.')
    path=(ROOT/json.loads(manifest.read_text())['dot']).resolve()
    if not path.is_relative_to((ROOT/'tools'/'graphviz').resolve()) or not path.is_file():
        raise RuntimeError('Invalid project Graphviz installation')
    return path


def quoted(text):
    # DOT quoted labels: do not interpret quotes, HTML, or Graphviz backslash escapes.
    return '"'+str(text).replace('\\','\\\\').replace('"','\\"').replace('\r','').replace('\n','\\n')+'"'


def to_dot(result):
    nodes=result.get('nodes',{})
    if nodes:
        errors=validate_tree(nodes)
        if errors:raise ValueError(f'Cannot render invalid reasoning tree: {errors}')
    lines=['digraph ClinicalReasoning {',
        'graph [rankdir=LR, bgcolor="white", pad="0.35", nodesep="0.3", ranksep="0.65", dpi=130];',
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=11, margin="0.16,0.12", penwidth=1.2];',
        'edge [color="#8593a7", arrowsize=0.7];']
    if not nodes:
        message=f"No supported reasoning tree\nStatus: {result.get('status','unknown')}"
        lines.append(f'notice [label={quoted(message)}, fillcolor="#eef2f6", color="#c4ceda"];')
    ids={key:f'n{i}' for i,key in enumerate(nodes)}
    for key,node in nodes.items():
        kind=node['type'];root=node['parent'] is None
        label=({'Input':f"Evidence · input{node.get('source_field','?')}",'Cause':'Rationale',
                'Intermedia':'Proposed diagnosis' if root else 'Diagnostic milestone'})[kind]
        short=node['content']
        if len(short)>220:short=short[:217].rsplit(' ',1)[0]+'…'
        content='\n'.join(textwrap.fill(line,width=42,break_long_words=True,break_on_hyphens=False)
            for line in short.splitlines())
        color,fill=({'Input':('#88add6','#eaf3ff'),'Cause':('#dfbd62','#fff6d9'),
            'Intermedia':('#6ba894','#e5f4ed')})[kind]
        lines.append(f'{ids[key]} [id="node-{ids[key]}", label={quoted(label+chr(10)+chr(10)+content)}, tooltip={quoted(node["content"])}, color="{color}", fillcolor="{fill}", penwidth={2 if root else 1.2}];')
    for key,node in nodes.items():
        if node['parent'] is not None:lines.append(f'{ids[key]} -> {ids[node["parent"]]} [id="edge-{ids[key]}"];')
    return '\n'.join(lines+['}'])+'\n'


def render_graph(result,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    source=folder/'graph.dot';source.write_text(to_dot(result),encoding='utf-8')
    executable=dot_executable()
    for kind in ['svg','png']:
        output=folder/f'graph.{kind}'
        completed=subprocess.run([str(executable),f'-T{kind}',str(source),'-o',str(output)],
            capture_output=True,timeout=45,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if completed.returncode:raise RuntimeError('Graphviz rendering failed: '+completed.stderr.decode(errors='replace')[:500])
    from src.demo.interactive_graph import write_interactive_graph
    write_interactive_graph(result,folder)
    return {kind:f'graph.{kind}' for kind in ['dot','svg','png','html']}
