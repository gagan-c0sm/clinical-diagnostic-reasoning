"""Strict clinical input parsing and an escaped standalone HTML report."""
import html
from src.data.structures import InputContent


def parse_case(data):
    if not isinstance(data,dict):raise ValueError('Case must be a JSON object')
    allowed={f'input{i}' for i in range(1,7)}|{'case_id'}
    unknown=set(data)-allowed
    if unknown:raise ValueError(f'Unknown fields: {sorted(unknown)}. Supply input1 through input6, plus optional case_id.')
    if any(not isinstance(v,str) for v in data.values()):raise ValueError('All case fields must be strings')
    if sum(len(v) for v in data.values())>200000:raise ValueError('Case text is too large')
    case=InputContent(**{f'input{i}':data.get(f'input{i}','') for i in range(1,7)})
    if not case.non_empty_fields():raise ValueError('At least one clinical field is required')
    return case


def render_report(result,inputs,graph=False):
    esc=lambda value:html.escape(str(value),quote=True)
    nodes=result.get('nodes',{});steps=[]
    education=result.get('education',{})
    teaching=[]
    for step in education.get('steps',[]):
        terms=''.join(f'<dt>{esc(t["term"])}</dt><dd>{esc(t["meaning"])}</dd>' for t in step.get('terms',[]))
        checks=''.join(f'<li><b>{esc(c["status"].replace("_"," "))}</b>: {esc(c["criterion_quote"])}<p>{esc(c["explanation"])}</p></li>' for c in step.get('checks',[]))
        teaching.append(f'<section class="step"><h2>{esc(step["name"])}</h2><p>{esc(step.get("definition",""))}</p>'
            f'<h3>In everyday language</h3><p>{esc(step["plain"])}</p><h3>Why it matters</h3><p>{esc(step["why"])}</p>'
            f'<h3>What remains unresolved</h3><p>{esc(step["unresolved"])}</p><ul>{checks}</ul><dl>{terms}</dl></section>')
    for key,node in nodes.items():
        if node['type']!='Intermedia':continue
        causes=[]
        for child in node['children']:
            cause=nodes[child]
            if cause['type']!='Cause':continue
            evidence=''.join(f'<li><b>input{nodes[o].get("source_field", "?")}</b> — {esc(nodes[o]["content"])}</li>'
                             for o in cause['children'] if nodes[o]['type']=='Input')
            causes.append(f'<div class="cause"><strong>Rationale</strong><p>{esc(cause["content"])}</p><ul>{evidence}</ul></div>')
        steps.append(f'<section class="step"><h2>{esc(node["content"])}</h2>'+''.join(causes)+'</section>')
    source=''.join(f'<h3>input{i}: {esc(InputContent.FIELD_NAMES[i])}</h3><pre>{esc(inputs.get_field(i))}</pre>' for i in inputs.non_empty_fields())
    warnings=''.join(f'<li>{esc(w)}</li>' for w in result.get('warnings',[]))
    graph_section=('''<section class="step"><h2>Reasoning graph</h2>
<p class="note">Blue: source evidence · Yellow: rationale · Green: diagnostic milestone. Arrows point from supporting evidence toward the proposed diagnosis.</p>
<p><a href="graph.html" target="_blank" rel="noopener">Open interactive playback</a> · <a href="graph.svg" download>SVG</a> · <a href="graph.png" download>PNG</a> · <a href="graph.dot" download>DOT</a> · <a href="result.json" download>Result JSON</a></p>
<iframe src="graph.html" title="Animated interactive diagnostic reasoning chain" sandbox="allow-scripts allow-same-origin allow-downloads" style="width:100%;height:1100px;border:1px solid #dbe3ec;border-radius:10px"></iframe></section>''' if graph else '')
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clinical reasoning case report</title><style>
body{{font:16px/1.55 system-ui,sans-serif;background:#f4f6f9;color:#17293c;margin:0}}main{{max-width:1000px;margin:32px auto;padding:24px}}
header,.step,details{{background:white;padding:24px;border-radius:12px;margin:18px 0;border:1px solid #dbe3ec}}
h1{{font-size:28px;margin:0}}h2{{font-size:20px;color:#126348}}.cause{{background:#fff7dc;padding:16px;margin:12px 0;border-radius:8px}}
li{{background:#eaf3ff;margin:8px 0;padding:10px;border-radius:6px}}ul{{padding-left:22px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}}
.status{{font-weight:600;color:#475569}}.note{{color:#526375;font-size:14px}}dt{{font-weight:600}}dd{{margin:4px 0 14px}}@media print{{body{{background:white}}main{{margin:0;padding:0}}iframe{{display:none}}.step{{break-inside:avoid}}details{{display:block}}}}</style><main>
<header><p class="status">{esc(result.get('status'))} · {esc(result.get('model_choice','base'))}</p>
<h1>Proposed diagnosis: {esc(result.get('diagnosis') or 'Not established')}</h1>
<p>Category: {esc(result.get('category') or 'Not established')} · {result.get('seconds',0):.1f} seconds</p>
<p class="note">Prototype output. Source grounding and structural validation do not establish medical correctness.</p></header>
<p class="note">{'Saved D4 run.' if result.get('replay') else 'New local analysis.'} Teaching explanations are added after diagnosis. Criteria comparisons are model assessments of cited evidence, not independent verification.</p>
{''.join(teaching)}{graph_section}<h2>Original evidence and model reasoning</h2>{''.join(steps) or '<p>No supported reasoning tree was produced.</p>'}
<p>{esc(result.get('error',''))}</p><ul>{warnings}</ul><details><summary>Original clinical input</summary>{source}</details>
</main></html>'''
