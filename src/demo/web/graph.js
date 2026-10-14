'use strict';
const data=JSON.parse(document.getElementById('graph-data').textContent);
const el=id=>document.getElementById(id),svg=el('canvas').querySelector('svg'),group=svg.querySelector('g.graph');
const total=data.phases.length;
let cursor=total,selected=null,timer=null,view=null,drag=null;
const fields=['','Chief complaint','History of present illness','Past medical history','Family history','Physical examination','Laboratory / investigations'];
const kind=node=>node.type==='Input'?'Source evidence':node.type==='Cause'?'Rationale':node.parent?'Diagnostic milestone':'Proposed diagnosis';
group.removeAttribute('transform');svg.removeAttribute('width');svg.removeAttribute('height');
el('model').textContent=data.model==='d4'?'D4 · step 400':'Base 4B';
el('timeline').max=total;
const nodeEl=id=>document.getElementById('node-'+id);
function stop(){clearInterval(timer);timer=null;el('play').textContent='▶ Play';}
function setView(box){view=box;svg.setAttribute('viewBox',[box.x,box.y,box.w,box.h].join(' '));}
function fit(){
  const boxes=Object.keys(data.nodes).filter(id=>!nodeEl(id).classList.contains('filtered')).map(id=>nodeEl(id).getBBox());
  if(!boxes.length){const b=group.getBBox();return setView({x:b.x-15,y:b.y-15,w:Math.max(b.width+30,100),h:Math.max(b.height+30,100)});}
  const x=Math.min(...boxes.map(b=>b.x)),y=Math.min(...boxes.map(b=>b.y));
  setView({x:x-25,y:y-25,w:Math.max(...boxes.map(b=>b.x+b.width))-x+50,h:Math.max(...boxes.map(b=>b.y+b.height))-y+50});
}
function point(event){const p=svg.createSVGPoint();p.x=event.clientX;p.y=event.clientY;return p.matrixTransform(svg.getScreenCTM().inverse());}
function zoom(factor,center=null){if(!view)return;center=center||{x:view.x+view.w/2,y:view.y+view.h/2};const w=Math.max(60,Math.min(20000,view.w*factor)),h=view.h*w/view.w;setView({x:center.x-(center.x-view.x)*w/view.w,y:center.y-(center.y-view.y)*h/view.h,w,h});}
function inspect(id){
  selected=id;const node=data.nodes[id];if(!node)return;
  el('node-kind').textContent=kind(node).toUpperCase();
  el('node-title').textContent=node.type==='Intermedia'?node.content:node.type==='Input'?`input${node.source_field} · ${fields[node.source_field]||'Clinical finding'}`:'Why this step was chosen';
  el('node-text').textContent=node.content;
  el('node-source').textContent=node.type==='Input'?'Verbatim source passage':node.type==='Cause'?'Model-generated explanation':'Selected diagnostic milestone';
  el('criteria').hidden=!node.criterion;el('criterion').textContent=node.criterion||'';
  el('connections').replaceChildren();
  for(const linked of [...node.children,...(node.parent?[node.parent]:[])]){
    const other=data.nodes[linked],button=document.createElement('button');button.type='button';
    button.textContent=`${kind(other)} · ${other.content.length>95?other.content.slice(0,92)+'…':other.content}`;
    button.addEventListener('click',()=>{stop();cursor=el('focus').checked?other.phase:Math.max(cursor,other.phase);inspect(linked);update();if(el('focus').checked)fit();});
    el('connections').append(button);
  }
}
function update(){
  const phase=data.phases[cursor-1],focus=el('focus').checked&&phase;
  el('timeline').value=cursor;
  el('phase').textContent=!total?`No supported reasoning tree · ${data.status}`:cursor===0?'Ready to replay':cursor===total&&!timer?'All steps visible':`${cursor}/${total} · ${phase.kind}: ${phase.milestone}`;
  el('previous').disabled=cursor<=0;el('next').disabled=cursor>=total;
  for(const [id,node] of Object.entries(data.nodes)){
    const item=nodeEl(id),nodePhase=data.phases[node.phase-1];
    item.classList.toggle('future',node.phase>cursor);
    item.classList.toggle('current',node.phase===cursor&&cursor>0);
    item.classList.toggle('selected',id===selected);
    item.classList.toggle('filtered',(!el('evidence').checked&&node.type==='Input')||Boolean(focus&&nodePhase&&nodePhase.step!==phase.step));
  }
  for(const [id,node] of Object.entries(data.nodes)){
    if(!node.parent)continue;const edge=document.getElementById('edge-'+id);if(!edge)continue;
    const reveal=Math.max(node.phase,data.nodes[node.parent].phase);
    edge.classList.toggle('future',reveal>cursor);
    edge.classList.toggle('active-edge',Boolean(timer&&reveal===cursor));
    edge.classList.toggle('filtered',nodeEl(id).classList.contains('filtered')||nodeEl(node.parent).classList.contains('filtered'));
  }
  for(const button of el('milestones').children)button.classList.toggle('current',data.phases[Number(button.dataset.phase)-1]?.step===phase?.step);
  if(typeof renderGuide==='function')renderGuide();
}
function seek(position){stop();cursor=Math.max(0,Math.min(total,position));const phase=data.phases[cursor-1];if(phase?.nodes.length)inspect(phase.nodes[0]);update();if(el('focus').checked)fit();}
function advance(){
  cursor=Math.min(total,cursor+1);const phase=data.phases[cursor-1];if(phase?.nodes.length)inspect(phase.nodes[0]);
  if(cursor===total)stop();update();if(el('focus').checked)fit();
}
function play(){if(timer){stop();update();return;}if(cursor===total)cursor=0;timer=setInterval(advance,Number(el('speed').value));el('play').textContent='Ⅱ Pause';advance();}
for(const milestone of data.milestones){const button=document.createElement('button');button.type='button';button.textContent=milestone.name;button.dataset.phase=milestone.phase;button.addEventListener('click',()=>seek(milestone.phase));el('milestones').append(button);}
for(const id of Object.keys(data.nodes)){
  const node=nodeEl(id);
  function select(){stop();cursor=el('focus').checked?data.nodes[id].phase:Math.max(cursor,data.nodes[id].phase);inspect(id);update();}
  node.addEventListener('click',select);node.addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key)){event.preventDefault();select();}});
}
el('play').addEventListener('click',play);el('previous').addEventListener('click',()=>seek(cursor-1));el('next').addEventListener('click',()=>seek(cursor+1));
el('overview').addEventListener('click',()=>{el('focus').checked=false;el('evidence').checked=true;seek(total);fit();});
el('timeline').addEventListener('input',event=>seek(Number(event.target.value)));
el('speed').addEventListener('change',()=>{if(timer){stop();play();}});
for(const id of ['evidence','focus'])el(id).addEventListener('change',()=>{update();fit();});
el('fit').addEventListener('click',fit);el('zoom-in').addEventListener('click',()=>zoom(.8));el('zoom-out').addEventListener('click',()=>zoom(1.25));
el('canvas').addEventListener('wheel',event=>{event.preventDefault();zoom(event.deltaY>0?1.12:.88,point(event));},{passive:false});
el('canvas').addEventListener('pointerdown',event=>{
  if(event.target.closest('g.node')||event.button!==0)return;
  drag={clientX:event.clientX,clientY:event.clientY,view:{...view}};el('canvas').setPointerCapture(event.pointerId);el('canvas').classList.add('dragging');
});
el('canvas').addEventListener('pointermove',event=>{
  if(!drag)return;const matrix=svg.getScreenCTM();
  setView({...drag.view,x:drag.view.x-(event.clientX-drag.clientX)/matrix.a,y:drag.view.y-(event.clientY-drag.clientY)/matrix.d});
});
function endDrag(){drag=null;el('canvas').classList.remove('dragging');}
el('canvas').addEventListener('pointerup',endDrag);el('canvas').addEventListener('pointercancel',endDrag);
if(!total){for(const id of ['play','previous','next','timeline','overview'])el(id).disabled=true;el('node-text').textContent='No supported diagnostic pathway was produced. Review the case report for the status and any warnings.';}
else inspect(data.milestones.at(-1).id);

let guided=false,lastGuideKey='';
const teaching=data.education?.steps||[];
function addText(parent,text,terms=[]){
  parent.replaceChildren();text=String(text||'');
  const unique=[...new Map(terms.map(t=>[t.term.toLowerCase(),t])).values()].filter(t=>t.term.trim()).sort((a,b)=>b.term.length-a.term.length);
  if(!unique.length){parent.textContent=text;return;}
  const escape=value=>value.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  const pattern=new RegExp('\\b('+unique.map(t=>escape(t.term)).join('|')+')\\b','gi');
  let at=0;for(const match of text.matchAll(pattern)){parent.append(document.createTextNode(text.slice(at,match.index)));const term=unique.find(t=>t.term.toLowerCase()===match[0].toLowerCase()),button=document.createElement('button');button.className='term-link';button.textContent=match[0];button.title=term.meaning;button.addEventListener('click',()=>{el('source-heading').textContent=term.term;el('source-text').textContent=term.meaning+'\n\nSource: '+term.source;el('source-dialog').querySelector('p').textContent='Terminology explanation';el('source-dialog').showModal();});parent.append(button);at=match.index+match[0].length;}parent.append(document.createTextNode(text.slice(at)));
}
function showSource(e){
  el('source-heading').textContent=fields[e.source_field]||'Original patient record';el('source-dialog').querySelector('p').textContent='Highlighted text is the exact passage cited by this step.';
  const source=data.source_inputs?.['input'+e.source_field]||e.text,index=source.indexOf(e.text);el('source-text').replaceChildren();
  if(index>=0){el('source-text').append(document.createTextNode(source.slice(0,index)));const mark=document.createElement('mark');mark.textContent=e.text;el('source-text').append(mark,document.createTextNode(source.slice(index+e.text.length)));}else el('source-text').textContent=source;
  el('source-dialog').showModal();el('source-text').querySelector('mark')?.scrollIntoView({block:'center'});
}
function renderGuide(){
  if(!el('guide'))return;
  const phase=data.phases[Math.max(0,cursor-1)],stepIndex=phase?.step||0,s=teaching[stepIndex],clinical=el('clinical').checked;
  el('guide').hidden=!guided;el('graph-workspace').hidden=guided;
  if(!s){el('step-name').textContent='No supported diagnostic pathway';el('step-definition').textContent='The pipeline could not establish a pathway for this case. The saved report contains its status and warnings.';el('guide').querySelector('.guide-columns').hidden=true;return;}
  const key=stepIndex+':'+clinical;
  if(lastGuideKey!==key){lastGuideKey=key;
    el('step-count').textContent=`Step ${stepIndex+1} of ${teaching.length}`;el('step-name').textContent=s.name;addText(el('step-definition'),s.definition,s.terms);
    addText(el('step-plain'),s.plain,s.terms);addText(el('step-why'),clinical?s.rationale:s.why,s.terms);el('step-original').textContent=s.rationale;el('original-rationale').open=clinical;
    addText(el('step-conclusion'),`The model selected ${s.name}. ${s.definition}`,s.terms);addText(el('step-unresolved'),s.unresolved,s.terms);addText(el('step-criterion'),s.criterion,s.terms);
    el('step-evidence').replaceChildren();for(const e of s.evidence){const box=document.createElement('div');box.className='evidence-passage';const quote=document.createElement('blockquote');addText(quote,e.text,s.terms);const source=document.createElement('button');source.className='source-link';source.textContent='View in '+(fields[e.source_field]||'patient record');source.addEventListener('click',()=>showSource(e));box.append(quote,source);el('step-evidence').append(box);}
    el('step-checks').replaceChildren();for(const check of s.checks){const item=document.createElement('div');item.className='criterion-check';const status=document.createElement('span');status.className='check-status '+check.status;status.textContent={supporting:'Supporting',conflicting:'Conflicting',not_documented:'Not documented in cited evidence'}[check.status];const quote=document.createElement('blockquote');addText(quote,check.criterion_quote,s.terms);const explanation=document.createElement('p');addText(explanation,check.explanation,s.terms);item.append(status,quote,explanation);for(const id of check.evidence_ids){const e=s.evidence.find(e=>e.id===id);if(e){const button=document.createElement('button');button.className='source-link';button.textContent='Inspect cited passage';button.addEventListener('click',()=>showSource(e));item.append(button);}}el('step-checks').append(item);}
    if(!s.checks.length)el('step-checks').textContent='No separate criteria comparison is available. Inspect the original criteria below.';
    el('step-terms').replaceChildren();const terms=[...new Map(s.terms.map(t=>[t.term.toLowerCase(),t])).values()];for(const t of terms){const details=document.createElement('details'),summary=document.createElement('summary'),p=document.createElement('p'),small=document.createElement('small');summary.textContent=t.term;p.textContent=t.meaning;small.textContent=t.source.startsWith('http')?'Curated definition · MedlinePlus':'Model-generated definition';details.append(summary,p,small);el('step-terms').append(details);}if(!terms.length)el('step-terms').textContent='No glossary entries were identified. Ambiguous abbreviations should not be assumed.';
    el('education-mode').textContent=s.mode==='generated'?'Plain-language teaching added after diagnosis. Source references are checked; medical interpretations remain model-generated.':'Original-output view: a generated teaching explanation was unavailable for this step.';
  }
  const stage=cursor?((cursor-1)%3):0;
  for(const [i,id] of ['finding-panel','reason-panel','conclusion-panel'].entries()){el(id).classList.toggle('pending',i>stage);el(id).classList.toggle('active-phase',i===stage);}
}
function switchView(value){guided=value;el('guided-view').setAttribute('aria-pressed',value);el('graph-view').setAttribute('aria-pressed',!value);renderGuide();if(!value){el('focus').checked=true;update();fit();}}
el('guided-view').addEventListener('click',()=>switchView(true));el('graph-view').addEventListener('click',()=>switchView(false));el('clinical').addEventListener('change',renderGuide);el('close-source').addEventListener('click',()=>el('source-dialog').close());
document.addEventListener('keydown',event=>{if(el('source-dialog').open||['INPUT','SELECT','TEXTAREA','BUTTON','SUMMARY'].includes(event.target.tagName))return;if(event.key==='ArrowRight'){event.preventDefault();seek(cursor+1);}if(event.key==='ArrowLeft'){event.preventDefault();seek(cursor-1);}if(event.code==='Space'&&total){event.preventDefault();play();}});
window.addEventListener('message',event=>{if(event.source===parent&&event.origin===location.origin&&event.data?.type==='presentation')document.body.classList.toggle('present',Boolean(event.data.active));});
new ResizeObserver(()=>{if(parent!==window)parent.postMessage({type:'pathway-height',height:document.documentElement.scrollHeight},location.origin);}).observe(document.body);
update();fit();

// Start the pathway once on opening; manual controls still pause or replay it.
if(total)play();
