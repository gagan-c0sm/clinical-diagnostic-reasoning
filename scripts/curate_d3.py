"""Deterministic diversity strata and approximate near-duplicate screening."""
import hashlib,json,random,re,shutil,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data import DiReCTDataset


def words(text):return re.findall(r'[a-z]+',text.casefold())


def main():
    folder=ROOT/'data'/'d3';out=ROOT/'data'/'d3_curated';out.mkdir(exist_ok=False)
    tokens=[json.loads(l) for l in (folder/'train_tokens.jsonl').open()]
    order=list(range(len(tokens)));random.Random(42).shuffle(order)
    used={tokens[i]['input_sha256'] for i in order[:800]}
    token_by_id={r['input_sha256']:r for r in tokens}
    records=[json.loads(l) for l in (folder/'train.jsonl').open(encoding='utf-8')]
    categories=DiReCTDataset(ROOT/'mimic-iv-ext-direct-1.0.0').categories
    vocabulary=set()
    for r in records:
        vocabulary.update(w for w in words(r.get('final_diagnosis','')) if len(w)>4 and w not in {'acute','chronic','disease','syndrome','associated','primary','secondary'})
    df=Counter(w for r in records for w in set(words(r['input'])) if w in vocabulary)
    groups=defaultdict(list)
    for r in records:
        key=r['input_sha256']
        if key in used:continue
        text=r['input'].casefold();terms=set(words(text))
        topic=r.get('final_diagnosis') or min((w for w in terms if w in vocabulary),key=lambda w:(df[w],w),default='other')
        length=len(token_by_id[key]['input_ids']);length_bin=min(length//250,4)
        skills=[name for name,pattern in [('quantitative',r'\d'),('differential',r'differential|rather than|however|although'),
            ('uncertainty',r'possible|uncertain|suspect|might'),('negative_evidence',r'\bno\b|denies|without'),
            ('tests',r'\bct\b|\bmri\b|laboratory|biopsy|culture')] if re.search(pattern,text)] or ['other']
        relevant=any(c.casefold() in text for c in categories)
        r['selection_tags']={'topic_proxy':topic,'length_bin':length_bin,'skills_proxy':skills,'direct_category_lexical_match':relevant}
        groups[(r['source'],relevant,topic,length_bin,skills[0])].append(r)
    rng=random.Random(42)
    for group in groups.values():rng.shuffle(group)
    keys=list(groups);rng.shuffle(keys)
    selected=[];signature_index=defaultdict(list);shingle_cache={};near_removed=0
    def accept(r):
        nonlocal near_removed
        ws=words(r['input']);shingles={' '.join(ws[i:i+5]) for i in range(max(1,len(ws)-4))}
        hashes=[hashlib.sha256(s.encode()).digest() for s in shingles]
        signatures=[min(h[i:i+8] for h in hashes) for i in (0,8,16)]
        candidates={idx for sig in signatures for idx in signature_index[sig]}
        for idx in candidates:
            other=shingle_cache[idx]
            if len(shingles&other)/max(1,len(shingles|other))>=.8:
                near_removed+=1;return False
        idx=len(selected);selected.append(r);shingle_cache[idx]=shingles
        for sig in signatures:signature_index[sig].append(idx)
        return True
    for source,quota in [('zou-lab/MedCaseReasoning',960),('FreedomIntelligence/medical-o1-reasoning-SFT',640)]:
        before=len(selected)
        # Half the quota targets explicit project-category lexical overlap where available.
        for require_relevance,limit in [(True,quota//2),(False,quota)]:
            eligible=[k for k in keys if k[0]==source and (not require_relevance or k[1])]
            while len(selected)-before<limit:
                progress=False
                for key in eligible:
                    if len(selected)-before>=limit:break
                    while groups[key]:
                        if accept(groups[key].pop()):progress=True;break
                if not progress:break
        if len(selected)-before!=quota:raise ValueError('Source quota cannot be filled')
    with (out/'train.jsonl').open('w',encoding='utf-8') as raw,(out/'train_tokens.jsonl').open('w') as tok:
        for r in selected:
            raw.write(json.dumps(r,ensure_ascii=False)+'\n');tok.write(json.dumps(token_by_id[r['input_sha256']])+'\n')
    for name in ['val.jsonl','val_tokens.jsonl']:shutil.copyfile(folder/name,out/name)
    assert not used&{r['input_sha256'] for r in selected}
    report={'records':len(selected),'excluded_pilot_examples':len(used),'source_counts':dict(Counter(r['source'] for r in selected)),
        'topic_proxy_count':len({r['selection_tags']['topic_proxy'] for r in selected}),
        'length_bins':dict(Counter(r['selection_tags']['length_bin'] for r in selected)),
        'skills_proxy_counts':dict(Counter(s for r in selected for s in r['selection_tags']['skills_proxy'])),
        'lexical_project_overlap':sum(r['selection_tags']['direct_category_lexical_match'] for r in selected),
        'approximate_near_duplicates_removed':near_removed,'selected_tokens':sum(len(token_by_id[r['input_sha256']]['input_ids']) for r in selected),
        'limitations':'Topic/skill/relevance labels are text proxies, not verified specialties. Three minhash signatures approximate duplicate screening; paraphrases can remain.',
        'policy':'Prepared only. Additional D3 training remains conditional on pipeline transfer results.'}
    (out/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report)


if __name__=='__main__':main()
