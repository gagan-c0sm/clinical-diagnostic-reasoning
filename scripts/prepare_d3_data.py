"""Prepare source-aware, question-deduplicated supplementary splits."""
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def fingerprint(text):
    return hashlib.sha256(' '.join(text.split()).casefold().encode()).hexdigest()


def main():
    output=ROOT/'data'/'d3'
    if output.exists(): raise FileExistsError('Preserve existing D3 splits; use a new output version')
    records=[]
    for path in sorted((ROOT/'data'/'supplementary').rglob('*.jsonl')):
        for line in path.open(encoding='utf-8'):
            r=json.loads(line)
            key=fingerprint(r['input'])
            split=r['source_split']
            if 'medical-o1' in r['source']:
                split='val' if int(key[:8],16)%100<5 else 'train'
            split='val' if split in ('val','validation') else split
            r.update(split=split,input_sha256=key)
            records.append(r)
    held_articles={r.get('pmcid') for r in records if r['split']!='train' and r.get('pmcid')}
    groups=defaultdict(list)
    for r in records: groups[r['input_sha256']].append(r)
    stats=Counter(); kept=defaultdict(list)
    for key,items in sorted(groups.items()):
        # Held-out questions take precedence over training copies.
        priority={'test':0,'val':1,'train':2}
        items.sort(key=lambda r:(priority[r['split']],r['source'],r['source_row']))
        winner=items[0]
        stats['duplicate_question_rows_removed']+=len(items)-1
        if winner['split']=='train' and winner.get('pmcid') in held_articles:
            stats['train_rows_removed_for_heldout_article']+=1
            continue
        if winner['split']=='train' and len({fingerprint(r['output']) for r in items})>1:
            stats['conflicting_train_questions_removed']+=1
            continue
        kept[winner['split']].append(winner)
    output.mkdir(parents=True)
    report={'seed_policy':'Medical-O1 validation: normalized-question SHA256 modulo100 <5',
        'dedup_policy':'Exact normalized input; held-out priority test then val then train; remove conflicting train targets',
        'input_rows':len(records),'excluded':dict(stats),'splits':{}}
    for split,rows in kept.items():
        with (output/f'{split}.jsonl').open('w',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        report['splits'][split]={'records':len(rows),'sources':dict(Counter(r['source'] for r in rows))}
    sets=[{r['input_sha256'] for r in kept[k]} for k in ['train','val','test']]
    assert not any(sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)])
    (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
