"""Three stages, with deterministic assembly and source-quote validation.

The public run method accepts clinical input fields and KGs only: annotation
labels/trees never enter the generation path.
"""
import json
import re
import time
from .schema import validate_tree


def obj(properties):
    return {"type":"object", "properties":properties,
            "required":list(properties), "additionalProperties":False}


STRING = {"type":"string"}


def source_passages(text):
    """Keep original contiguous passages; never synthesize evidence text."""
    return list(dict.fromkeys(p.strip() for p in re.split(
        r"\n+|(?<=[.!?])\s+(?=[A-Z])", text) if p.strip()))


class DiagnosticPipeline:
    def __init__(self, client, knowledge_graphs, categories, routing='single'):
        if routing not in ('single', 'shortlist'):
            raise ValueError('Unknown routing mode')
        self.client = client
        self.kgs = knowledge_graphs
        self.categories = sorted(categories)
        self.routing = routing

    def run(self, inputs):
        start = time.perf_counter()
        call_start = len(self.client.calls)
        result = {"schema_version":1, "routing":self.routing, "category":None, "diagnosis":None,
                  "status":"abstained", "observations":[], "nodes":{},
                  "warnings":[], "trace":[]}
        try:
            self._run(inputs, result)
        except Exception as exc:
            result["status"] = "error"
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["seconds"] = time.perf_counter()-start
        result["calls"] = self.client.calls[call_start:]
        return result

    def _single_route(self, inputs, result):
        categories = self.categories + ["Uncertain"]
        choice = self.client.generate("classify",
            "Select the disease category best supported by the diagnostic findings in "
            "this note. The target can be a chronic condition being evaluated, not only "
            "an acute reason for admission. Consider the specific test results and "
            "disease-defining findings across all fields, not just the chief complaint. "
            "Prefer a supported specific disease over a nonspecific syndrome. "
            "Summarize the decisive evidence concisely before choosing. "
            "If no category is supported choose Uncertain.\n"
            + json.dumps(categories) + "\nClinical note:\n" + inputs.get_all_text(),
            obj({"evidence_summary":STRING, "category":{"type":"string", "enum":categories}}), 300)
        category = choice.get("category")
        if category not in categories:
            raise ValueError("Invalid category")
        result["category"] = category
        result["classification_evidence"] = choice.get("evidence_summary", "")
        result['initial_category'] = category
        result['category_candidates'] = [] if category == 'Uncertain' else [category]
        return category

    def _shortlist_route(self, inputs, result):
        response = self.client.generate('shortlist',
            'Identify up to three plausible disease categories supported by the note. '
            'Rank most likely first. Include plausible alternatives so the next step can '
            'compare their diagnostic criteria. The target may be a chronic condition '
            'being evaluated, not just the acute admission reason. Consider specific '
            'test results across all fields. Do not pad the list with unsupported '
            'categories. Return an empty list if none are supported.\n'
            f'Allowed categories: {json.dumps(self.categories)}\n'
            f'Clinical note:\n{inputs.get_all_text()}',
            obj({'evidence_summary':STRING, 'ranked_categories':{'type':'array',
                'maxItems':3, 'uniqueItems':True,
                'items':{'type':'string','enum':self.categories}}}), 400)
        candidates = response.get('ranked_categories')
        if not isinstance(candidates,list) or len(candidates)>3 or any(c not in self.categories for c in candidates):
            raise ValueError('Invalid category shortlist')
        candidates = list(dict.fromkeys(candidates))
        result['category_candidates'] = candidates
        result['initial_category'] = candidates[0] if candidates else 'Uncertain'
        result['classification_evidence'] = response.get('evidence_summary','')
        if not candidates:
            result['category'] = 'Uncertain'
            return 'Uncertain'
        knowledge = {}
        for category in candidates:
            kg = self.kgs.get(category)
            knowledge[category] = ({'milestones':kg.get_all_node_names(), 'criteria':kg.knowledge}
                if kg else {'criteria':None, 'note':'No supplied KG. Missing criteria are not evidence against this category.'})
        review = self.client.generate('compare_categories',
            'Compare the shortlisted categories against the original clinical note '
            'using the supplied diagnostic criteria. For each candidate state decisive '
            'supporting findings and missing or contradicting findings in at most two '
            'sentences. Missing tests are unknown, not negative. A nonspecific symptom '
            'or syndrome should not outweigh evidence for a more specific disease. '
            'Do not favor the first candidate merely because it is first. Do not '
            'reject a category solely because no KG is supplied. Choose the best '
            'supported category, or Uncertain if the evidence cannot resolve the '
            'choice. Do not invent findings.\n'
            f'Candidate knowledge: {json.dumps(knowledge,ensure_ascii=False)}\n'
            f'Clinical note:\n{inputs.get_all_text()}',
            obj({'assessments':{'type':'array','minItems':len(candidates),'maxItems':len(candidates),
                'items':obj({'category':{'type':'string','enum':candidates},'assessment':STRING})},
                'category':{'type':'string','enum':candidates+['Uncertain']}}), 1000)
        if (review.get('category') not in candidates+['Uncertain'] or
            {a.get('category') for a in review.get('assessments',[])} != set(candidates)):
            raise ValueError('Incomplete candidate comparison')
        result['routing_review'] = review
        result['category'] = review['category']
        return review['category']

    def _run(self, inputs, result):
        category = (self._shortlist_route(inputs,result) if self.routing == 'shortlist'
                    else self._single_route(inputs,result))
        if category == "Uncertain":
            return
        kg = self.kgs.get(category)
        if kg is None:
            result["status"] = "unsupported_kg"
            return

        criteria = json.dumps(kg.knowledge, ensure_ascii=False)
        observations = []
        for field in inputs.non_empty_fields():
            passages = source_passages(inputs.get_field(field))
            obs_schema = obj({"selected_ids":{"type":"array", "maxItems":16,
                "items":{"type":"integer", "enum":list(range(1,len(passages)+1))},
                "uniqueItems":True}})
            extracted = self.client.generate(f"extract_input{field}",
                f"Select the IDs of up to 16 source passages relevant to {category}. "
                "Include relevant negative findings and contradictory evidence. Select only "
                "passages with actual clinical information relevant to these criteria. "
                "Use an empty selected_ids array when no passages are relevant. "
                "Do not pad the list. Return each selected ID only once.\n"
                f"Criteria: {criteria}\nClinical field input{field} passages:\n"
                + json.dumps({i:p for i,p in enumerate(passages,1)}, ensure_ascii=False),
                obs_schema, 250)
            ids = extracted.get("selected_ids")
            if not isinstance(ids, list):
                raise ValueError("Missing passage selections")
            for passage_id in ids:
                if type(passage_id) is not int or not 1 <= passage_id <= len(passages):
                    result["warnings"].append(f"input{field}:rejected_invalid_passage_id")
                    continue
                text = passages[passage_id-1]
                if any(o["text"] == text and o["source_field"] == field for o in observations):
                    continue
                observations.append({"id":f"o{len(observations)+1}",
                                     "text":text, "source_field":field})
        result["observations"] = observations
        if not observations:
            result["warnings"].append("no_grounded_observations")
            return
        # Support root first; then choose among sibling branches, allowing a stop.
        current = None
        candidates = [kg.diagnostic_tree]
        path = []
        for _ in range(20):
            options = [n.name for n in candidates] + ["STOP"]
            schema = obj({"choice":{"type":"string", "enum":options},
                "evidence_ids":{"type":"array", "items":{"type":"string", "enum":[o["id"] for o in observations]}, "uniqueItems":True},
                "rationale":STRING})
            decision = self.client.generate("reason",
                "Select the best-supported next diagnostic milestone, or STOP if the "
                "criteria are not established or alternatives remain unresolved. "
                "Do not treat absent tests as negative results. Prioritize current findings "
                "over history. Provide evidence_ids and a concise explanation relating "
                "the evidence to the criterion. Evidence must support the selected step.\n"
                f"Current milestone: {current.name if current else 'none'}\n"
                f"Options and criteria: {json.dumps({n.name:kg.get_premise(n.name) for n in candidates})}\n"
                f"Observations: {json.dumps(observations)}", schema, 600)
            result["trace"].append(decision)
            if decision.get("choice") == "STOP":
                break
            selected = next((n for n in candidates if n.name == decision.get("choice")), None)
            ids = decision.get("evidence_ids", [])
            if (selected is None or not ids or not decision.get("rationale", "").strip()
                    or any(i not in {o['id'] for o in observations} for i in ids)):
                raise ValueError("Invalid or unsupported reasoning step")
            path.append((selected, decision))
            current = selected
            if current.is_leaf:
                break
            candidates = current.children
        if not path:
            return
        nodes = {}
        for index, (milestone, decision) in enumerate(path):
            d = f"d{index}"
            c = f"c{index}"
            nodes[d] = {"content":milestone.name, "type":"Intermedia",
                        "parent":f"d{index+1}" if index+1 < len(path) else None,
                        "children":([f"d{index-1}"] if index else []) + [c],
                        "kg_premise":kg.get_premise(milestone.name)}
            nodes[c] = {"content":decision["rationale"], "type":"Cause",
                        "parent":d, "children":[]}
            for j, oid in enumerate(dict.fromkeys(decision["evidence_ids"])):
                o = next(o for o in observations if o["id"] == oid)
                key = f"i{index}_{j}"
                nodes[key] = {"content":o["text"], "type":"Input", "parent":c,
                              "children":[], "source_field":o["source_field"]}
                nodes[c]["children"].append(key)
        errors = validate_tree(nodes, inputs)
        if errors:
            raise ValueError(f"Invalid tree: {errors}")
        result["nodes"] = nodes
        result["diagnosis"] = current.name
        result["status"] = "complete" if current.is_leaf else "partial"
