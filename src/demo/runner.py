"""Shared case workflow for the command line and browser form."""
import json,os,subprocess,time
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
from src.data import DiReCTDataset
from src.demo.case_io import parse_case,render_report
from src.demo.graphviz_renderer import render_graph
from src.pipeline.client import LocalClient
from src.pipeline.pipeline import DiagnosticPipeline

ROOT=Path(__file__).resolve().parents[2]


def selected_model(choice='selected'):
    if choice=='selected':
        path=ROOT/'data'/'models'/'selected_model.json'
        choice=json.loads(path.read_text())['selected'] if path.exists() else 'base'
    if choice not in ('base','d4'):raise ValueError('Model must be base or d4')
    return choice


@contextmanager
def model_lock():
    import psutil
    path=ROOT/'data'/'runtime'/'manual.lock';path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(2):
        try:
            descriptor=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY);break
        except FileExistsError:
            try:owner=json.loads(path.read_text())
            except (ValueError,OSError):raise RuntimeError('Another case process is acquiring the model. Retry shortly.')
            if psutil.pid_exists(owner['pid']):raise RuntimeError('Another case is running. Wait for it to finish.')
            if attempt:raise RuntimeError('Could not acquire the model')
            path.unlink(missing_ok=True)
    try:
        os.write(descriptor,json.dumps({'pid':os.getpid()}).encode())
        ledger=ROOT/'data'/'training'/'budget.json'
        if ledger.exists() and json.loads(ledger.read_text()).get('active'):
            raise RuntimeError('The model is busy with training or evaluation. Wait for that run to finish.')
        yield
    finally:
        os.close(descriptor);path.unlink(missing_ok=True)


def run_case(data,output=None,choice='selected',port=8081,progress=None):
    started=time.perf_counter()
    inputs=parse_case(data);choice=selected_model(choice)
    output=Path(output) if output else ROOT/'data'/'manual_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    # A case folder must be new, so old patient records cannot be overwritten.
    if output.exists():raise ValueError('Output folder already exists')
    with model_lock():
        client=LocalClient(port,progress=progress);client.request('/health');adapters=client.request('/lora-adapters')
        if choice=='d4':
            expected=(ROOT/'data'/'models'/'d4_best.gguf').resolve()
            if len(adapters)!=1 or Path(adapters[0]['path']).resolve()!=expected:
                raise ValueError('Start the model with the D4 adapter loaded before selecting D4')
        if adapters:client.request('/lora-adapters',[{'id':a['id'],'scale':1. if choice=='d4' else 0.} for a in adapters])
        dataset=DiReCTDataset(ROOT/'mimic-iv-ext-direct-1.0.0')
        result=DiagnosticPipeline(client,dataset.knowledge_graphs,dataset.categories).run(inputs)
        result.update(case_id=data.get('case_id','manual_case'),model_choice=choice,
            adapter_configuration=client.request('/lora-adapters'))
        from src.demo.education import explain,teaching_adapter
        education_start=len(client.calls)
        with teaching_adapter(client):result['education']=explain(result,client,progress)
        result['education_calls']=client.calls[education_start:]
    result['source_inputs']=data
    if progress:progress('Rendering pathway and saving report')
    output.mkdir(parents=True,exist_ok=False)
    (output/'case.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    graph=False
    try:
        result['graph_files']=render_graph(result,output);graph=True
    except (RuntimeError,ValueError,OSError,subprocess.SubprocessError) as exc:
        result.setdefault('warnings',[]).append(f'Graph export unavailable: {exc}')
    result['total_seconds']=time.perf_counter()-started
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'report.html').write_text(render_report(result,inputs,graph=graph),encoding='utf-8')
    return result,output
