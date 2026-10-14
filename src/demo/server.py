"""Loopback-only input form and background case execution; standard library HTTP."""
import json,re,secrets,threading,uuid,inspect
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from src.demo.case_io import parse_case
from src.demo.runner import ROOT,run_case,selected_model
from src.pipeline.client import LocalClient

ASSETS=Path(__file__).with_name('web')
ARTIFACTS={'report.html':'text/html; charset=utf-8','result.json':'application/json',
    'case.json':'application/json','graph.svg':'image/svg+xml','graph.png':'image/png','graph.dot':'text/plain; charset=utf-8',
    'graph.html':'text/html; charset=utf-8','graph.js':'text/javascript; charset=utf-8','graph.css':'text/css; charset=utf-8'}


class DemoServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,model_port=8081,runner=run_case):
        if address[0]!='127.0.0.1':raise ValueError('The case form must bind to IPv4 loopback')
        super().__init__(address,Handler)
        self.model_port=model_port;self.runner=runner;self.csrf=secrets.token_urlsafe(32)
        self.jobs={};self.case_lock=threading.Lock()

    def execute(self,job_id,data):
        try:
            self.jobs[job_id]={'state':'running'}
            kwargs={'port':self.model_port}
            if 'progress' in inspect.signature(self.runner).parameters:
                kwargs['progress']=lambda stage:self.jobs[job_id].update(stage=stage)
            result,folder=self.runner(data,ROOT/'data'/'manual_runs'/job_id,**kwargs)
            self.jobs[job_id]={'state':'done','status':result['status'],'diagnosis':result.get('diagnosis'),
                'category':result.get('category'),'seconds':result.get('total_seconds',result.get('seconds',0)),
                'report':f'/runs/{job_id}/report.html','graph':bool(result.get('graph_files'))}
        except Exception as exc:
            self.jobs[job_id]={'state':'error','error':str(exc)}
        finally:self.case_lock.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self,format,*args):pass  # Never log submitted clinical text.

    def reply(self,code,body,kind='application/json'):
        if kind=='application/json' and not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode()
        elif isinstance(body,str):body=body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self'; connect-src 'self'; frame-src 'self'; frame-ancestors 'self'; form-action 'self'; base-uri 'none'")
        self.end_headers();self.wfile.write(body)

    def valid_host(self):
        return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')

    def do_GET(self):
        if not self.valid_host():return self.reply(403,{'error':'Invalid local host'})
        path=urlsplit(self.path).path
        if path in ('/','/app.js','/style.css'):
            name,kind={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),
                '/style.css':('style.css','text/css; charset=utf-8')}[path]
            return self.reply(200,(ASSETS/name).read_bytes(),kind)
        if path=='/api/state':
            choice=selected_model();detail=''
            try:
                client=LocalClient(self.server.model_port,timeout=2);client.request('/health')
                adapters=client.request('/lora-adapters')
                expected=(ROOT/'data'/'models'/'d4_best.gguf').resolve()
                ready=(len(adapters)==1 and Path(adapters[0]['path']).resolve()==expected and adapters[0]['scale']==1.) if choice=='d4' else all(a['scale']==0. for a in adapters)
                if not ready:detail='Selected model is not active. Restart the demo launcher.'
            except Exception:ready=False;detail='Local model offline'
            return self.reply(200,{'ready':ready,'detail':detail,'model':choice,'token':self.server.csrf,
                'busy':self.server.case_lock.locked()})
        if path=='/api/example':
            return self.reply(200,json.loads((ROOT/'examples'/'direct_dev_case.json').read_text()))
        if path=='/api/examples':
            from src.demo.library import catalog
            return self.reply(200,{'cases':catalog()})
        match=re.fullmatch(r'/api/examples/([a-f0-9]{16})',path)
        if match:
            from src.demo.library import case_for
            try:return self.reply(200,case_for(match[1]))
            except KeyError:return self.reply(404,{'error':'Example not found'})
        match=re.fullmatch(r'/api/jobs/([a-f0-9]{32})',path)
        if match:
            job=self.server.jobs.get(match[1])
            return self.reply(200,job) if job else self.reply(404,{'error':'Case not found'})
        match=re.fullmatch(r'/runs/([a-f0-9]{32})/([a-z.]+)',path)
        if match and match[2] in ARTIFACTS:
            root=(ROOT/'data'/'manual_runs').resolve()
            file=(root/match[1]/match[2]).resolve()
            if file.is_relative_to(root) and file.is_file():return self.reply(200,file.read_bytes(),ARTIFACTS[match[2]])
        return self.reply(404,{'error':'Not found'})

    def do_POST(self):
        if not self.valid_host():return self.reply(403,{'error':'Invalid local host'})
        origin=self.headers.get('Origin')
        allowed={f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
        if (origin and origin not in allowed) or not secrets.compare_digest(self.headers.get('X-Case-Token',''),self.server.csrf):
            return self.reply(403,{'error':'Reload the local form before submitting'})
        replay=re.fullmatch(r'/api/examples/([a-f0-9]{16})/replay',urlsplit(self.path).path)
        if replay:
            from src.demo.library import saved_for,case_for
            from src.demo.education import explain
            from src.demo.graphviz_renderer import render_graph
            from src.demo.case_io import render_report
            try:
                key=replay[1];result=saved_for(key);case=case_for(key)
                cached=ROOT/'data/demo_education'/f'{key}.json'
                result['education']=json.loads(cached.read_text(encoding='utf-8')) if cached.exists() else explain(result)
                result['source_inputs']=case
                job_id=uuid.uuid4().hex;folder=ROOT/'data/manual_runs'/job_id
                result['graph_files']=render_graph(result,folder)
                for name,value in [('result',result),('case',case)]:
                    (folder/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
                (folder/'report.html').write_text(render_report(result,parse_case(case),graph=True),encoding='utf-8')
                return self.reply(200,{'id':job_id,'state':'done','status':result['status'],'diagnosis':result.get('diagnosis'),
                    'category':result.get('category'),'seconds':result.get('seconds',0),'graph':True,'replay':True,
                    'report':f'/runs/{job_id}/report.html'})
            except KeyError:return self.reply(404,{'error':'Example not found'})
            except (ValueError,RuntimeError,OSError) as exc:return self.reply(400,{'error':str(exc)})
        if urlsplit(self.path).path!='/api/cases':return self.reply(404,{'error':'Not found'})
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':
            return self.reply(415,{'error':'Submit a JSON case'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=1000000:return self.reply(413,{'error':'Case is too large or empty'})
            data=json.loads(self.rfile.read(length));parse_case(data)
        except (ValueError,UnicodeError) as exc:return self.reply(400,{'error':str(exc)})
        if not self.server.case_lock.acquire(blocking=False):return self.reply(409,{'error':'Another case is running. Wait for it to finish.'})
        job_id=uuid.uuid4().hex;self.server.jobs[job_id]={'state':'queued'}
        threading.Thread(target=self.server.execute,args=(job_id,data),daemon=True).start()
        return self.reply(202,{'id':job_id})
