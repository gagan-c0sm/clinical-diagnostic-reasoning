import json,tempfile,threading,time,unittest,uuid
from pathlib import Path
from urllib.request import Request,urlopen,build_opener,ProxyHandler
from urllib.error import HTTPError
from src.demo.graphviz_renderer import to_dot,render_graph,dot_executable
from src.demo.server import DemoServer
from src.demo.runner import ROOT
from src.demo.interactive_graph import graph_data


class GraphTests(unittest.TestCase):
    def test_native_render_escapes_content_and_handles_empty_tree(self):
        try:dot_executable()
        except RuntimeError:self.skipTest('Portable Graphviz not installed')
        content='Literal "quote" <script> \\N\nSecond line'
        result={'status':'complete','nodes':{'bad"id':{'content':content,'type':'Intermedia','parent':None,'children':[]}}}
        dot=to_dot(result)
        self.assertNotIn('bad"id',dot)
        with tempfile.TemporaryDirectory(dir=ROOT/'data'/'test_tmp') as folder:
            files=render_graph(result,folder)
            self.assertEqual(set(files),{'dot','svg','png','html'})
            self.assertTrue((Path(folder)/'graph.png').read_bytes().startswith(b'\x89PNG'))
            svg=(Path(folder)/'graph.svg').read_text(encoding='utf-8')
            self.assertIn('&lt;script&gt;',svg)
            interactive=(Path(folder)/'graph.html').read_text(encoding='utf-8')
            self.assertIn('graph.js',interactive)
            self.assertNotIn('<script> ',interactive)
            self.assertTrue((Path(folder)/'graph.css').exists())
            render_graph({'status':'abstained','nodes':{}},folder)
            self.assertIn('No supported reasoning tree',(Path(folder)/'graph.svg').read_text())

    def test_rejects_invalid_graph(self):
        with self.assertRaises(ValueError):to_dot({'nodes':{'x':{'type':'Input','content':'a','parent':None,'children':[],'source_field':7}}})

    def test_playback_orders_evidence_before_rationale_and_diagnosis(self):
        nodes={
            'final':{'content':'Final','type':'Intermedia','parent':None,'children':['early','c2']},
            'early':{'content':'Suspected','type':'Intermedia','parent':'final','children':['c1']},
            'c1':{'content':'First rationale','type':'Cause','parent':'early','children':['i1']},
            'i1':{'content':'Finding one','type':'Input','parent':'c1','children':[],'source_field':1},
            'c2':{'content':'Second rationale','type':'Cause','parent':'final','children':['i2']},
            'i2':{'content':'Finding two','type':'Input','parent':'c2','children':[],'source_field':6}}
        data=graph_data({'nodes':nodes,'model_choice':'d4'})
        self.assertEqual([p['kind'] for p in data['phases']],['Evidence','Rationale','Milestone']*2)
        self.assertEqual([m['name'] for m in data['milestones']],['Suspected','Final'])
        self.assertEqual(data['nodes']['n3']['phase'],1)
        self.assertEqual(data['nodes']['n0']['phase'],6)


class FormServerTests(unittest.TestCase):
    def test_form_validation_origin_and_serial_execution(self):
        started=threading.Event();release=threading.Event()
        def runner(data,output,port):
            started.set();release.wait(4)
            return {'status':'abstained','diagnosis':None,'seconds':0},output
        server=DemoServer(('127.0.0.1',0),runner=runner)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        opener=build_opener(ProxyHandler({}))
        def request(path,data=None,token=None,origin=None):
            headers={'Content-Type':'application/json'}
            if token:headers['X-Case-Token']=token
            if origin:headers['Origin']=origin
            with opener.open(Request(base+path,data=json.dumps(data).encode() if data is not None else None,headers=headers)) as response:
                return response.status,json.load(response)
        try:
            with opener.open(base) as response:self.assertIn(b'case-form',response.read())
            with opener.open(base+'/app.js') as response:self.assertIn(b'/api/cases',response.read())
            artifact_dir=ROOT/'data'/'manual_runs'/uuid.uuid4().hex
            artifact_dir.mkdir(parents=True)
            try:
                for filename in ('result.json','case.json'):
                    payload={'text':'A saved finding','value':42}
                    (artifact_dir/filename).write_text(json.dumps(payload),encoding='utf-8')
                    code,received=request(f'/runs/{artifact_dir.name}/{filename}')
                    self.assertEqual(code,200)
                    self.assertEqual(received,payload)
            finally:
                for file in artifact_dir.iterdir():file.unlink()
                artifact_dir.rmdir()
            token=server.csrf
            for data,provided,origin,expected in [({'input1':'finding'},None,None,403),
                ({'input1':'finding'},token,'http://example.com',403),({},token,None,400),
                ({'input1':'finding','diagnosis':'answer'},token,None,400)]:
                with self.assertRaises(HTTPError) as raised:request('/api/cases',data,provided,origin)
                self.assertEqual(raised.exception.code,expected)
            code,job=request('/api/cases',{'input1':'finding'},token)
            self.assertEqual(code,202);self.assertTrue(started.wait(2))
            with self.assertRaises(HTTPError) as raised:request('/api/cases',{'input1':'other'},token)
            self.assertEqual(raised.exception.code,409)
            release.set()
            for _ in range(50):
                _,state=request('/api/jobs/'+job['id'])
                if state['state']=='done':break
                time.sleep(.01)
            self.assertEqual(state['status'],'abstained')
            with self.assertRaises(HTTPError):request('/runs/'+job['id']+'/../../budget.json')
        finally:release.set();server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
