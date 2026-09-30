"""Persistent single-job wall-time accounting across D3/D4 and resumes."""
import json
import os
import time
import threading
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
LEDGER=ROOT/'data'/'training'/'budget.json'


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)


def initialize(path=LEDGER):
    if path.exists():return
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(path,{'total_seconds':14400,'charged_historical_seconds':2400,
        'stage_caps_seconds':{'d3':3600,'d4':6000,'assessment':2400},
        'sessions':[],'active':None,
        'historical_note':'Conservative 40-minute charge includes completed 26.5-minute pilot and short trials.'})


class SharedBudget:
    def __init__(self,stage,max_seconds,path=LEDGER):
        self.stage=stage;self.requested=max_seconds;self.path=Path(path)
        self.lock=self.path.with_suffix('.lock')
        self.guard=threading.Lock();self.finished=False
    def __enter__(self):
        initialize(self.path)
        try:self.fd=os.open(self.lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:raise RuntimeError('Another budgeted job or unrecovered lock exists; inspect budget ledger')
        try:
            self.ledger=json.loads(self.path.read_text(encoding='utf-8'))
            if self.ledger['active']:raise RuntimeError('Recover interrupted budget session before continuing')
            sessions=self.ledger['sessions']
            used=sum(s['seconds'] for s in sessions)
            stage_used=sum(s['seconds'] for s in sessions if s['stage']==self.stage)
            self.limit=min(self.requested,self.ledger['total_seconds']-self.ledger['charged_historical_seconds']-used,
                self.ledger['stage_caps_seconds'][self.stage]-stage_used)
            if self.limit<180:raise RuntimeError('Insufficient time budget, including finalization reserve')
            self.start=time.monotonic()
            self.ledger['active']={'stage':self.stage,'pid':os.getpid(),'utc_started':datetime.now(timezone.utc).isoformat(),
                'limit_seconds':self.limit}
            atomic_json(self.path,self.ledger)
            self.timer=threading.Timer(self.limit,self._expire)
            self.timer.daemon=True;self.timer.start()
            return self
        except BaseException:
            os.close(self.fd);self.lock.unlink();raise
    def remaining(self):return self.limit-(time.monotonic()-self.start)
    def __exit__(self,kind,value,trace):
        self.timer.cancel()
        self._finish('failed' if kind else 'finished')
    def _finish(self,outcome):
        with self.guard:
            if self.finished:return
            self.finished=True
            elapsed=time.monotonic()-self.start
            self.ledger['sessions'].append({'stage':self.stage,'seconds':elapsed,
                'utc_finished':datetime.now(timezone.utc).isoformat(),'outcome':outcome})
            self.ledger['active']=None
            try:atomic_json(self.path,self.ledger)
            finally:os.close(self.fd);self.lock.unlink(missing_ok=True)
    def _expire(self):
        # Normal training stops with a 3-minute finalization reserve. This is a
        # backstop for a stalled CUDA call; last complete checkpoints survive.
        self._finish('hard_deadline')
        import psutil
        for child in psutil.Process().children(recursive=True):
            try:child.kill()
            except psutil.Error:pass
        os._exit(124)


def recover_interrupted(path=LEDGER):
    """Explicit recovery only after verifying the recorded process has exited."""
    import psutil
    ledger=json.loads(path.read_text(encoding='utf-8'));active=ledger['active']
    if not active:raise RuntimeError('No active session')
    if psutil.pid_exists(active['pid']):raise RuntimeError('Recorded process may still be running')
    elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(active['utc_started'])).total_seconds()
    ledger['sessions'].append({'stage':active['stage'],'seconds':max(0,elapsed),'outcome':'recovered_conservatively'})
    ledger['active']=None;atomic_json(path,ledger)
    path.with_suffix('.lock').unlink(missing_ok=True)
