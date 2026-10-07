"""Run one bounded child job and account for its complete wall time."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.training.budget import SharedBudget
p=argparse.ArgumentParser();p.add_argument('stage',choices=['d3','d4','assessment']);p.add_argument('minutes',type=float)
p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
with SharedBudget(a.stage,a.minutes*60) as budget:
    command=a.command
    if command and command[0]=='--':command=command[1:]
    subprocess.run([sys.executable,*command],cwd=ROOT,check=True,timeout=max(1,budget.remaining()-30))
