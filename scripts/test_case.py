"""Diagnose one local JSON case and export JSON plus a readable HTML report."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.demo.runner import run_case


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('case',type=Path)
    parser.add_argument('--model',choices=['selected','base','d4'],default='selected')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--port',type=int,default=8081)
    args=parser.parse_args()
    data=json.loads(args.case.read_text(encoding='utf-8-sig'))
    result,output=run_case(data,args.output,args.model,args.port)
    print(f"{result['status']}: {result['diagnosis']}\nReport: {output/'report.html'}",flush=True)
    if result['status']=='error':raise SystemExit(1)


if __name__=='__main__':main()
