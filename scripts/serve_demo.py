"""Serve the clinical input form on this computer only."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.demo.server import DemoServer
from src.demo.graphviz_renderer import dot_executable


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);p.add_argument('--model-port',type=int,default=8081)
    a=p.parse_args();dot_executable()
    server=DemoServer(('127.0.0.1',a.port),a.model_port)
    print(f'Clinical input form: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
