#!/usr/bin/env python3
"""Build a standalone HTML flowchart from an editable graph JSON and template."""
import argparse,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--graph',type=Path,default=ROOT/'figures/source/graph_data.json')
p.add_argument('--output',type=Path,required=True,help='A new HTML file; existing paths are not overwritten.')
a=p.parse_args();data=json.loads(a.graph.read_text());assert isinstance(data.get('graphs'),list)
s=(HERE/'index.template.html').read_text();assert s.count('__DATA__')==1
with a.output.open('x') as f:f.write(s.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')))
print('Created',a.output)
