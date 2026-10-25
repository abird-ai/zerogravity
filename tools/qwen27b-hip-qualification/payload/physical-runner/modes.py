"""Exact derived source union bytes AND modes; no source mutation.

Successor to the immutable 1650-file check: the native authority is 1650 frozen
paths and the natural tool is the disjoint 11-file package under its repo-relative
prefix. The union count is derived from the assembled manifest and asserted, not
hardcoded to 1661; runtime-nine authority remains separately identified by the
natural-authority package.
"""
import stat
from pins import parsed,pin,require,relative

NATURAL_PREFIX = 'tools/qwen27b-natural-full-window-reference/'

def check_source(base):
 view,_=parsed(base/'assembled-source.json')
 files=view['files']
 native=[k for k in files if not k.startswith(NATURAL_PREFIX)]
 natural=[k for k in files if k.startswith(NATURAL_PREFIX)]
 require(view['source_count']==len(files)==view['native_source_count']+view['natural_source_count']
         and len(native)==view['native_source_count'] and len(natural)==view['natural_source_count']
         and len(natural)==11 and view['natural_source_count']==11,
         'derived native+natural source union view')
 for name,expected in files.items():
  p=base/'source'/relative(name)
  require(expected['kind']=='file' and pin(p)=={k:expected[k] for k in ('bytes','sha256')} and stat.S_IMODE(p.stat(follow_symlinks=False).st_mode)==expected['mode'],'source bytes/mode changed: '+name)
