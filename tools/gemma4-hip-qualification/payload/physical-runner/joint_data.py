"""Original70 held source FDs; two genuine own cache identities and exact lock seams."""
import json
from pathlib import Path
from pins import Held, require
import qwen_data as q
import gemma_data as g
from gemma_data import enclosed_build

Q_CACHE = Path('/home/pvl/.cache/abird/zerogravity/qualification/qwen27b-hip-main400-v1-p1-source469-20261024-1/xdg-cache/abird/zerogravity/models')
G_CACHE = Path('/home/pvl/.cache/abird/zerogravity/qualification/gemma4-hip-main314-v5-source466-20261024-1/xdg-cache/abird/zerogravity/models')
PARENTS = (
 (q, Q_CACHE, 5849, Path('/home/pvl/.local/state/abird/zerogravity/qwen27b-hip-main400-v1-p1-source469-20261024-1/import-1/invocation.json'), {'bytes':10989,'sha256':'a408f2e7b060bc4bb85d000d4a534fee405a06ad1071339ca197419a6e136400'}),
 (g, G_CACHE, 1547, Path('/home/pvl/.local/state/abird/zerogravity/gemma4-hip-main314-v5-source466-20261024-1/import-1/invocation.json'), {'bytes':9769,'sha256':'51acd1afe8680ac7748c304b84d875e188e9b17867d8cc3245be3843689c7c25'}),
)

Q_IMPORT_BASE = Q_CACHE.parents[3]
Q_IMPORT_SOURCE = {'bytes':509059,'sha256':'5af864e6bb1f368a3420901bad0f3c25c69f742dfd918db0eb2f43d8f0f4c4dc'}
Q_IMPORT_FILES = {'runner/run.py': {'bytes': 21250, 'sha256': 'e69b224c48a690d84880e05ad586089dade4c373ace571cae524b96ba59641d4'}, 'source/crates/zerogravity/examples/qwen27b_hip_final_owner.rs': {'bytes': 10579, 'sha256': '63195c791808e42951b1e371266bc27ed4ccd992d362999b63ab1f0782a75a88'}, 'source/crates/zerogravity/examples/qwen27b_hip_matrix/cache_import.rs': {'bytes': 2925, 'sha256': '8c819aba1b221b93b3676729124749425bc3655b3dd1025d1ff5d59c3ebbeb2d'}}

# One loader operand: the exact read-only FD mount of the held build artifact.
HELD_BINARY = '/tmp/zg-hip-joint-held-binary'

def original_import(value, module, canonical):
    require(value['mode']=='import' and value['status']=='CLOSED_PASS_SCOPED'
            and value['observed_exit']==0 and value['errors']==[]
            and value['source_after']=='UNCHANGED' and value['data_after']=='UNCHANGED_HELD_FULL_SHA'
            and value['semantic']['canonical_snapshot']==module.SNAPSHOT
            and value['semantic']['new_download'] is False and value['semantic']['gpu_model_called'] is False
            and value['data_closure']==module.DATA_PROOF and value['held_data_files']==35,
            'same genuine original cache import')
    if module is q:
        # Q receipt records its source/cwd/argv, not Gemma's canonical-cache field.
        # Held original controller+importer prove its XDG root; never invent a field.
        require(canonical==Q_IMPORT_BASE/'xdg-cache/abird/zerogravity/models'
                and value['source_closure']==Q_IMPORT_SOURCE
                and value['binary']['source_closure']==Q_IMPORT_SOURCE
                and value['cwd']==str(Q_IMPORT_BASE/'source')
                and value['binary']['path']==str(Q_IMPORT_BASE/'target/release/examples/qwen27b_hip_final_owner')
                and value['read_only_original_data_root']==str(q.DATA_ROOT)
                and value['original_received_proof']==q.RECEIVED_PROOF
                and value['network_payload_transfer'] is False
                and value['semantic']['read_only_reused_data_root']==str(q.DATA_ROOT)
                and value['semantic']['network_payload_transfer'] is False
                and value['argv'][-2:]==['--import-source',str(q.DATA_ROOT/'source')],
                'exact original Qwen import namespace/source/controller-derived canonical root')
    else:
        require(module is g and value['new_canonical_cache']==str(canonical)
                and value['original_read_only_FD_bindings'] is True,
                'exact original Gemma explicit canonical cache field')
    c=value['foreground_cleanup']
    require(c['leader_reaped'] is True and c['group_absent'] is True
            and c['kill_attempts_before_reap']==2 and c['leader_reap_attempts']==1
            and c['all_nonzero_signals_before_reap'] is True and c['stage_errors']==[]
            and value['log']['eof'] is True and value['log']['closed'] is True,
            'original import group/reap/EOF closure')

class CacheLease:
    """Cheap authorities BEFORE opening/hash-scanning any large original data."""
    def __init__(self):
        self.held=[]; self.imports=[]
        try:
            for module,canonical,size,path,proof in PARENTS:
                require(canonical.resolve(strict=True)==canonical
                        and (canonical/'locks').resolve(strict=True)==canonical/'locks',
                        'exact original own cache and locks, no aliases')
                receipt=Held(path,proof,limit=1024**2); self.held.append(receipt)
                value=json.load(receipt.file);original_import(value,module,canonical)
                self.imports.append({'family':'qwen' if module is q else 'gemma','proof':proof,'canonical':str(canonical)})
                manifest=Held(canonical/'snapshots'/module.SNAPSHOT/'manifest.json',
                              {'bytes':size,'sha256':module.SNAPSHOT},limit=32768)
                self.held.append(manifest)
            authority=Held(Q_IMPORT_BASE/'closure.json',Q_IMPORT_SOURCE,limit=1024**2)
            self.held.append(authority);closure=json.load(authority.file)
            for name,proof in Q_IMPORT_FILES.items():
                require(closure['records'][name]==proof,'same original Qwen import controller/source authority')
                self.held.append(Held(Q_IMPORT_BASE/name,proof,limit=1024**2))
        except BaseException:
            self.close();raise
    def after(self):
        for item in self.held:item.after()
    def close(self):
        for item in self.held:item.close()

class JointDataLease:
    def __init__(self, base):
        self.leases=[]; self.cache=[]; self.held=[]; self.records={}; self.cache_authority=None
        try:
            self.cache_authority=CacheLease()
            self.cache=self.cache_authority.held
            for name,module in [('qwen',q),('gemma',g)]:
                lease=module.DataLease(base/name,module.DATA_ROOT)
                self.leases.append(lease);self.held.extend(lease.held)
                self.records.update({name+'/'+key:proof for key,proof in lease.records.items()})
            require(len(self.held)==70 and len({v.before[:2] for v in self.held})==70,
                    'all70 distinct original source/reference FDs held')
        except BaseException:
            self.close();raise
    def bindings(self):
        return [binding for lease in self.leases for binding in lease.bindings()]
    def authority_after(self):
        for lease in self.leases: lease.authority_after()
    def close(self):
        for lease in self.leases: lease.close()
        for item in self.cache: item.close()

def enclosed(base,source,binary,data,command,gpu,output=None):
    require(gpu and output is not None and source==base/'source'
            and output.parent==Path('/home/pvl/.local/state/abird/zerogravity')/base.name,
            'one fresh joint physical namespace')
    # Mandatory original user/net wrapper surrounds this exact read-only host/PID route.
    argv=['/run/current-system/sw/bin/bwrap','--unshare-pid','--die-with-parent',
          '--ro-bind','/','/','--bind',str(base),str(base),'--bind',str(output),str(output),
          '--dev-bind','/dev','/dev']
    for _,canonical,_,_,_ in PARENTS:
        argv+=['--bind',str(canonical/'locks'),str(canonical/'locks')]
    argv+=['--proc','/proc','--tmpfs','/tmp','--ro-bind-fd',str(binary.file.fileno()),HELD_BINARY]
    for fd,path in data.bindings(): argv+=['--ro-bind-fd',str(fd),path]
    return argv+['--chdir',str(source),'--',*command]
