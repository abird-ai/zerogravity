"""One accepted original-natural CPU capture: receipt plus four exact artifacts.

The capture is produced by tools/qwen27b-natural-full-window-reference as a
separate accepted CPU run. This unit only holds that finite external authority;
it never produces, copies or fabricates a pin. While the authority says
AWAITING_ACCEPTED_ORIGINAL_CPU_CAPTURE every natural mode refuses before
transport/launch/build binding, and the teacher receipt is never a substitute.
"""
import json

from pins import Held, parsed, require

AUTHORITY = 'natural-authority.json'
AUTHORITY_SCHEMA = 'zg.natural-reference-authority.v1'
AWAITING = 'AWAITING_ACCEPTED_ORIGINAL_CPU_CAPTURE'
ACCEPTED = 'ACCEPTED_ORIGINAL_CPU_CAPTURE'
NATURAL_RECEIPT = 'accepted-original-natural-full-window-1.json'
NATURAL_SCHEMA = 'zg.actual-original27b-full-window-natural-reference.v1'
NATURAL_STATUS = 'ACTUAL_ORIGINAL27B_FULL_WINDOW_NATURAL_REFERENCE_ONLY'
NATURAL_IMAGE = 'sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe'
NATURAL_IMAGE_DIGEST = 'sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634'
NATURAL_SNAPSHOT = '98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e'
NATURAL_PINS = ('manifest', 'raw', 'invocation', 'image_inspect')
PIN_BASENAME = {'manifest': 'manifest.json', 'raw': 'raw.bin',
                'invocation': 'invocation.json', 'image_inspect': 'image-inspect.json'}
HELD_NAMES = (NATURAL_RECEIPT, 'manifest.json', 'raw.bin', 'invocation.json', 'image-inspect.json')
CAPS = {NATURAL_RECEIPT: 1024**2, 'manifest.json': 16*1024**2, 'raw.bin': 1024**3,
        'invocation.json': 16*1024**2, 'image-inspect.json': 1024**2}


def load_authority(root):
    """Finite external authority. Absent pins refuse here, before any FD/transport."""
    value, proof = parsed(root/AUTHORITY, limit=1024**2)
    require(value.get('schema') == AUTHORITY_SCHEMA
            and value.get('status') in (AWAITING, ACCEPTED),
            'finite natural reference authority')
    package = value.get('package')
    require(type(package) is dict and len(package) == 9
            and all(type(row) is dict and set(row) == {'bytes', 'sha256'}
                    and type(row['bytes']) is int and row['bytes'] > 0
                    and len(row['sha256']) == 64 and row['sha256'] == row['sha256'].lower()
                    for row in package.values()),
            'frozen nine-file natural tool package format authority')
    if value['status'] != ACCEPTED:
        require(value.get('receipt') is None and value.get('artifacts') is None,
                'awaiting authority carries no receipt or artifact pins')
        raise ValueError('accepted original natural capture pins absent: run the frozen '
                         'natural tool, then validate --receipt (no teacher substitute)')
    require(type(value['receipt']) is dict and set(value['receipt']) == {'bytes', 'sha256'}
            and type(value['artifacts']) is dict and set(value['artifacts']) == set(NATURAL_PINS),
            'accepted authority has one receipt pin and its four exact artifact pins')
    return value, proof


def validate_receipt(value, package):
    require(value['schema'] == NATURAL_SCHEMA and value['status'] == NATURAL_STATUS
            and value['independent_original_natural_trajectory'] is True
            and value['image_id'] == NATURAL_IMAGE and value['image_digest'] == NATURAL_IMAGE_DIGEST
            and value['snapshot'] == NATURAL_SNAPSHOT
            and value['records'] == 1040 and value['raw_bytes'] == 934776928
            and value['text_roles'] == 851 and value['layers'] == 64 and value['cache_domains'] == 112
            and value['full_hf_conv_width'] == 4 and value['calls'] == 2
            and value['original_repetitions'] == 2 and value['seed'] == 11
            and value['parameter_mode'] == 'raw-source-assign-no-cast'
            and value['original_source_and_owned_lifetime_closed'] is True
            and value['rust_numerical_parity_qualified'] is False
            and value['mkl_exp_producer_qualified'] is False
            and value['state_root_native_serving_performance_qualified'] is False,
            'natural original producer scope/identity')
    scope = value['scope']
    require(type(scope) is dict and len(scope) == 10
            and scope['natural_feedback_qualified'] is True
            and scope['teacher_forcing_only'] is False
            and scope['all_851_text_source_roles'] is True and scope['full_hf_conv4'] is True
            and scope['mixed_source_parameters_preserved'] is True
            and scope['original_full_text_forward'] is True
            and scope['all_bf16_loader_equivalence_qualified'] is False
            and scope['mkl_exp_producer_qualified'] is False
            and scope['rust_numerical_parity_qualified'] is False
            and scope['state_root_native_serving_performance_qualified'] is False,
            'natural feedback qualified, teacher forcing refused, no parity claim')
    inputs = value['natural_inputs']
    require(type(inputs) is list and len(inputs) == 2
            and all(type(row) is list and len(row) == 2 and row[0] == 11
                    and type(row[1]) is int and 0 <= row[1] < 248320 for row in inputs),
            'two natural cohorts start at seed11 with an original call0 argmax next input')
    # Repeat determinism: both cohorts declare the same call1 feedback input. The
    # Rust side additionally decodes each held call0 fullV record and requires the
    # declared input to equal that independently decoded argmax; this owner only
    # binds the finite external declaration and never decodes the corpus.
    require(inputs[0][1] == inputs[1][1],
            'both natural cohorts declare the same call1 feedback input')
    require(type(value['package']) is dict and value['package'] == package,
            'receipt package is the same frozen nine-file natural authority')
    for name in NATURAL_PINS:
        pin = value[name]
        require(type(pin) is dict and set(pin) == {'path', 'bytes', 'sha256'}
                and type(pin['bytes']) is int and 0 < pin['bytes']
                and pin['path'].rsplit('/', 1)[-1] == PIN_BASENAME[name]
                and len(pin['sha256']) == 64 and pin['sha256'] == pin['sha256'].lower(),
                'declared natural pin '+name)
    return inputs


class NaturalReferenceLease:
    """Five distinct read-only FDs (receipt + four artifacts); no copy, no producer run."""

    def __init__(self, root):
        self.root = root
        self.held = []
        self.receipt = None
        self.inputs = None
        self.authority = None
        self.authority_proof = None
        try:
            require(root.is_absolute() and root.is_dir()
                    and root.resolve(strict=True) == root, 'exact fresh natural corpus root, no alias')
            self.authority, self.authority_proof = load_authority(root.parent)
            names = sorted(p.name for p in root.iterdir())
            require(names == sorted(HELD_NAMES), 'exactly the five natural corpus files: '+repr(names))
            self.receipt = Held(root/NATURAL_RECEIPT, self.authority['receipt'],
                                limit=CAPS[NATURAL_RECEIPT])
            value = json.load(self.receipt.file)
            self.receipt.identity()
            self.inputs = validate_receipt(value, self.authority['package'])
            for name in NATURAL_PINS:
                held_name = PIN_BASENAME[name]
                self.held.append(Held(root/held_name, self.authority['artifacts'][name],
                                      limit=CAPS[held_name]))
            sizes = {item.path.name: item.proof['bytes'] for item in self.held}
            require(sizes['raw.bin'] == value['raw_bytes'] == 934776928
                    and sizes['manifest.json'] == value['manifest']['bytes'],
                    'held natural raw/manifest extents equal the receipt')
        except BaseException:
            self.close()
            raise

    def external_receipt_pin(self):
        """Extent/SHA supplied by the independently accepted launch authority."""
        require(self.authority is not None and self.authority['status'] == ACCEPTED
                and self.receipt is not None, 'accepted natural authority receipt')
        pin = self.authority['receipt']
        require(type(pin) is dict and set(pin) == {'bytes', 'sha256'}
                and type(pin['bytes']) is int and 0 < pin['bytes'] <= CAPS[NATURAL_RECEIPT]
                and len(pin['sha256']) == 64 and pin['sha256'] == pin['sha256'].lower(),
                'external natural receipt extent/SHA')
        return pin['bytes'], pin['sha256']

    def bindings(self):
        """Receipt first, then the four artifacts the validator actually read."""
        return [(self.receipt.file.fileno(), str(self.receipt.path))] + [
            (item.file.fileno(), str(item.path)) for item in self.held]

    def after(self):
        """Exactly once, before close: verify the five held FDs, then close them."""
        try:
            for item in [self.receipt] + self.held:
                item.after()
        finally:
            self.close()

    def close(self):
        for item in self.held + ([self.receipt] if self.receipt is not None else []):
            item.close()
