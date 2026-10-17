"""Synthetic loopback only; independent Azure wire signature verifier (stdlib)."""
import base64
import datetime
import email.utils
import hashlib
import hmac
import socket
import sys
import urllib.parse

requested_mode = sys.argv[1]
content = requested_mode.startswith('content-')
mode = requested_mode.removeprefix('content-')
assert mode in ('public', 'bearer', 'key', 'sas')
key = bytes(range(64))
account = 'fixtureaccount'
version = '2025-10-16T12:00:00.0000000Z'
raw_sas = 'sv=2023-11-03&sr=bv&sp=r&se=2030-01-01T00%3a00%3A00Z&spr=https&sig=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA%3d'
if content:
    raw_sas = raw_sas.replace('sr=bv', 'sr=b')
body = b'actual isolated Azure payload'
with socket.socket() as listener:
    listener.settimeout(15)
    listener.bind(('127.0.0.1', 0))
    listener.listen(4)
    print(listener.getsockname()[1], flush=True)
    for expected_method in (('GET',) if content else ('HEAD', 'GET')):
        conn, _ = listener.accept()
        with conn:
            conn.settimeout(10)
            raw = b''
            while b'\r\n\r\n' not in raw:
                chunk = conn.recv(4096)
                assert chunk
                raw += chunk
                assert len(raw) < 32768
            lines = raw.decode('ascii').split('\r\n')
            method, target, protocol = lines[0].split(' ')
            assert method == expected_method and protocol == 'HTTP/1.1'
            fields = {}
            for line in lines[1:]:
                if not line:
                    break
                name, value = line.split(':', 1)
                name = name.lower()
                assert name not in fields
                fields[name] = value.strip()
            assert fields['accept-encoding'] == 'identity'
            assert fields['x-ms-version'] == '2023-11-03'
            assert fields.get('range') is None
            if content:
                if mode == 'sas':
                    assert target == '/container/dir/a%20b.bin?api-version=2023-11-03&' + raw_sas
                else:
                    assert target == '/container/dir/a%20b.bin'
            else:
                assert target.startswith('/container/dir/a%20b.bin?versionid=2025-10-16T12%3A00%3A00.0000000Z')
                if mode == 'sas':
                    assert target.endswith('&api-version=2023-11-03&' + raw_sas)
                else:
                    assert target.endswith('versionid=2025-10-16T12%3A00%3A00.0000000Z')
            if mode == 'sas':
                assert 'authorization' not in fields
            if mode == 'key':
                date = fields['x-ms-date']
                parsed_date = email.utils.parsedate_to_datetime(date)
                assert abs((datetime.datetime.now(datetime.timezone.utc) - parsed_date).total_seconds()) < 120
                # Independent REST definition: twelve standard fields, then sorted
                # x-ms headers, encoded path and decoded/sorted query values.
                uri = urllib.parse.urlsplit(target)
                values = {}
                for pair in (uri.query.split('&') if uri.query else []):
                    name, value = pair.split('=', 1)
                    name = urllib.parse.unquote(name).lower()
                    values.setdefault(name, []).append(urllib.parse.unquote(value))
                resource = '/' + account + uri.path
                for name in sorted(values):
                    resource += '\n' + name + ':' + ','.join(sorted(values[name]))
                standard = [method] + [''] * 10 + [fields.get('range', '')]
                preimage = '\n'.join(standard) + '\nx-ms-date:' + date + '\nx-ms-version:2023-11-03\n' + resource
                signature = base64.b64encode(hmac.new(key, preimage.encode('utf-8'), hashlib.sha256).digest()).decode('ascii')
                assert hmac.compare_digest(fields['authorization'], 'SharedKey ' + account + ':' + signature)
            elif mode == 'bearer':
                assert fields['authorization'] == 'Bearer synthetic-azure-bearer'
            elif mode == 'public':
                assert 'authorization' not in fields
            headers = ('HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: ' + str(len(body)) + '\r\nETag: "wire-etag"\r\nx-ms-blob-type: BlockBlob\r\nx-ms-version-id: ' + version + '\r\n\r\n').encode('ascii')
            conn.sendall(headers + (body if method == 'GET' else b''))
print('VERIFIED ' + requested_mode, flush=True)
