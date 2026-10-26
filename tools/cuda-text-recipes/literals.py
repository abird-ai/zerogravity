"""Closed parser for the existing generated Rust data; never eval supplied code."""
import json
import re


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


TOKEN = re.compile(r'\s+|//[^\n]*|"(?:\\.|[^"\\])*"|[A-Za-z_][A-Za-z_0-9]*|[0-9][0-9_]*|::|[^\s]')
STRUCTS = frozenset(('Packet', 'CubinKernelManifest', 'CubinBinding', 'CubinParameter',
                     'CubinPointerFixup', 'SiteRecipe', 'Rule', 'Cohort'))


class Literal:
    def __init__(self, text):
        self.tokens = [token for token in TOKEN.findall(text)
                       if not token.isspace() and not token.startswith('//')]
        self.at = 0

    def peek(self):
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def take(self, expected=None):
        token = self.peek()
        require(token is not None and (expected is None or token == expected),
                'complete closed generated Rust literal: ' + str(expected))
        self.at += 1
        return token

    def sequence(self, end):
        values = []
        while self.peek() != end:
            values.append(self.value())
            if self.peek() == end:
                break
            self.take(',')
        self.take(end)
        return values

    def value(self):
        token = self.take()
        if token == '&':
            self.take('[')
            value = self.sequence(']')
        elif token == '[':
            value = self.sequence(']')
        elif token == '(':
            value = self.sequence(')')
        elif token.startswith('"'):
            value = json.loads(token)
        elif token[0].isdigit():
            value = int(token.replace('_', ''))
        elif token in ('true', 'false'):
            value = token == 'true'
        elif token == 'None':
            value = None
        else:
            name = token
            while self.peek() == '::':
                self.take('::')
                name += '::' + self.take()
            if name == 'vec':
                self.take('!')
                self.take('[')
                value = self.sequence(']')
            elif name == 'Some':
                self.take('(')
                value = {'Some': self.value()}
                self.take(')')
            elif self.peek() == '{':
                require(name in STRUCTS, 'known generated Rust data structure')
                self.take('{')
                value = {'type': name}
                while self.peek() != '}':
                    key = self.take()
                    require(re.fullmatch(r'[a-z_]+', key) and key not in value,
                            'unique generated literal field')
                    self.take(':')
                    value[key] = self.value()
                    if self.peek() != '}':
                        self.take(',')
                self.take('}')
            else:
                require(re.fullmatch(r'(?:Role|CubinIo|CubinLaunchAttributes|ProjectionImage)::\w+|qwen(?:08|2b)_packet_\d+', name),
                        'closed generated enum or packet reference')
                value = name
        while self.peek() == '.':
            self.take('.')
            require(self.take() in ('into', 'to_vec'), 'literal ownership conversion only')
            self.take('(')
            self.take(')')
        return value

    def parse(self):
        value = self.value()
        require(self.at == len(self.tokens), 'no trailing executable Rust content')
        return value


def balanced(text, offset, opening='{', closing='}'):
    require(text[offset] == opening, 'exact generated declaration start')
    depth, quoted, escaped = 0, False, False
    for at in range(offset, len(text)):
        char = text[at]
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return offset, at + 1
    raise ValueError('incomplete generated declaration')


def declaration(text, name):
    matches = list(re.finditer(r'(?m)^(?:static|const|pub\(super\) const) ' + re.escape(name) + r'\b', text))
    require(len(matches) == 1, 'unique generated declaration: ' + name)
    start = matches[0].start()
    equals = text.index('=', matches[0].end())
    if text[equals + 1:].lstrip().startswith('&['):
        a, b = balanced(text, text.index('[', equals), '[', ']')
        value_start = a - 1
    else:
        a, b = balanced(text, text.index('{', equals))
        value_start = equals + 1
    end = text.index(';', b) + 1
    return {'start': start, 'end': end, 'value_start': value_start, 'value_end': b,
            'value': Literal(text[value_start:b]).parse(), 'raw': text[start:end]}


def factories(text, prefix):
    result = {}
    pattern = r'fn (' + re.escape(prefix) + r'_packet_\d+)\(\) -> Packet\s*\{'
    for match in re.finditer(pattern, text):
        name = match[1]
        require(name not in result, 'unique generated packet factory')
        a, b = balanced(text, match.end() - 1)
        value = Literal(text[a + 1:b - 1]).parse()
        require(value['type'] == 'Packet', 'literal packet factory only')
        result[name] = {'value': value, 'raw': text[match.start():b]}
    require(bool(result), 'existing genuine packet factories present')
    return result
