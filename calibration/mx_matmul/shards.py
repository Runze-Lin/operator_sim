"""Read large CA physical-core trace shards without retaining all instructions."""

import json
import re

CHUNK = 1 << 20
ARRAY_START = re.compile(r'"traceEvents"\s*:\s*\[')


def trace_events(path):
    if path.stat().st_size <= 8 * CHUNK:
        yield from json.loads(path.read_bytes())['traceEvents']
        return
    # Vendor traceEvents is a top-level array and its final field. Decode one
    # event at a time; compact the input string only at chunk boundaries.
    decoder = json.JSONDecoder()
    with path.open(encoding='utf-8') as handle:
        buf = handle.read(CHUNK)
        match = ARRAY_START.search(buf)
        if not match or 'traceEvents' not in json.loads(buf[:match.end()] + ']}'):
            raise ValueError(f'Unsupported trace header: {path}')
        pos = match.end()
        first = True
        while True:
            while pos >= len(buf) or buf[pos].isspace():
                if pos < len(buf):
                    pos += 1
                    continue
                buf = handle.read(CHUNK)
                if not buf:
                    raise ValueError(f'Incomplete traceEvents in {path}')
                pos = 0
            if buf[pos] == ']':
                if (buf[pos+1:] + handle.read()).strip() != '}':
                    raise ValueError(f'Incomplete or unexpected trace footer in {path}')
                return
            if not first:
                if buf[pos] != ',':
                    raise ValueError(f'Missing event separator in {path}')
                pos += 1
            first = False
            while True:
                try:
                    while pos < len(buf) and buf[pos].isspace():
                        pos += 1
                    event, pos = decoder.raw_decode(buf, pos)
                    break
                except json.JSONDecodeError:
                    extra = handle.read(CHUNK)
                    if not extra:
                        raise ValueError(f'Invalid or truncated CA event in {path}') from None
                    buf, pos = buf[pos:] + extra, 0
            yield event
