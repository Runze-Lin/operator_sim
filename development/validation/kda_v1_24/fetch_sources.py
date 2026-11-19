"""Restore three official release source files; content verified by pinned Git blob."""
import hashlib,json,urllib.request
from investigate import SOURCES, RAW,probe

def main():
    tree=json.loads((probe.RAW/'fla-source-tree.json').read_bytes())
    if tree['sha']!='5cc699069c7f91a35aabd583032c8720a0b626fa':
        raise ValueError('unexpected release tree')
    hashes={e['path']:e['sha'] for e in tree['tree']}
    for name,repo_path in SOURCES.items():
        dest=RAW/name
        data=dest.read_bytes() if dest.is_file() else urllib.request.urlopen(
            'https://raw.githubusercontent.com/flashserve/flash-linear-attention-npu/v26.9.1-release/'+repo_path,timeout=45).read()
        actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if actual!=hashes[repo_path]:raise ValueError('unverified release source: '+name)
        if not dest.is_file():dest.write_bytes(data)
        print(name,actual)
if __name__=='__main__':main()
