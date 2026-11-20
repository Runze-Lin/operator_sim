"""Benchmark optional native replayer and verify every output against the independent strict Python pass."""
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory
from time import perf_counter
from watch import OUT, ROOT, read, save, digest

if __name__=='__main__':
    rows=[]
    with TemporaryDirectory(prefix='matmul33-replay-') as tmp:
        binary=Path(tmp)/'replay'
        src=Path(__file__).with_name('replay.cc')
        run(['c++','-O3','-std=c++17','-Wall','-Wextra',str(src),'-lcrypto','-o',str(binary)],check=True)
        for cert in sorted(OUT.glob('*-certificate.json')):
            old=read(cert)
            stage,=read(ROOT/old['reference'])['stages']
            directory=(ROOT/stage['trace']).parent
            paths=sorted(directory.glob('core*/trace.json'))
            start=perf_counter()
            lines=run([str(binary),*map(str,paths)],check=True,text=True,capture_output=True).stdout.splitlines()
            wall=perf_counter()-start
            assert len(lines)==len(paths)==len(old['cores'])
            for path,line in zip(paths,lines):
                first,last,marker,executed,mmad,sha=line.split('\t')
                r=old['cores'][path.parent.name]
                assert list(map(float,(first,last,marker)))==[r['first_us'],r['retire_us'],r['end_marker_us']]
                assert (int(executed),int(mmad),sha)==(r['count'],r['mmad'],r['sha256'])
            rows.append(dict(shape=old['shape'],sku=old['sku'],raw_json_bytes=old['shard_input_bytes'],core_count=len(paths),
                             archive_certificate=cert.name,archive_certificate_sha256=digest(cert),
                             python_wall_s=old['observed_offline_reduce_s'],native_wall_s=wall,
                             observed_replay_ratio=old['observed_offline_reduce_s']/wall))
            print(old['shape'],old['sku'],round(wall,3),flush=True)
    save('native-replay.json',dict(source_sha256=digest(src),scope='Single-shard-at-a-time mmap; SHA-256, complete records, physical END, MMAD and extrema exactly equal to independent archive on every core. C++ compile excluded. OS caches and host load affect cost.',rows=rows))
