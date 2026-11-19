"""Verify the published multi-launch tree if msopprof moved it after its parse.

The original collector checks the pre-publication device0 tree, which survives a
cap. Successful msopprof publication moves the kernel/ordinal tree up one level.
Only the source directory lookup changes. All instruction/ELF/time checks and
original receipts stay intact. This script never executes the simulator.
"""
import sys
import prefix
import verify


class PublishedFolder:
    def __init__(self, capture, published):
        self.capture = capture
        self.published = published

    def __truediv__(self, relative):
        return self.capture / relative

    def glob(self, pattern):
        if pattern != 'profile/OPPROF_*/device0/' + verify.KERNEL:
            raise ValueError('unexpected profiling directory lookup')
        return [self.published]


def inspect(chip):
    plan = prefix.read(prefix.PLAN)
    source = prefix.OUT / (chip + '.json')
    original = prefix.read(source)
    if 'controls' in original or original.get('reason') != "ValueError('device profile path')":
        return source
    dest = prefix.OUT / ('verified-published-' + chip + '.json')
    if dest.is_file():
        prior = prefix.read(dest)
        if prior.get('collector_sha256') == verify.digest(source) and prior.get('registration_sha256') == verify.digest(prefix.PLAN):
            return dest
    capture = prefix.RAW / chip
    [published] = list(capture.glob('profile/OPPROF_*/' + verify.KERNEL))
    expected = {str(job['i']) for job in plan['queue'][chip]['rows']}
    observed = {p.name for p in published.iterdir() if p.is_dir() and p.name.isdigit()}
    if observed != expected:
        raise ValueError('published launch ordinals mismatch ' + repr((observed, expected)))
    data = prefix.assess(chip, plan['queue'][chip], plan,
                         PublishedFolder(capture, published), original['process'])
    data['collector_receipt'] = verify.rel(source)
    data['collector_sha256'] = verify.digest(source)
    data['published_device_path'] = verify.rel(published)
    prefix.put(dest, data)
    return dest


if __name__ == '__main__':
    for chip in sys.argv[1:] or prefix.TARGETS:
        print(inspect(chip))
