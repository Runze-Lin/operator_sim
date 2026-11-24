"""Run the same 1Mi QSLI fixture with all 32 tiles explicitly owned by team zero."""
import long_row_probe as probe


class SingleOwnerMetadata:
    def __init__(self, original):
        self.original = original

    def run_host(self, **kwargs):
        # Request only ONE owner when building the *input metadata*. The AOT
        # device binary, its normal launch width, tensors and golden are untouched.
        assert kwargs['workers'] == 32 and kwargs['ratio'] == 1
        rc = self.original.run_host(**{**kwargs, 'workers': 1})
        assert rc == 0 and kwargs['output'] == [1,0,0,0,1,0,0,0]+[0]*1016
        print('STATIC_SINGLE_OWNER worker0, 32 key tiles, no cross-team merge',flush=True)
        return rc


if __name__ == '__main__':
    probe.metadata_kernel = SingleOwnerMetadata(probe.metadata_kernel)
    probe.main()
