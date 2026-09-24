"""Test whether hard top8 prevents the new retrieval metric learning missing values.

All available records compete by the SAME cosine score, temperature and causal
mask. This changes gradient support, not the byte target, capacity or loss.
It is a dense pointer-retrieval control, not a new named semantic algorithm.
"""
from drrem.core.induction_transport import InductionTransportMachine
from drrem.core.induction import induction_candidates


class DenseInductionMachine(InductionTransportMachine):
    consumer_description='same final continuation expert with differentiable support over all eligible past records; no hard top8 search barrier'

    def candidates(self,features,ids,valid):
        return induction_candidates(features,ids,valid,near=64,span=1024,topm=ids.shape[1],vocab=self.cfg.vocab)
