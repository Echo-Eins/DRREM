"""Select the saved transport computation, not merely compatible tensor shapes."""
from pathlib import Path

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine
from drrem.data.protocol import file_digest


def model_from_protocol(protocol):
    if 'semantic_flywheel' in protocol:
        if any(key in protocol for key in ('adaptive_phase', 'temporal_memory', 'prompt_banks', 'flywheel')):
            raise ValueError('semantic flywheel requires its declared attention core')
        from drrem.core.semantic_flywheel import model_from_semantic_flywheel_protocol
        return model_from_semantic_flywheel_protocol(protocol)
    if sum(key in protocol for key in ('ring_frame','content_shift','adaptive_phase'))>1:
        raise ValueError('checkpoint declares two different phase frames')
    if 'prompt_banks' in protocol:
        if 'adaptive_phase' not in protocol or 'query_phase' in protocol:
            raise ValueError('prompt banks require their saved adaptive phase configuration')
        from drrem.core.prompt_phase_transport import model_from_prompt_bank_protocol
        return model_from_prompt_bank_protocol(protocol)
    if 'query_phase' in protocol:
        if 'adaptive_phase' not in protocol:raise ValueError('query phase requires the saved base phase configuration')
        from drrem.core.query_phase_transport import model_from_query_phase_protocol
        return model_from_query_phase_protocol(protocol)
    if 'adaptive_phase' in protocol:
        from drrem.core.adaptive_phase_transport import model_from_adaptive_protocol
        return model_from_adaptive_protocol(protocol)
    if 'ring_frame' in protocol:
        from drrem.core.ring_phase_transport import model_from_ring_protocol
        return model_from_ring_protocol(protocol)
    if 'content_shift' in protocol:
        from drrem.core.phase_shift_transport import model_from_shift_protocol
        return model_from_shift_protocol(protocol)
    cfg=CausalTransportConfig(**protocol['model'])
    schedule=protocol.get('schedule',{'schedule':'synchronous'})
    name=schedule['schedule']
    files=['drrem/core/causal_transport.py']
    if 'temporal_memory' in protocol:
        if name!='synchronous':raise ValueError('bounded temporal memory requires synchronous spatial transport')
        files.append('drrem/core/nondecay_transport.py')
    if name=='sequential down/up sweeps':files.append('drrem/core/sweep_transport.py')
    elif name!='synchronous':raise ValueError(f'unsupported saved transport schedule: {name}')
    root=Path(__file__).resolve().parents[2]
    for file in files:
        expected=protocol.get('source_hashes',{}).get(file)
        if expected is not None and file_digest(root/file)!=expected:
            raise ValueError(f'architecture source differs from checkpoint: {file}')
    if 'temporal_memory' in protocol:
        from drrem.core.nondecay_transport import MemoryConfig,NondecayTransportMachine
        return NondecayTransportMachine(cfg,MemoryConfig(**protocol['temporal_memory']))
    if name=='synchronous':return CausalTransportMachine(cfg)
    from drrem.core.sweep_transport import SweepTransportMachine
    return SweepTransportMachine(cfg,cycles=schedule['cycles'],temporal_placement=schedule['temporal_placement'])
