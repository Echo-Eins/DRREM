"""Compare compiled training, compiled inference, and native BF16 forwards."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.root/'source'))
    import torch
    from drrem.core.causal_transport import CausalTransportConfig
    from scripts.train_full_signal_trial import make_trial_model
    def helper(name):
        path = Path(__file__).resolve().parents[1]/'drrem/diagnostics'/f'{name}.py'
        spec = importlib.util.spec_from_file_location(name,path)
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    native, logic = helper('native_generation'), helper('logic_tasks')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    prompts = [logic.build_tasks()[i]['prompt'] for i in (0,1,4,5,32,33,36,37)]
    result = {}
    def diff(x,y,at):
        x,y=x[:,at].float(),y[:,at].float()
        p,q=y[:,0].log_softmax(-1),x[:,0].log_softmax(-1)
        return dict(max_abs=float((x-y).abs().max()), mean_abs=float((x-y).abs().mean()),
                    max_kl=float((p.exp()*(p-q)).sum(-1).max()),
                    argmax_agreement=float((x[:,0].argmax(-1)==y[:,0].argmax(-1)).float().mean()),
                    bitwise_equal=bool(torch.equal(x,y)))
    for name in ('warm_base','warm_fourier','fresh_bridge'):
        ck=torch.load(a.root/name/'checkpoint_3mb.pt',map_location='cpu',weights_only=False,mmap=True)
        model=make_trial_model(ck['protocol']['variant'],CausalTransportConfig(**ck['protocol']['model'])).cuda().eval()
        model.load_state_dict(ck['model'])
        frame=native.NativeGenerationFrame(model,prompts,512,512)
        original=model.transport_hop;compiled=torch.compile(original,dynamic=False)
        row=[]
        for step in range(3):
            model.transport_hop=original
            eager=frame.all_logits()
            model.transport_hop=compiled;model.train()
            with torch.enable_grad(),frame.autocast():
                x=model(frame.ids,frame.valid)
            training=x.detach();del x
            model.eval()
            with torch.no_grad(),frame.autocast():inference=model(frame.ids,frame.valid)
            with torch.enable_grad(),frame.autocast():x=model(frame.ids,frame.valid)
            eval_grad=x.detach();del x
            report=dict(step=step,native_vs_training=diff(eager,training,frame.cursor),
                        compiled_nograd_vs_training=diff(inference,training,frame.cursor),
                        compiled_eval_grad_vs_training=diff(eval_grad,training,frame.cursor))
            row.append(report);print(json.dumps(dict(model=name,**report)),flush=True)
            if step<2:frame.consume(training[:,frame.cursor,0].argmax(-1))
            del training,inference,eval_grad,eager
        result[name]=row;a.out.write_text(json.dumps(result,indent=2)+'\n')
        model.transport_hop=original
        del ck,model,frame,original,compiled;torch.cuda.empty_cache()


if __name__=='__main__':main()
