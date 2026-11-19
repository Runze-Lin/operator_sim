"""Run reviewed upstream decoder methods with symbolic shape and recording leaf modules."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
LOCK = json.loads(Path(__file__).with_name('sources.json').read_text())
MODELS = {'DeepSeek-V4-Pro':'deepseek_v4_model', 'DeepSeek-V4.1-Flash':'deepseek_v41_model','Kimi-K3':'kimi_k3'}
SCENARIOS = [('unchunked_prefill',8,32768,32768),('chunked_prefill',8,256,32768),('verify_8',8,9,32768)]

class Symbol:
    """Shape/dtype-only tensor. Only reviewed operations of selected methods are implemented."""
    def __init__(self,*dims,dtype='bf16',on_float=None):
        self.on_float=on_float
        self.shape=dims
        self.dtype=dtype
    def float(self):
        if self.on_float is not None:
            self.on_float()
        return self.to('fp32')
    def clone(self):
        return Symbol(*self.shape,dtype=self.dtype)
    def view(self,*dims):
        if len(dims)==2 and dims[0]==-1 and self.shape[-1]==dims[1]:
            return self
        if self.shape!=dims: raise ValueError(('unsupported reshape', self.shape,dims))
        return self
    def to(self,dtype):
        return Symbol(*self.shape,dtype=dtype)

def read_sources(cache,fetch):
    classes={}
    for key,rec in LOCK['files'].items():
        path=cache/rec['path']
        if not path.exists() and fetch:
            url=f"https://raw.githubusercontent.com/{LOCK['repo']}/{LOCK['revision']}/{rec['path']}"
            path.parent.mkdir(parents=True,exist_ok=True)
            with urllib.request.urlopen(url,timeout=30) as source:
                path.write_bytes(source.read())
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=rec['sha256']:
            raise ValueError(f'Source digest differs: {path}')
        tree=ast.parse(raw)
        if key=='profile':
            profiles=[node for node in ast.walk(tree) if isinstance(node,ast.Dict)]
            for profiles_dict in profiles:
                for target,entry in zip(profiles_dict.keys,profiles_dict.values):
                    if isinstance(target,ast.Attribute) and target.attr=='A5':
                        caps=next(item.value for item in entry.keywords if item.arg=='capabilities')
                        classes['a5_capabilities']={v.attr for v in ast.walk(caps) if isinstance(v,ast.Attribute)}
        classes[key]={node.name:node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef))}
    return classes

def bind(classes,file,cls,name,env):
    node=(next(f for f in classes[file][cls].body if isinstance(f,ast.FunctionDef) and f.name==name)
          if cls is not None else classes[file][name])
    # Annotation evaluation and class decorators are irrelevant to a decoder call.
    node.decorator_list=[]
    module=ast.Module([ast.ImportFrom('__future__',[ast.alias('annotations')],0),node],[])
    exec(compile(ast.fix_missing_locations(module),LOCK['files'][file]['path'],'exec'),env)
    return env[name]

def probe(model,cfg,scenario,classes):
    phase,batch,q,ctx=scenario
    tokens=batch*q
    hidden=cfg['hidden_size']; x=Symbol(tokens,hidden); p=Symbol(tokens,dtype='int64')
    experts=cfg.get('n_routed_experts',cfg.get('num_experts'))
    k=cfg.get('num_experts_per_tok',cfg.get('num_experts_per_token'))
    counts=Counter(); paths={}
    def leaf(name,ret):
        def run(*args,**kwargs):
            events.append(name)
            return ret
        return run
    file=MODELS[model]
    torch=NS(ops=NS(_C_ascend=NS(),vllm=NS()),int32='int32',float32='fp32',empty_like=lambda t:t.clone())
    env={'torch':torch}
    def method(cls,n,owner,f=file):
        return MethodType(bind(classes,f,cls,n,env),owner)
    weight=NS(weight=Symbol(hidden),variance_epsilon=cfg.get('rms_norm_eps',1e-6))
    is_kimi=model=='Kimi-K3'
    is_v41=model=='DeepSeek-V4.1-Flash'
    cls='AscendKimiDecoderLayer' if is_kimi else ('DeepseekV41DecoderLayer' if is_v41 else 'DeepseekV4DecoderLayer')
    layer=NS(hc_attn_fn=None,hc_attn_scale=None,hc_attn_base=None,hc_ffn_fn=None,hc_ffn_scale=None,hc_ffn_base=None,
             use_sequence_parallel=False,use_sequence_parallel_moe=False,enable_dsa_cp=False,input_layernorm=leaf('RMSNorm',x),
             post_attention_layernorm=None)
    forward=method(cls,'forward',layer)
    if is_kimi:
        layer.use_attn_residuals=True
        layer.prepare_attn_residual=method(cls,'prepare_attn_residual',layer)
        layer.forward_attn_residual=method(cls,'forward_attn_residual',layer)
        for n in ('input_layernorm','post_attention_layernorm','mlp_res_proj','mlp_res_norm','self_attention_res_proj','self_attention_res_norm'):
            setattr(layer,n,weight)
        torch.ops._C_ascend.attn_res_fwd=leaf('_C_ascend.attn_res_fwd',(x,x,None))
        mlp=NS(gate=leaf('gate:FP32_logits',(Symbol(tokens,experts,dtype='fp32'),None)))
        select=NS(bias_vl=None,top_k=k,topk_group=cfg['topk_group'],num_expert_group=cfg['num_expert_group'],
                  renormalize=cfg['moe_renormalize'],scoring_func=cfg['moe_router_activation_func'],
                  e_score_correction_bias=Symbol(experts,dtype='fp32'),routed_scaling_factor=cfg['routed_scaling_factor'])
        select.is_fused_supported=method('AscendFusedTopKRouter','is_fused_supported',select,'fused_topk_router')
        eligible=bind(classes,'router_factory',None,'check_npu_moe_gating_top_k',env)
        assert eligible(k,select.renormalize,select.topk_group,select.num_expert_group,select.scoring_func)
        route=method('AscendFusedTopKRouter','_compute_routing',select,'fused_topk_router')
        env['DeviceOperator']=NS(moe_gating_top_k=bind(classes,'device_op','A5DeviceAdaptor','moe_gating_top_k',env))
        vals=Symbol(tokens,k,dtype='fp32');ids=Symbol(tokens,k,dtype='int32')
        def gate(logits,**kw):
            assert logits.shape==(tokens,experts) and logits.dtype=='fp32' and kw['k']==k and kw['norm_type']==1
            assert (kw['group_count'],kw['k_group'],kw['renorm'],kw['group_select_mode'])==(1,1,0,1)
            events.append('torch_npu.npu_moe_gating_top_k')
            return vals,ids,None
        env['torch_npu']=NS(npu_moe_gating_top_k=gate)
        def experts_stub(hidden_states,router_logits):
            # The fused-expert runner is a test double selecting the documented fused router.
            route(hidden_states,router_logits,None)
            events.append('FusedMoE.experts')
            return x
        mlp.experts=experts_stub
        routed=method('AscendKimiMoE','forward',mlp)
    else:
        layer.hc_pre=leaf('hc_pre_v3' if is_v41 else 'hc_pre',(x,None,None,None) if is_v41 else (x,None,None))
        hc=Symbol(tokens,cfg['hc_mult'],hidden)
        layer.hc_post=leaf('hc_post',hc)
        # Exercise upstream helper including its fallback and cast. Built-in A5
        # lacks RMS_NORM_CAST in the pinned profile; Pro explicitly disables
        # the custom implementation (registered only for 910B/910_93).
        layer.post_attention_layernorm=leaf('post_attention_RMSNorm',
                Symbol(tokens,hidden,on_float=lambda:events.append('activation.float:BF16_to_FP32')))
        layer.post_attention_layernorm.weight=weight.weight
        layer.post_attention_layernorm.variance_epsilon=weight.variance_epsilon
        assert 'RMS_NORM_CAST' not in classes['a5_capabilities']
        env['DeviceOperator']=NS(rms_norm_cast=lambda *args:None)
        env['enable_custom_op']=lambda:False
        layer.rms_norm_cast=method(cls,'rms_norm_cast',layer)
        # Execute the decoder's real MoE method for the external FP32 gate
        # scenario. Fused experts/collectives stay opaque.
        exp_leaf=leaf('FusedMoE.experts',x)
        exp_leaf.is_internal_router=False
        moe=NS(gate=NS(tid2eid=None,weight=Symbol(experts,hidden,dtype='fp32')),experts=exp_leaf,
               is_sequence_parallel=False,tp_size=1)
        env['F']=NS(linear=leaf('gate:FP32_logits',Symbol(tokens,experts,dtype='fp32')))
        layer.mlp=method('DeepseekV41MoE' if is_v41 else 'DeepseekV4MoE','forward',moe)
        attn=NS(v41_layer_name='probe.dsa',dsa_attn=leaf('DSA',x))
        torch.ops.vllm.dsa_v41_forward=leaf('vllm.dsa_v41_forward',None)
        if is_v41:
            hc.device=NS(type='npu')  # Device marker only, no runtime.
            edim=(cfg['engram_max_ngram_size']-1)*cfg['engram_n_heads']*cfg['engram_head_dim']
            ew=NS(wkv=leaf('engram.wkv',Symbol(tokens,(cfg['hc_mult']+1)*hidden)),
                  k_weight=None,q_weight=None,eps=weight.variance_epsilon)
            env['HardwareCapability']=NS(ENGRAM_UNROTATED_GATE='ENGRAM_UNROTATED_GATE')
            env['get_current_hardware_profile']=lambda:NS(supports=classes['a5_capabilities'].__contains__)
            env['fused_engram_gate']=leaf('triton.fused_engram_gate',hc)
            engram=method('AscendEngram','forward',ew,'engram')
        common=method('DeepseekV41Attention' if is_v41 else 'DeepseekV4Attention','forward',attn)
    for idx in range(cfg['num_hidden_layers']):
        events=[]
        if is_kimi:
            b=cfg['attn_res_block_size']
            layer.block_write_idx=idx//b
            layer.is_block_write_layer=idx%b==0
            layer.prev_valid_blocks=(idx+b-1)//b
            is_moe=idx>=cfg['first_k_dense_replace'] and idx%cfg['moe_layer_freq']==0
            layer.mlp=routed if is_moe else leaf('dense_mlp',x)
            kda=idx+1 in cfg['linear_attn_config']['kda_layers']
            layer.self_attn=leaf('KDA' if kda else 'no_RoPE_MLA',x)
            forward(p,x,x,defer_mlp_add=True,optimize_prefill=phase!='verify_8')
        else:
            if is_v41 and idx in cfg['engram_layer_ids']:
                engram(hc,Symbol(tokens,edim),Symbol(tokens,dtype='bool'),None)
            layer.self_attn=common
            forward(p,hc,None,input_ids=p)
        counts.update(events)
        paths.setdefault('/'.join(events),[]).append(idx)
    info=dict(model=model,phase=phase,batch=batch,q_per_request=q,context_per_request=ctx,total_tokens=tokens,
              hidden=hidden,decoder_layers=cfg['num_hidden_layers'],leaf_counts=dict(counts),layer_paths=paths)
    if is_v41:
        info['engram_per_call']=dict(row_dim=edim,kv_dim=(cfg['hc_mult']+1)*hidden,
                                     lookup_bytes_if_bf16=2*tokens*edim,
                                     wkv_scalar_ops=2*tokens*edim*(cfg['hc_mult']+1)*hidden)
    return info

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-cache',type=Path,default=ROOT/'runs/workloads19/sources')
    p.add_argument('--fetch',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'results/v0.19/workloads/decoder-probe.json')
    args=p.parse_args()
    classes=read_sources(args.source_cache,args.fetch)
    results=[]
    for model in MODELS:
        full=json.loads((ROOT/'configs/models'/f'{model}.json').read_text())
        cfg=full.get('text_config',full)
        results.extend(probe(model,cfg,s,classes) for s in SCENARIOS)
    args.output.write_text(json.dumps(dict(kind='isolated decoder source execution with leaf doubles',sources=LOCK,
        scenarios=results),indent=2)+'\n')

if __name__=='__main__':
    main()
