#!/usr/bin/env python3
"""Deduplicated workload shape coverage; predictions are not CA validation."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.model_v4 import Case,Predictor

def build():
 p=Predictor(ROOT/'results/v0.4/model.json');rows=[];unique={};models=[]
 for name in ('DeepSeek-V4.1-Flash','Kimi-K3','DeepSeek-V4-Pro'):
  path=ROOT/'configs/models'/f'{name}.json';full=json.loads(path.read_text());d=full.get('text_config',full)
  models.append({'name':name,'config_sha256':sha256(path),'hidden_size':d['hidden_size'],'query_heads_per_tp8':d['num_attention_heads']//8,
                 'routed_expert_hidden_size':d.get('routed_expert_hidden_size'),
                 'actual_head_dim':d.get('head_dim'),'qk_nope_head_dim':d.get('qk_nope_head_dim'),'qk_rope_head_dim':d.get('qk_rope_head_dim'),
                 'sliding_window':d.get('sliding_window'),'compress_ratios':d.get('compress_ratios'),
                 'linear_attn_config':d.get('linear_attn_config')})
  for context in (32768,131072,262144,524288):
   for phase,batches,qs in [('prefill',(1,4,16),(512,2048)),('decode',(1,4,16,32,128,256),(1,9))]:
    for batch in batches:
     for q in qs:
      for progress in ((0,1024) if phase=='decode' else (0,)):
       # Equal-length requests, per-request chunk, TP splits heads only.
       quant=Case('quant',(batch*q,d['hidden_size']));fa=Case('fa',(q,context+progress,128,batch,d['num_attention_heads']//8))
       ids=[]
       for c in (quant,fa):
        key=c.op+':'+'x'.join(map(str,c.shape));ids.append(key)
        if key not in unique:
         pred=p.predict(c)
         unique[key]={'op':c.op,'shape':list(c.shape),'prediction':pred,'status':'analytical_only_not_ca_validated'}
       rows.append({'model':name,'phase':phase,'prompt_tokens':context,'output_tokens':1024,'generated_progress':progress,
                    'batch_per_card':batch,'q_per_request':q,'attention_tp_assumption':8,'quant_key':ids[0],'dense_fa_proxy_key':ids[1]})
 return {'assumptions':{'prefill_batch_per_card':[1,4,16],'decode_batch_per_card':[1,4,16,32,128,256],
  'prompt_lengths':[32768,131072,262144,524288],'output_tokens':1024,'speculative_draft':8,'verification_q':9,
  'prefill_chunk_per_request':[512,2048],'attention_tp':8,
  'warning':'Chunk sizes and TP are unconfirmed defaults. Batch is local per-card equal-length requests; no extra division by TP for tokenwise quant. Real schedulers may budget total batched tokens, changing concurrent chunks.'},
  'scope':'Quant activation-width proxies plus ordinary noncausal D128 dense FA component proxies. Not model kernel inventory; MLA/KDA/compressed attention, masks, cache layouts and TP collectives require actual runtime dispatch. Output1k is generation horizon, not one Q1024 invocation.',
  'models':models,'scenarios':rows,'unique_predictions':unique}

def main():
 result=build();out=ROOT/'results/v0.4';(out/'workload-coverage.json').write_text(json.dumps(result,indent=2)+'\n')
 print('scenarios',len(result['scenarios']),'unique',len(result['unique_predictions']))
if __name__=='__main__':main()
