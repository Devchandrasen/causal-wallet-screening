"""Two-hop TGAT node classifier adapted to strict causal wallet screening.

Implements harmonic time encoding and recursive temporal attention from Xu et
al., ICLR 2020, https://openreview.net/forum?id=rJeW1yHYwH. This is a task
adaptation, not the authors' original benchmark or pretrained implementation.
Node features are zero (no identity embeddings); directed edge attributes are
two directional indicators. The classification head also receives the same
20 cutoff-safe wallet variables used by PC-TET. No graph-wide statistics are
computed. A chronological event index serves only cutoff-filtered queries.
Root queries include the complete cutoff bin. Child queries are strictly
earlier than the triggering interaction bin, since intra-bin order is unknown.
"""
from __future__ import annotations
import argparse
import copy
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from reconstruct_wallet_experiments import ACTIVITY, SNA, load_raw, evaluate, select_f1_threshold
from audit_utils import environment, config_hash, save_json, sha256, utc_now, prediction_diagnostics


class EventIndex:
    def __init__(self, source, destination, times, directions, transaction):
        order = np.lexsort((transaction, destination, times, source))
        self.source = source[order]
        self.destination = destination[order]
        self.time = times[order]
        self.direction = directions[order]
        nodes, starts, counts = np.unique(self.source, return_index=True, return_counts=True)
        self.bounds = {int(n): (int(s), int(s+c)) for n, s, c in zip(nodes, starts, counts)}

    def query(self, node, cutoff, count, inclusive=False):
        start, stop = self.bounds.get(int(node), (0, 0))
        end = start + np.searchsorted(self.time[start:stop], cutoff, side='right' if inclusive else 'left')
        begin = max(start, end-count)
        selected = np.arange(begin, end)
        assert np.all(self.time[selected] <= cutoff) if inclusive else np.all(self.time[selected] < cutoff)
        return self.destination[selected], self.time[selected], self.direction[selected]


def sampler_tests():
    index = EventIndex(np.array([1,1,1,2]), np.array([2,3,4,1]), np.array([1,2,5,1]), np.array([1,1,1,-1]), np.array([1,2,3,1]))
    a = index.query(1, 2, 16, inclusive=True)
    assert a[0].tolist() == [2,3]
    assert index.query(1, 2, 16)[0].tolist() == [2]
    changed = EventIndex(np.array([1,1,1,2,1]), np.array([2,3,99,1,100]), np.array([1,2,5,1,8]), np.array([1,1,1,-1,1]), np.array([1,2,3,1,4]))
    assert all(np.array_equal(x,y) for x,y in zip(a, changed.query(1,2,16,True)))
    assert len(index.query(999,2,16)[0]) == 0
    return {'root_cutoff_inclusive': True, 'child_cutoff_strict': True, 'future_mutation_invariance': True, 'empty_node': True}


def prepare(data_dir, results_dir, cache, k=16):
    cache.mkdir(parents=True, exist_ok=True)
    if (cache/'manifest.json').exists():
        return
    samples = pd.read_parquet(results_dir/'wallet_samples.parquet')
    _, _, inputs, outputs = load_raw(data_dir)
    pairs = inputs[['input_address','txId','Time step']].merge(outputs[['output_address','txId']], on='txId')
    pairs = pairs[pairs.input_address != pairs.output_address].drop_duplicates()
    names = pd.Index(pd.concat([pairs.input_address, pairs.output_address, samples.address]).unique()).sort_values()
    src = names.get_indexer(pairs.input_address).astype(np.int32)
    dst = names.get_indexer(pairs.output_address).astype(np.int32)
    t = pairs['Time step'].to_numpy(np.int16)
    tx = pairs.txId.to_numpy(np.int64)
    index = EventIndex(np.r_[src,dst], np.r_[dst,src], np.r_[t,t], np.r_[np.ones(len(t),np.int8),-np.ones(len(t),np.int8)], np.r_[tx,tx])
    root_node = names.get_indexer(samples.address)
    root_cutoff = samples.cutoff.to_numpy()
    root_time = np.zeros((len(samples), k),np.int16)
    root_direction = np.zeros((len(samples),k),np.int8)
    child_index = np.zeros((len(samples),k),np.int32)
    child_times = [np.zeros(k,np.int16)]
    child_directions = [np.zeros(k,np.int8)]
    lookup = {}
    for i, (node, cutoff) in enumerate(zip(root_node,root_cutoff)):
        neighbors, times, directions = index.query(node,cutoff,k,True)
        root_time[i,:len(times)] = times
        root_direction[i,:len(times)] = directions
        for j,(neighbor,event_time) in enumerate(zip(neighbors,times)):
            key = (int(neighbor),int(event_time))
            if key not in lookup:
                _, t_child, d_child = index.query(neighbor,event_time,k,False)
                ct, cd = np.zeros(k,np.int16),np.zeros(k,np.int8)
                ct[:len(t_child)],cd[:len(d_child)] = t_child,d_child
                lookup[key] = len(child_times)
                child_times.append(ct); child_directions.append(cd)
            child_index[i,j] = lookup[key]
        if i % 50000 == 0:
            print(f'prepared {i}/{len(samples)}; unique child states={len(lookup)}',flush=True)
    child_times, child_directions = np.stack(child_times),np.stack(child_directions)
    np.savez(cache/'temporal_queries.npz',root_time=root_time,root_direction=root_direction,
             child_index=child_index,child_time=child_times,child_direction=child_directions)
    save_json(cache/'manifest.json',{'neighbors_per_hop':k,'layers':2,'sample_count':len(samples),
              'directed_events':2*len(pairs),'child_states':len(lookup),'sampler_tests':sampler_tests(),
              'raw_sha256':{f:sha256(data_dir/f) for f in ['txs_features.csv','AddrTx_edgelist.csv','TxAddr_edgelist.csv']},
              'input_sha256':sha256(results_dir/'wallet_samples.parquet')})


class HarmonicTime(nn.Module):
    def __init__(self,dimension=32):
        super().__init__()
        self.frequency = nn.Parameter(torch.tensor(10.**(-np.linspace(0,9,dimension)),dtype=torch.float32))
        self.phase = nn.Parameter(torch.zeros(dimension))
    def forward(self,delta):
        return torch.cos(delta.unsqueeze(-1)*self.frequency+self.phase)


class TemporalAttention(nn.Module):
    def __init__(self,dim=32):
        super().__init__()
        self.attention = nn.MultiheadAttention(3*dim,4,dropout=.1,batch_first=True)
        self.normalization = nn.LayerNorm(3*dim)
        self.merge = nn.Sequential(nn.Linear(4*dim,64),nn.ReLU(),nn.Linear(64,dim))
    def forward(self,source,neighbor,edge,time_encoding,zero_time,mask):
        query = torch.cat([source,torch.zeros_like(source),zero_time],-1).unsqueeze(1)
        key = torch.cat([neighbor,edge,time_encoding],-1)
        # Empty neighborhoods need a masked-safe sentinel to prevent NaN.
        empty = mask.all(1)
        safe_mask = mask.clone()
        safe_mask[empty,0] = False
        attended,_ = self.attention(query,key,key,key_padding_mask=safe_mask,need_weights=False)
        attended = self.normalization(attended[:,0]+query[:,0])
        result = self.merge(torch.cat([attended,source],-1))
        return torch.where(empty[:,None],torch.zeros_like(result),result)


class CausalTGAT(nn.Module):
    def __init__(self,wallet_dim):
        super().__init__()
        self.time = HarmonicTime(32)
        self.layer1,self.layer2 = TemporalAttention(),TemporalAttention()
        self.classifier = nn.Sequential(nn.Linear(32+wallet_dim,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,1))
    def aggregate(self,layer,source,neighbor,direction,delta):
        edge = torch.zeros((*direction.shape,32),device=source.device)
        edge[...,0] = (direction==1).float()
        edge[...,1] = (direction==-1).float()
        return layer(source,neighbor,edge,self.time(delta),self.time(torch.zeros(len(source),device=source.device)),direction==0)
    def forward(self,wallet,root_direction,root_delta,child_direction,child_delta):
        b,k = root_direction.shape
        zero = torch.zeros((b,32),device=wallet.device)
        root1 = self.aggregate(self.layer1,zero,zero[:,None].expand(-1,k,-1),root_direction,root_delta)
        cd,ct = child_direction.reshape(b*k,k),child_delta.reshape(b*k,k)
        childzero = torch.zeros((b*k,32),device=wallet.device)
        child1 = self.aggregate(self.layer1,childzero,childzero[:,None].expand(-1,k,-1),cd,ct).reshape(b,k,32)
        root2 = self.aggregate(self.layer2,root1,child1,root_direction,root_delta)
        return self.classifier(torch.cat([wallet,root2],1)).squeeze(1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    cache=args.results_dir/'tgat_queries'
    prepare(args.data_dir,args.results_dir,cache)
    if args.prepare_only:return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    samples=pd.read_parquet(args.results_dir/'wallet_samples.parquet')
    data=np.load(cache/'temporal_queries.npz')
    train=np.flatnonzero(samples.split.eq('train'))
    val=np.flatnonzero(samples.split.eq('validation'))
    test=np.flatnonzero(samples.split.eq('test'))
    y=samples.label.to_numpy(np.float32)
    scaler=StandardScaler().fit(samples.iloc[train][ACTIVITY+SNA])
    wallet=scaler.transform(samples[ACTIVITY+SNA]).astype(np.float32)
    def batch(indices):
        ci=data['child_index'][indices]
        rt=data['root_time'][indices]
        child_time=data['child_time'][ci]
        return [torch.from_numpy(x).to(device) for x in [wallet[indices],data['root_direction'][indices],
            (samples.cutoff.to_numpy()[indices,None]-rt).astype(np.float32),data['child_direction'][ci],
            (rt[:,:,None]-child_time).astype(np.float32)]]
    model=CausalTGAT(len(ACTIVITY+SNA)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y[train]==0).sum()/(y[train]==1).sum()),device=device))
    rng=np.random.default_rng(args.seed)
    def predict(indices):
        model.eval();out=[]
        with torch.no_grad():
            for i in range(0,len(indices),512):out.append(torch.sigmoid(model(*batch(indices[i:i+512]))).cpu().numpy())
        return np.concatenate(out)
    best=-np.inf;state=None;stale=0;history=[]
    for epoch in range(1,31):
        model.train();order=rng.permutation(train);cumulative=0.
        for i in range(0,len(order),512):
            indices=order[i:i+512]
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(*batch(indices)),torch.from_numpy(y[indices]).to(device))
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
            cumulative+=float(loss.item())*len(indices)
        score=predict(val);ap=float(average_precision_score(y[val],score))
        history.append({'epoch':epoch,'loss':cumulative/len(train),'validation_ap':ap})
        print(f'TGAT seed{args.seed} epoch{epoch} valAP={ap:.6f}',flush=True)
        if ap>best+1e-5:best=ap;state=copy.deepcopy(model.state_dict());stale=0
        else:
            stale+=1
            if stale>=6:break
    model.load_state_dict(state)
    vs,ts=predict(val),predict(test)
    threshold=select_f1_threshold(y[val],vs)
    config={'seed':args.seed,'dim':32,'heads':4,'layers':2,'neighbors_per_hop':16,'batch':512,'lr':8e-4,'epochs':30,'patience':6,'features':ACTIVITY+SNA}
    result={'model':'TGAT (causal adaptation)','seed':args.seed,**evaluate(y[test],ts,threshold),
            **prediction_diagnostics(y[test],ts),'validation_ap':best,'epochs':len(history),
            'parameter_count':sum(p.numel() for p in model.parameters()),'elapsed_seconds':time.perf_counter()-started,
            'run_timestamp':utc_now(),'configuration_hash':config_hash(config)}
    pd.DataFrame([result]).to_csv(args.output_dir/'metrics.csv',index=False)
    pd.DataFrame(history).to_csv(args.output_dir/'training_history.csv',index=False)
    for name,indices,score in [('validation',val,vs),('test',test,ts)]:
        frame=samples.iloc[indices][['address','label','first_step','cutoff']].copy()
        frame['score'],frame['threshold']=score,threshold
        frame.to_parquet(args.output_dir/f'{name}_predictions.parquet',index=False)
    torch.save(state,args.output_dir/'model_state.pt')
    joblib.dump(scaler,args.output_dir/'preprocessing.joblib')
    save_json(args.output_dir/'manifest.json',{'config':config,'environment':environment(),
        'result':result,'sampler_tests':sampler_tests(),'script_sha256':sha256(Path(__file__)),
        'scope':'two-hop harmonic temporal attention with zero identity-free node inputs and directional edge inputs; wallet-level classification head adaptation',
        'initialization':'PyTorch defaults; learned harmonic frequencies initialized log-uniform from 1 to 1e-9, phase0'})
    print(result,flush=True)

if __name__=='__main__':main()
