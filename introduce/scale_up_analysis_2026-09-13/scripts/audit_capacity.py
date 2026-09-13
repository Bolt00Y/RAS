#!/usr/bin/env python3
"""Static capacity audit; no TensorFlow import, graph, profiler or training.

Run from any directory. Exact means arithmetic exact for the current source and
args only, never historical-run attestation. FLOPs are explicitly estimates.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / 'data'
MODELS = ROOT / 'src/models/rankmixer'


def args_file(path):
    values = {}
    for line in path.read_text().splitlines():
        if line.startswith('--') and '=' in line:
            key, value = line[2:].split('=', 1)
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            values[key] = value
    return values, json.loads(values['model_args'])


def static_source_counter(source_path, kwargs):
    """Extract only data assignments and a pure arithmetic counting method.

    No constructor, imports, TF operations, feature reader or graph verifier runs.
    Initializer expressions that need framework objects are deliberately skipped.
    """
    tree = ast.parse(source_path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MLPModel')
    env = {'_ids': lambda text: tuple(text.split())}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), 'exec'), env)
            except (NameError, AttributeError, TypeError):
                pass
    obj = SimpleNamespace(**kwargs)
    for node in cls.body:
        if isinstance(node, ast.Assign):
            try:
                value = eval(compile(ast.Expression(node.value), str(source_path), 'eval'), env)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        setattr(obj, target.id, value)
            except (NameError, AttributeError, TypeError):
                pass
    env.update(self=obj, _kwargs=kwargs)
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    for node in init.body:
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'self' for t in node.targets):
            # Exclude framework calls before evaluating. Only basic configuration
            # getters/casts/comprehensions and arithmetic are needed here.
            calls = [n for n in ast.walk(node.value) if isinstance(n, ast.Call)]
            def allowed(call):
                return (isinstance(call.func, ast.Name) and call.func.id in {'int', 'float', 'bool', 'str', 'len', 'tuple', 'list'}) or (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == '_kwargs' and call.func.attr == 'get')
            if not all(allowed(c) for c in calls):
                continue
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), 'exec'), env)
            except (NameError, AttributeError, TypeError):
                pass
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_calculate_dense_trainable_params')
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), 'exec'), env)
    count = env['_calculate_dense_trainable_params'](obj)
    total = count['total'] if isinstance(count, dict) else count
    expected = getattr(obj, '_EXPECTED_DENSE_TRAINABLE_PARAMS', None)
    if expected is not None:
        assert total == expected, (source_path.name, total, expected)
    return total, count, obj, method.lineno


def e2_budget(d, layers, hidden=704, compact=False):
    """Generalized E2 algebra; these custom shapes are rejected by shipped guards."""
    t, n, e = 32, 31, 20978
    block = layers * (2*t*(3*d*hidden+2*hidden+d)+2*t*d)
    head_dims = [d,256,128,1] if compact else [t*d,2048,2048,256,1]
    head = sum(a*b+b+(2*b if i < len(head_dims)-2 else 0) for i,(a,b) in enumerate(zip(head_dims,head_dims[1:])))
    final_norm = 2*d if compact else t*d
    total = 41956+522112+(e*d+2*n*d)+(e*d+d*d+3*d)+block+final_norm+head
    # Existing repository's arithmetic convention: Dense MAC=2, SiLU=2,
    # gate product=1, RMSNorm=4D+2, GELU2=9, inference BN=4 per element.
    flops = 83912+1087414 + (2*e*d+9*n*d+n*(4*d+2)) + (2*e*d+2*d*d+13*d+2)
    flops += layers * (2*t*(6*d*hidden+3*hidden+4*d+2)+2*t*d)
    flops += t*(8*d+2)+t*d if compact else t*(4*d+2)
    flops += sum(2*a*b+(13*b if i < len(head_dims)-2 else 1) for i,(a,b) in enumerate(zip(head_dims,head_dims[1:])))
    return {'total':total,'blocks':block,'head':head,'flops_estimate':flops}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    rows, breakdowns = [], {}
    # Old-family totals independently reconstructed from documented module
    # algebra; FLOPs retained as prior static estimates, not reprofiled.
    old = {
        'v1': (167293157,335944329,16,768,2,3072,'GELU per-token x1','mean+linear'),
        'v1_lrfix': (167293157,335944329,16,768,2,3072,'GELU per-token x1','mean+linear'),
        'v2': (95809126,192547107,16,768,2,1536,'GELU per-token x1','gated pool+bucket cross'),
        'v3': (95809126,192547107,16,768,2,1536,'GELU per-token x1','gated pool+bucket cross'),
        'v4': (96439272,193818155,16,768,2,1536,'GELU per-token x1','v3+QI cross'),
        'v5': (348432486,705277790,32,1024,2,704,'SwiGLU per-token x2','global pool+gated flat+deep head'),
        'v6': (177217126,358638942,32,512,2,704,'SwiGLU per-token x2','global pool+gated flat+deep head'),
        'v7': (102113126,205185571,16,768,2,1536,'GELU per-token x1','v3+deep head'),
    }
    for name, (total,f,t,d,l,m,ffn,readout) in old.items():
        if name in {'v1','v1_lrfix','v2','v3','v4','v7'}:
            k=4 if name in {'v1','v1_lrfix'} else 2
            block=l*(t*(2*k*d*d+(k+1)*d)+(6*d if k==4 else 4*d))
            calc=41956+20978*d+t*d+block+(d+1)
            if k==2: calc+=522112+768+3541249+1536
            if name=='v4': calc+=630146
            if name=='v7': calc+=6304000
        else:
            block=l*(2*t*(3*d*m+2*m+d)+2*t*d)
            calc=41956+522112+20978*d+62*d+20978*d+d*d+3*d+block+t*d+2*(d*128+128)+(31*d*512+1025)
            dims=[2*d+512,2048,2048,256,1]
            calc+=sum(a*b+b+(2*b if i<3 else 0) for i,(a,b) in enumerate(zip(dims,dims[1:])))
        assert total==calc,(name,total,calc)
        p=MODELS/('cvr_bn_rankmixer_'+name+'.py')
        a=ROOT/'bash'/('set-rankmixer-'+name.replace('_','-')+'-args.txt')
        outer, kw=args_file(a) if a.exists() else ({},{})
        rows.append(dict(model=name,source=str(p.relative_to(ROOT)),source_sha256=sha(p),args=str(a.relative_to(ROOT)) if a.exists() else '',args_sha256=sha(a) if a.exists() else '',T_fact=t,D_fact=d,L_fact=l,M_fact=m,ffn_fact=ffn,readout_fact=readout,dense_params_static_fact=total,block_params_static_fact=block,flops_per_sample_estimate=f,flops_basis='prior_document_static_estimate',historical_run_binding='unverified',source_count_method_line='',result_status='result_present' if name!='v1_lrfix' else 'no_distinct_result',train_dates_current=outer.get('train_dates',''),test_date_current=outer.get('test_date','')))
    modern = ['v6_e2','v6_e2_small','v6_e2_small_1','v6_e2_small_2','v6_e2_small_3','v6_e3','v6_e4','v8','v9','v10']
    modern += ['mature_v1','mature_v2','mature_v3','mature_v4','mature_v5']
    for name in modern:
        mature=name.startswith('mature_')
        p=MODELS/(('cvr_senet_mature_rankmixer_'+name[-2:] if mature else 'cvr_bn_rankmixer_'+name)+'.py')
        args_name='set-rankmixer-mature-3bucket-d256-args.txt' if name=='mature_v1' else 'set-rankmixer-'+name.replace('_','-')+'-args.txt'
        a=ROOT/'bash'/args_name
        outer,kw=args_file(a)
        total, detail, obj, line=static_source_counter(p,kw)
        t,d,l,m=(obj.mixup_token_num,obj.mixup_token_dim,obj.mlp_mixer_layers,obj.mixer_hidden_dim) if mature else (obj.rm_token_num,obj.rm_hidden_dim,obj.rm_layer_num,obj.rm_swiglu_hidden_dim)
        block=(detail['mixer']-2*d) if mature else l*(2*t*(3*d*m+2*m+d)+(4*d if name in {'v6_e3','v10'} else 2*t*d))
        f=''; basis='not_audited'
        if name.startswith('v6_e2'):
            budget=e2_budget(d,l,m,compact=name.endswith(('_1','_2')))
            assert budget['total']==total
            f=budget['flops_estimate'];basis='recomputed_static_arithmetic_estimate'
        prior={'v6_e3':399355903,'v8':388878806,'v9':403287434,'v10':399355903,'mature_v1':220714000}
        if name in prior: f=prior[name];basis='prior_document_static_estimate'
        breakdowns[name]=detail
        rows.append(dict(model=name,source=str(p.relative_to(ROOT)),source_sha256=sha(p),args=str(a.relative_to(ROOT)),args_sha256=sha(a),T_fact=t,D_fact=d,L_fact=l,M_fact=m,ffn_fact=('pSiLU per-token x1' if name in {'mature_v4','mature_v5'} else 'SwiGLU per-token x1' if mature else 'SwiGLU per-token x2'),readout_fact='mean+creative+compact head' if mature else kw.get('rm_readout_type','enhanced'),dense_params_static_fact=total,block_params_static_fact=block,flops_per_sample_estimate=f,flops_basis=basis,historical_run_binding='unverified',source_count_method_line=line,result_status='no_result_found' if name in {'v10','mature_v2','mature_v3'} else 'result_present',train_dates_current=outer.get('train_dates',''),test_date_current=outer.get('test_date','')))
    # Baseline and UniMixer use separate structural formulas, not a parameter
    # counter copied from a model with a similar-looking name.
    baseline=2*20978+522112+2*(2*20978*500+500+3*20978)
    dims=[20978,2048,2048,256,1]
    baseline+=sum(a*b+b+(2*b if i<3 else 0) for i,(a,b) in enumerate(zip(dims,dims[1:])))
    assert baseline==90341785
    unimixer=sum([41956,521344,768,10757120,32768,286720,100827136,3584,38286337])
    assert unimixer==150757733
    for name,relative,args_relative,total,f,ffn,readout in [
        ('base','src/models/seq_model/cvr_bn_senet_dcnm_fst.py','bash/rankmixer_first_batch_20260814/00-e0-base-args.txt',baseline,180923051,'DCNM rank500 x2','flat+deep head'),
        ('unimixer_v1','src/models/rankmixer/cvr_bn_unimixer_v1.py','bash/set-unimixer-v1-args.txt',unimixer,'','SwiGLU per-token x1','pure flat+deep head'),
    ]:
        p=ROOT/relative;a=ROOT/args_relative;outer,kw=args_file(a)
        rows.append(dict(model=name,source=relative,source_sha256=sha(p),args=args_relative,args_sha256=sha(a),T_fact=32 if name=='unimixer_v1' else '',D_fact=512 if name=='unimixer_v1' else '',L_fact=2,M_fact=1024 if name=='unimixer_v1' else '',ffn_fact=ffn,readout_fact=readout,dense_params_static_fact=total,block_params_static_fact='',flops_per_sample_estimate=f,flops_basis='prior_document_static_estimate' if f else 'not_audited',historical_run_binding='unverified',source_count_method_line='',result_status='result_present',train_dates_current=outer.get('train_dates',''),test_date_current=outer.get('test_date','')))
    columns=list(rows[0])
    with (OUT/'model_capacity.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    (OUT/'capacity_breakdown.json').write_text(json.dumps(breakdowns,indent=2,ensure_ascii=False)+'\n')
    proposals={f'D{d}_L{l}_M{m}':e2_budget(d,l,m) for d,l,m in [(256,2,704),(256,3,704),(352,2,704),(256,4,704),(256,2,1056)]}
    (OUT/'capacity_proposals.json').write_text(json.dumps(proposals,indent=2)+'\n')
    for row in rows: print(row['model'],row['dense_params_static_fact'],row['flops_per_sample_estimate'])
    print('PROPOSALS',json.dumps(proposals))


if __name__=='__main__':
    main()
