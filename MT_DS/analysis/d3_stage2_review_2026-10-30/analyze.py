"""Review existing experiment records; never load checkpoints or run inference."""
import csv
import hashlib
import json
from pathlib import Path
import runpy
import statistics as st

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OTHER = PROJECT.parents[1] / "MT_DS_LossNorm_Channel/analysis/channel_interaction/D2_E3_LU_interaction"
CONDITIONS = dict(baseline="", m0="M0_D3_readout_v1", d3="D3_boundary_skip_w0015_v1",
                  random="D3_L2U_random_v1", boundary="D3_L2U_boundary_v1",
                  band03="A1_boundary_supervision_w003_v1", dual03="A1_independent_boundary_head_w003_v1",
                  e4="E4_legacy_boundary_w003_v1")
CORE = ("baseline", "m0", "d3", "random", "boundary")
DATASETS = ("LA_8l", "Pancreas_6l")
METRICS = ("dice", "jc", "hd95", "asd")
PHASES = ((1, 2000), (2001, 4000), (4001, 5000), (5001, 8000), (8001, 12000), (12001, 15000))


def js(p):
    return json.loads(p.read_text())


def jl(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def read_csv(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def stats(values):
    x = [float(v) for v in values if v is not None and v != ""]
    if not x:
        return dict(n=0)
    assert np.isfinite(x).all()
    return dict(n=len(x), mean=st.mean(x), median=st.median(x), min=min(x), max=max(x),
                q05=float(np.quantile(x, .05)), q95=float(np.quantile(x, .95)))


def csv_out(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def paired_cases(a, b):
    before = {r["case"]: r for r in b["per_case"]}
    rows = [dict(case=r["case"], **{m:r[m]-before[r["case"]][m] for m in METRICS}) for r in a["per_case"]]
    return dict(metrics={m:dict(stats(r[m] for r in rows),
                               better=sum(r[m]*(1 if m in ("dice","jc") else -1)>0 for r in rows),
                               worse=sum(r[m]*(1 if m in ("dice","jc") else -1)<0 for r in rows))
                         for m in METRICS},
                worst_dice=sorted(rows,key=lambda r:r["dice"])[:4],
                best_dice=sorted(rows,key=lambda r:r["dice"],reverse=True)[:4],
                worst_hd95=sorted(rows,key=lambda r:r["hd95"],reverse=True)[:3])


def sign_summary(values, tolerance):
    x=list(values)
    return dict(stats(x), tolerance=tolerance,
                meaningful_positive=sum(v>tolerance for v in x),
                meaningful_negative=sum(v<-tolerance for v in x),
                within_tolerance=sum(abs(v)<=tolerance for v in x))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    probe_summary=runpy.run_path(str(PROJECT/'analysis/e4_d3_review_2026-10-29/analyze.py'))['summarize_probes']
    report=dict(scope="saved training records only; no weight loading, new inference, or training",
                datasets={}, other_project={}, limitations=[
                    "single seed42 and repeated historical test-list online model selection",
                    "validation timepoints are correlated, not independent replications",
                    "voxel HD95 and directed ASD; boundary-band F1 is not surface Dice",
                    "D3 random Pancreas resumed at step4000 without sampler/worker-prefetch state",
                    "D2/E3 uses LossNorm/keyed data/strict FP32 and has a separate none control"])
    loaded, metric_rows, curve_rows, phase_rows, case_rows, readout_rows = {}, [], [], [], [], []
    train_rows, transfer_rows, injection_rows = [], [], []
    expected_val=list(range(800,15001,200))
    expected_train=[1,*range(10,15001,10)]
    main_config=("seed","labelnum","batch_size","labeled_bs","max_iterations","warmup_iterations",
                 "base_lr","ema_decay","consistency","num_workers","cpu_threads","patch_size",
                 "val_interval","val_start","normalization","teacher_bn","consistency_loss")
    for dataset in DATASETS:
        runs={}
        for condition,folder in CONDITIONS.items():
            path=PROJECT/'Results'/folder/'seed_42'/dataset
            run=dict(path=path, config=js(path/'config.json'), state=js(path/'run_status.json'),
                     best=js(path/'best_metrics.json'), test=js(path/'test_metrics.json'),
                     val=jl(path/'validation_metrics.jsonl'), train=read_csv(path/'training_metrics.csv'))
            assert run['state']['state']=='completed' and run['state']['iteration']==15000
            assert [r['iteration'] for r in run['val']]==expected_val
            assert [int(r['iteration']) for r in run['train']]==expected_train
            assert all(np.isfinite(float(r['loss_total'])) for r in run['train'])
            assert all(r['cases']==20 for r in run['val'])
            assert run['best']['dice']==max(r['dice'] for r in run['val'])
            assert all(run['best'][m]==run['test'][m] for m in METRICS)
            assert run['best']['surface_metric_cases']==20
            assert all(abs(st.mean(c[m] for c in run['best']['per_case'])-run['best'][m])<1e-12 for m in METRICS)
            if condition in ('m0','d3','random','boundary'):
                run.update(probes=jl(path/'channel_interventions.jsonl'),readout=jl(path/'boundary_validation_metrics.jsonl'))
                assert [r['iteration'] for r in run['probes']]==[1,*range(200,15001,200)]
                assert [r['iteration'] for r in run['readout']]==expected_val
                assert all(digest(path/key)==h for key,h in js(path/'source_manifest.json').items())
            if condition in ('m0','random','boundary'):
                run['routes']=jl(path/'batch_routes.jsonl')
                assert [r['iteration'] for r in run['routes']]==expected_train
            if condition in ('random','boundary'):
                run['injections']=jl(path/'labeled_injection_checks.jsonl')
                assert [r['iteration'] for r in run['injections']]==list(range(4200,15001,200))
                assert run['config']['boundary_weight']==.015
                assert (run['config']['boundary_start'],run['config']['boundary_ramp'])==(2000,2000)
                assert (run['config']['interaction_start'],run['config']['interaction_ramp'])==(4000,1000)
            runs[condition]=run
        loaded[dataset]=runs
        base=runs['baseline']; reference=runs['d3']
        audit={}
        for condition,run in runs.items():
            assert js(run['path']/'data_split_manifest.json')==js(base['path']/'data_split_manifest.json')
            assert all(run['config'].get(k)==base['config'].get(k) for k in main_config)
            assert all(digest(run['path']/rel)==digest(base['path']/rel) for rel in (
                'code/evaluate.py','code/objectives.py','code/networks/vnet_ds.py'))
            best,last=run['best'],run['val'][-1]
            metric=dict(dataset=dataset,condition=condition,best_step=best['iteration'])
            for m in METRICS:
                metric.update({f'best_{m}':best[m],f'last_{m}':last[m],
                               f'best_minus_baseline_{m}':best[m]-base['best'][m],
                               f'best_minus_d3_{m}':best[m]-reference['best'][m]})
            metric_rows.append(metric)
            a=dict(metrics=metric, best_cases_vs_baseline=paired_cases(best,base['best']),
                   best_cases_vs_d3=paired_cases(best,reference['best']),phases={},
                   audit=dict(completed=True,validation_records=72,train_records=1501,
                              best_retest_identical=True,source_and_split_and_main_protocol_valid=True,
                              surface_incomplete_steps=[r['iteration'] for r in run['val'] if r['surface_metric_cases']<20]))
            for receiver_ref in ('baseline','d3'):
                bcases={r['case']:r for r in runs[receiver_ref]['best']['per_case']}
                for c in best['per_case']:
                    case_rows.append(dict(dataset=dataset,condition=condition,reference=receiver_ref,case=c['case'],
                                          **{f'delta_{m}':c[m]-bcases[c['case']][m] for m in METRICS}))
            base_map={r['iteration']:r for r in base['val']}
            d3_map={r['iteration']:r for r in reference['val']}
            for r in run['val']:
                curve_rows.append(dict(dataset=dataset,condition=condition,iteration=r['iteration'],
                                       surface_metric_cases=r['surface_metric_cases'],**{m:r[m] for m in METRICS}))
            for lo,hi in PHASES:
                val=[r for r in run['val'] if lo<=r['iteration']<=hi]
                tr=[r for r in run['train'] if lo<=int(r['iteration'])<=hi]
                p=dict(validation={m:dict(stats(r[m] for r in val),
                    better_baseline=sum((r[m]-base_map[r['iteration']][m])*(1 if m in ('dice','jc') else -1)>0 for r in val),
                    better_d3=sum((r[m]-d3_map[r['iteration']][m])*(1 if m in ('dice','jc') else -1)>0 for r in val)) for m in METRICS},training={})
                for m in ('loss_supervised','loss_consistency','loss_boundary','gradient_norm','boundary_head_gradient_norm','backbone_gradient_norm','transfer_gradient_norm'):
                    if m in tr[0]: p['training'][m]=stats(r[m] for r in tr)
                a['phases'][f'{lo}_{hi}']=p
                phase_rows.append(dict(dataset=dataset,condition=condition,lo=lo,hi=hi,**{f'mean_{m}':p['validation'][m]['mean'] for m in METRICS},
                                       better_baseline_dice=p['validation']['dice']['better_baseline'],better_d3_dice=p['validation']['dice']['better_d3']))
                train_rows.append(dict(dataset=dataset,condition=condition,lo=lo,hi=hi,**{m:v['mean'] for m,v in p['training'].items()}))
            if 'readout' in run:
                a['probes_mature']=probe_summary(run['probes'],4000,15000)
                a['readout_final']={k:v for k,v in run['readout'][-1].items() if k!='per_case'}
                a['readout_at_main_best']={k:v for k,v in next(r for r in run['readout'] if r['iteration']==best['iteration']).items() if k!='per_case'}
                a['readout_late']={k:stats(r[k] for r in run['readout'] if r['iteration']>12000) for k in ('f1','loss','precision','recall')}
                for r in run['readout']:
                    readout_rows.append(dict(dataset=dataset,condition=condition,**{k:r[k] for k in ('iteration','loss','f1','precision','recall')}))
            if condition=='m0':
                a['baseline_validation_parity']={m:dict(max_abs_difference=max(abs(r[m]-b[m]) for r,b in zip(run['val'],base['val'])),identical_points=sum(r[m]==b[m] for r,b in zip(run['val'],base['val']))) for m in METRICS}
                a['baseline_training_parity']={m:max(abs(float(r[m])-float(b[m])) for r,b in zip(run['train'],base['train'])) for m in ('loss_supervised','loss_consistency','ce','dice')}
            if condition in ('random','boundary'):
                before=[(r,d) for r,d in zip(run['train'],reference['train']) if int(r['iteration'])<=4000]
                a['pre_interaction_parity']={k:max(abs(float(r[k])-float(d[k])) for r,d in before) for k in ('loss_supervised','loss_consistency','loss_boundary','gradient_norm','boundary_band_fraction')}
                a['interaction']={}
                for lo,hi in ((4001,15000),(5001,15000),(12001,15000)):
                    tr=[r for r in run['train'] if lo<=int(r['iteration'])<=hi]
                    rt=[r for r in run['routes'] if lo<=r['iteration']<=hi]
                    a['interaction'][f'{lo}_{hi}']=dict(
                        sampled_steps=len(tr),sampled_active_steps=sum(int(r['interaction_active_samples'])>0 for r in tr),
                        active_sample_fraction=st.mean(int(r['interaction_active_samples'])/2 for r in tr),
                        valid_donor_fraction=st.mean(int(r['interaction_valid_donors'])/2 for r in tr),
                        mask_mean=stats(r['interaction_mask_mean'] for r in tr),
                        residual_ratio_per_sample=stats(v for r in rt for v in r['route']['relative_residual_norm']),
                        residual_ratio_mean=stats(r['interaction_residual_ratio_mean'] for r in tr),
                        residual_ratio_max=stats(r['interaction_residual_ratio_max'] for r in tr),
                        transfer_gradient=stats(r['transfer_gradient_norm'] for r in tr),
                        transfer_to_backbone_gradient=stats(float(r['transfer_gradient_norm'])/max(float(r['backbone_gradient_norm']),1e-20) for r in tr),
                        consistency_difference=sign_summary((float(r['loss_consistency_interacted'])-float(r['loss_consistency_clean']) for r in tr),1e-8),
                        abs_relative_consistency_difference=stats(abs(float(r['loss_consistency_interacted'])-float(r['loss_consistency_clean']))/max(float(r['loss_consistency_clean']),1e-20) for r in tr),
                        selected_score_mean=stats(st.mean(v for side in r['route']['selected_scores'] for v in side) for r in rt))
                for r in run['train']:
                    transfer_rows.append(dict(dataset=dataset,condition=condition,**{k:r[k] for k in ('iteration','interaction_active_samples','interaction_valid_donors','interaction_mask_mean','interaction_residual_ratio_mean','interaction_residual_ratio_max','transfer_gradient_norm','backbone_gradient_norm','loss_consistency_clean','loss_consistency_interacted')}))
                a['labeled_injection']={}
                for lo,hi in ((4001,15000),(12001,15000)):
                    inj=[r for r in run['injections'] if lo<=r['iteration']<=hi]
                    a['labeled_injection'][f'{lo}_{hi}']=dict(count=len(inj),
                        active=sum(r['interaction_active_samples']>0 for r in inj),
                        changes={k:sign_summary((r['actual_delta'][k] for r in inj),1e-5 if k=='patch_dice' else 1e-6) for k in inj[0]['actual_delta']})
                for r in run['injections']:
                    injection_rows.append(dict(dataset=dataset,condition=condition,iteration=r['iteration'],
                        residual_ratio=r['interaction_residual_ratio_mean'],**{f'delta_{k}':v for k,v in r['actual_delta'].items()}))
            audit[condition]=a
        # Directly inspect the saved target/case traces around the Pancreas resume.
        audit['trace_comparison']={}
        for condition in ('random','boundary'):
            r=runs[condition]; m0=runs['m0']
            for lo,hi in ((1,4000),(4001,15000)):
                pairs=[(x,y) for x,y in zip(r['routes'],m0['routes']) if lo<=x['iteration']<=hi]
                training_pairs=[(x,y) for x,y in zip(r['train'],m0['train']) if lo<=int(x['iteration'])<=hi]
                audit['trace_comparison'][f'{condition}_{lo}_{hi}']=dict(
                    logged_batches=len(pairs),same_L_and_U_ID_batches=sum(x['labeled_case_ids']==y['labeled_case_ids'] and x['unlabeled_case_ids']==y['unlabeled_case_ids'] for x,y in pairs),
                    max_boundary_mass_difference=max(abs(float(x['boundary_positive_mass'])-float(y['boundary_positive_mass'])) for x,y in training_pairs))
        report['datasets'][dataset]=audit

    # Separate non-boundary LossNorm experiment; compare only to its own none.
    d2_loaded={}
    d2_metrics, d2_cases, d2_curve_rows = [], [], []
    final=js(OTHER/'FINAL_STATUS.json')
    assert final['status']=='COMPLETE'
    for dataset in ('LA','Pancreas'):
        modes={}
        for mode in ('none','random','gradient'):
            p=OTHER/'results/training/seed_42/run_01'/f'{dataset}_{mode}'
            v=jl(p/'validation_metrics.jsonl'); tr=read_csv(p/'training_metrics.csv'); best=js(p/'best_metrics.json')
            assert [r['iteration'] for r in v]==expected_val and len(tr)==15000
            assert best['dice']==max(r['dice'] for r in v)
            verified=js(p/'COMPLETION_VERIFICATION.json')
            assert verified['training_complete_verified'] and verified['recorded_optimizer_steps']==15000
            modes[mode]=dict(path=p,val=v,train=tr,best=best,verified=verified)
        d2_loaded[dataset]=modes
        report['other_project'][dataset]={}
        none=modes['none']
        for mode,r in modes.items():
            last=r['val'][-1]; late=r['val'][-10:]
            info=dict(best={k:r['best'][k] for k in (*METRICS,'iteration')},
                      final={k:last[k] for k in (*METRICS,'iteration')},
                      final_delta_none={m:last[m]-none['val'][-1][m] for m in METRICS},
                      best_delta_none={m:r['best'][m]-none['best'][m] for m in METRICS},
                      last10={m:stats(x[m] for x in late) for m in METRICS},
                      best_cases_vs_none=paired_cases(r['best'],none['best']),
                      final_cases_vs_none=paired_cases(last,none['val'][-1]),
                      completion_status_recovery=r['verified'].get('completion_status_recovery'),
                      completion_error_scope=r['verified'].get('error_scope'))
            info['incomplete_surface_records']=[dict(iteration=x['iteration'],dice=x['dice'],
                surface_cases=sum(c['hd95'] is not None for c in x['per_case']),
                zero_dice_cases=[c['case'] for c in x['per_case'] if c['dice']==0])
                for x in r['val'] if any(c['hd95'] is None for c in x['per_case'])]
            info['last10_minimum']=dict(iteration=min(late,key=lambda x:x['dice'])['iteration'],
                                       dice=min(x['dice'] for x in late))
            d2_metrics.append(dict(dataset=dataset,mode=mode,best_step=r['best']['iteration'],
                **{f'best_{m}':r['best'][m] for m in METRICS},**{f'last_{m}':last[m] for m in METRICS},
                **{f'last10_{m}':info['last10'][m]['mean'] for m in METRICS}))
            for point,metrics in (('best',r['best']),('last',last)):
                refs={c['case']:c for c in (none['best'] if point=='best' else none['val'][-1])['per_case']}
                for case in metrics['per_case']:
                    d2_cases.append(dict(dataset=dataset,mode=mode,point=point,case=case['case'],
                        **{f'delta_{m}':case[m]-refs[case['case']][m] for m in METRICS}))
            for x in r['val']:
                d2_curve_rows.append(dict(dataset=dataset,mode=mode,iteration=x['iteration'],
                    surface_cases=sum(c['hd95'] is not None for c in x['per_case']),**{m:x[m] for m in METRICS}))
            info['phases']={}
            for lo,hi in PHASES:
                values=[x for x in r['val'] if lo<=x['iteration']<=hi]
                info['phases'][f'{lo}_{hi}']={m:stats(x[m] for x in values) for m in METRICS}
            joint=[x for x in r['train'] if int(x['iteration'])>2000]
            info['interaction']=dict(steps=len(joint),
                selected=stats(x['selected_count'] for x in joint),
                feature_relative_change=stats(x['E3_relative_feature_change'] for x in joint))
            if mode!='none':
                info['interaction']['reference_vs_actual_mse_relative_change']=stats(abs(float(x['loss_mt'])-float(x['reference_loss_mt']))/max(float(x['reference_loss_mt']),1e-20) for x in joint)
                channels=read_csv(r['path']/'channel_records.csv')
                selected=[x for x in channels if x['selected']=='True']
                info['selected_score']=stats(x['score'] for x in selected)
                info['selected_cosine']=stats(x['cosine'] for x in selected)
                info['positive_selected_fraction']=st.mean(float(x['score'])>0 for x in selected)
            # Keyed batch/augmentation hashes allow exact protocol comparisons.
            info['batch_hash_matches_none']=sum(x['batch_hash']==y['batch_hash'] for x,y in zip(r['train'],none['train']))
            report['other_project'][dataset][mode]=info
    for name,rows in [('metrics.csv',metric_rows),('validation_curves.csv',curve_rows),('validation_phases.csv',phase_rows),
                      ('case_deltas.csv',case_rows),('readout_curves.csv',readout_rows),('training_phases.csv',train_rows),
                      ('interaction_training.csv',transfer_rows),('labeled_injection.csv',injection_rows)]:
        csv_out(name,rows)
    for name,rows in (('separate_lossnorm_E3_metrics.csv',d2_metrics),('separate_lossnorm_E3_case_deltas.csv',d2_cases),
                      ('separate_lossnorm_E3_validation_curves.csv',d2_curve_rows)):
        csv_out(name,rows)
    (OUT/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    plots(loaded,d2_loaded)
    print(json.dumps(dict(status='PASS',output=str(OUT),main_core_metrics=[r for r in metric_rows if r['condition'] in CORE]),indent=2))


def plots(loaded,d2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(baseline='#333333',m0='#a7a7a7',d3='#cc8b21',random='#4777bd',boundary='#ae3067')
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for i,dataset in enumerate(DATASETS):
        for j,key in enumerate(('dice','hd95','asd')):
            ax=axes[i,j]
            for condition in ('baseline','d3','random','boundary'):
                values=loaded[dataset][condition]['val']
                ax.plot([x['iteration'] for x in values],[x[key]*(100 if key=='dice' else 1) for x in values],label=condition,color=colors[condition],lw=1.2)
            ax.axvline(4000,color='#777777',ls='--',lw=.7);ax.grid(alpha=.2)
            ax.set_title(f'{dataset}: {key}');ax.set_xlabel('Training step');ax.set_ylabel('%' if key=='dice' else 'voxel');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'validation_curves.png',dpi=170);fig.savefig(OUT/'validation_curves.pdf')
    for ax in axes.flat:
        ax.set_xlim(10000,15000)
        data=[line.get_ydata()[np.array(line.get_xdata())>=10000] for line in ax.lines if len(line.get_xdata())>2]
        lo,hi=min(min(x) for x in data),max(max(x) for x in data);pad=max((hi-lo)*.08,.02);ax.set_ylim(lo-pad,hi+pad)
    fig.savefig(OUT/'late_validation_curves.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for i,dataset in enumerate(DATASETS):
        for mode in ('random','boundary'):
            rows=[r for r in loaded[dataset][mode]['train'] if int(r['iteration'])>4000]
            steps=np.array([int(r['iteration']) for r in rows])
            ratio=np.array([float(r['interaction_residual_ratio_mean']) for r in rows])
            delta=np.array([abs(float(r['loss_consistency_interacted'])-float(r['loss_consistency_clean']))/max(float(r['loss_consistency_clean']),1e-20) for r in rows])
            axes[i,0].semilogy(steps,ratio,label=mode,color=colors[mode],alpha=.75,lw=.9)
            axes[i,1].semilogy(steps,np.maximum(delta,1e-10),label=mode,color=colors[mode],alpha=.75,lw=.9)
            inj=loaded[dataset][mode]['injections']
            axes[i,2].scatter([r['iteration'] for r in inj],[r['actual_delta']['final_boundary_ce'] for r in inj],label=mode,color=colors[mode],s=14,alpha=.75)
        axes[i,0].axhline(.1,color='black',ls='--',lw=.8,label='configured cap .1')
        axes[i,0].set_ylabel('residual / configured reference norm')
        axes[i,1].set_ylabel('|MSE(interact)-MSE(clean)| / MSE(clean)')
        axes[i,2].axhline(0,color='black',lw=.7);axes[i,2].set_ylabel('Actual labeled receiver boundary-CE delta')
        for j in range(3): axes[i,j].set_title(f'{dataset}: '+('actual amplitude','U consistency effect','L-to-L diagnostic')[j]);axes[i,j].set_xlabel('Training step');axes[i,j].grid(alpha=.2);axes[i,j].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'interaction_strength_and_effect.png',dpi=170);fig.savefig(OUT/'interaction_strength_and_effect.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for i,dataset in enumerate(DATASETS):
        for mode in ('m0','d3','random','boundary'):
            values=loaded[dataset][mode]['readout']
            for j,key in enumerate(('f1','loss')):
                axes[i,j].plot([r['iteration'] for r in values],[r[key] for r in values],color=colors[mode],label=mode)
        for j in range(2): axes[i,j].set_title(f'{dataset}: readout '+('F1','loss')[j]);axes[i,j].set_xlabel('Training step');axes[i,j].grid(alpha=.2);axes[i,j].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'readout_learning.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for i,dataset in enumerate(('LA','Pancreas')):
        for j,key in enumerate(('dice','hd95','asd')):
            ax=axes[i,j]
            for mode,color in (('none','#333333'),('random','#4777bd'),('gradient','#ae3067')):
                values=d2[dataset][mode]['val']
                ax.plot([r['iteration'] for r in values],[r[key]*(100 if key=='dice' else 1) for r in values],color=color,label=mode,lw=1.1)
            ax.set_xlim(10000,15000);ax.set_title(f'LossNorm E3 / {dataset}: {key}');ax.set_xlabel('Training step');ax.grid(alpha=.2);ax.legend(fontsize=8)
            values=[line.get_ydata()[np.array(line.get_xdata())>=10000] for line in ax.lines]
            lo,hi=min(min(x) for x in values),max(max(x) for x in values);pad=max((hi-lo)*.08,.02);ax.set_ylim(lo-pad,hi+pad)
    fig.tight_layout();fig.savefig(OUT/'separate_lossnorm_E3_late_curves.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    main()
