"""Reviewer: scan all 1206 MAST shots through cd.mast_shot_inputs; NaN/geometry/frames report."""
import sys, json, time, numpy as np, torch
sys.path.insert(0, '/gpfs/radev/scratch/lu_lu/ss5235/Fusion')
from fusion import c2_fields_data as cd
setup = cd.build_machine_setup('MAST')
feat = cd.FrameFeaturizer(setup, torch.device('cpu'))
paths = cd.mast_parquet_paths()[::int(sys.argv[1]) if len(sys.argv) > 1 else 1]
print('shots', len(paths), 'table_names', setup.table_names, 'L', setup.L, flush=True)
tot = 0; errs = []; nan_cur = 0; nan_ip = 0; nan_tf = 0; nan_pe = 0; unobs = {}; ipneg = 0; tfpos = 0; ipobs_bad = 0; pe_valid = []; nonfinite_channels = 0; nonfinite_phi_ext = 0
ipmin = []; ipmax = []; t0 = time.time(); ch_sum = None; ch_sq = None; nfr = 0; cond_sum = None; cond_sq = None
for i, p in enumerate(paths):
    try:
        s = cd.mast_shot_inputs(p, setup)
    except Exception as e:
        errs.append((p.name, repr(e)[:200])); continue
    tot += s.n_frames
    nan_cur += int(~np.isfinite(s.currents).all()); nan_ip += int(~np.isfinite(s.Ip_kA).all()); nan_tf += int(~np.isfinite(s.tf_kA).all()); nan_pe += int(~np.isfinite(s.pe[s.pe_mask]).all())
    for k, v in s.meta['unobserved_current_frames'].items(): unobs[k] = unobs.get(k, 0) + v
    ipneg += int(s.ip_sign < 0); tfpos += int(np.median(s.tf_kA) > 0); ipobs_bad += int((~s.Ip_observed).sum()); pe_valid.append(s.pe_mask.mean())
    ipmin.append(s.Ip_kA.min()); ipmax.append(s.Ip_kA.max())
    store = cd.FrameStore([s], with_labels=False); st = store.to_torch(torch.device('cpu'), False)
    out = feat(cd.batch_from_store(st, torch.arange(store.n)))
    nonfinite_channels += int(~torch.isfinite(out['channels']).all()); nonfinite_phi_ext += int(~torch.isfinite(out['phi_ext']).all())
    c = out['channels'].flatten(2); m = c.mean(2).sum(0); q = c.pow(2).mean(2).sum(0)
    ch_sum = m if ch_sum is None else ch_sum + m; ch_sq = q if ch_sq is None else ch_sq + q
    cm = out['cond'].sum(0); cq = out['cond'].pow(2).sum(0)
    cond_sum = cm if cond_sum is None else cond_sum + cm; cond_sq = cq if cond_sq is None else cond_sq + cq
    nfr += store.n
    if (i + 1) % 200 == 0: print(i + 1, tot, f'{time.time()-t0:.0f}s', flush=True)
print(json.dumps({'shots_ok': len(paths) - len(errs), 'errors': errs[:10], 'n_errors': len(errs), 'total_frames': tot,
  'shots_with_nan_currents': nan_cur, 'shots_with_nan_ip': nan_ip, 'shots_with_nan_tf': nan_tf, 'shots_with_nan_pe_masked': nan_pe,
  'unobserved_current_frames': unobs, 'ip_sign_negative_shots': ipneg, 'tf_median_positive_shots': tfpos, 'ip_unobserved_frames': ipobs_bad,
  'pe_valid_frac_mean': float(np.mean(pe_valid)), 'pe_valid_frac_min': float(np.min(pe_valid)), 'ip_min_quantiles': np.quantile(ipmin, [0, .05, .5]).tolist(), 'ip_max_quantiles': np.quantile(ipmax, [0, .5, 1]).tolist(),
  'shots_nonfinite_channels': nonfinite_channels, 'shots_nonfinite_phi_ext': nonfinite_phi_ext,
  'channel_mean': dict(zip(cd.CHANNEL_NAMES, (ch_sum / nfr).tolist())), 'channel_rms': dict(zip(cd.CHANNEL_NAMES, (ch_sq / nfr).sqrt().tolist())),
  'cond_mean': dict(zip(cd.COND_NAMES, (cond_sum / nfr).tolist())), 'cond_rms': dict(zip(cd.COND_NAMES, (cond_sq / nfr).sqrt().tolist())),
  'seconds': time.time() - t0}, indent=1), flush=True)
