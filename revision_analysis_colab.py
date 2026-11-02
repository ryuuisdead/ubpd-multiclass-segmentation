# ============================================================
# ANÁLISIS DE REVISIÓN — UBPD v34 (Paper 143)
# ------------------------------------------------------------
# EJECUTAR EN GOOGLE COLAB como una celda NUEVA, DESPUÉS de las celdas del
# notebook que definen: data_pairs, FOLD_SPLITS, all_fold_histories,
# build_model, get_val_transform, UBPDDataset, get_patient_id,
# _predict_loader, _predict_loader_tta, _hd95_single y las constantes
# (CHECKPOINT_DIR, RESULTS_DIR, IMAGES_DIR, ANNOTATIONS_DIR, MODEL_NAME,
#  CLASS_NAMES, EVAL_THRESHOLD, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, DEVICE).
#
# NO reentrena nada. Solo hace inferencia con los checkpoints ya guardados.
# Produce los números que piden los revisores:
#   PARTE 0  Auditoría del dataset (955/88 vs 1,055/101; etiquetas excluidas)
#   PARTE 1  Predicciones por imagen (best_ema y, si existe, swa)
#   PARTE 2  Presencia de estructuras, Dice agregado vs por paciente,
#            IC 95 % por bootstrap agrupado por paciente, HD95 convencional
#            vs HD95 con penalización, tasas de omisión / falsa alarma
#   PARTE 3  (opcional) Evaluación en una partición independiente (p. ej. test)
# ============================================================

import os, json, collections
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

N_BOOT = 5000
RNG = np.random.default_rng(42)
CLASSES = list(CLASS_NAMES)          # ['nerve', 'muscle', 'artery', 'vein']

# Opcional (PARTE 3): rutas de una partición independiente con anotaciones.
# Déjalas en None si no existe o si aún no la has verificado.
TEST_IMAGES_DIR = None        # p. ej. '/content/ubpd_dataset/test/JPEGImages'
TEST_ANNOTATIONS_DIR = None   # p. ej. '/content/ubpd_dataset/test_mask/json_test'

# ------------------------------------------------------------
# PARTE 0 — Auditoría del dataset
# ------------------------------------------------------------
print('=' * 70)
print('PARTE 0 — AUDITORÍA DEL DATASET')
print('=' * 70)
jpgs = sorted(f[:-4] for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg'))
jsons = sorted(f[:-5] for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.json'))
print(f'JPG en {IMAGES_DIR}: {len(jpgs)}')
print(f'JSON en {ANNOTATIONS_DIR}: {len(jsons)}')
print(f'JPG sin JSON : {len(set(jpgs) - set(jsons))}  {sorted(set(jpgs) - set(jsons))[:10]}')
print(f'JSON sin JPG : {len(set(jsons) - set(jpgs))}  {sorted(set(jsons) - set(jpgs))[:10]}')
print(f'Pares usados : {len(data_pairs)} | Pacientes: '
      f'{len({get_patient_id(p["image_path"]) for p in data_pairs})}')

root = os.path.dirname(os.path.dirname(IMAGES_DIR))
print(f'\nContenido de {root} (busca particiones hermanas como test/):')
for dirpath, _, files in os.walk(root):
    n_img = sum(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files)
    n_js = sum(f.endswith('.json') for f in files)
    if n_img or n_js:
        print(f'  {dirpath}  -> imágenes: {n_img} | json: {n_js}')

lab_shapes, lab_images, shape_types = collections.Counter(), collections.defaultdict(set), collections.Counter()
for p in data_pairs:
    d = load_json_safe(p['annotation_path'])
    for s in d.get('shapes', []):
        lab = s.get('label', '')
        lab_shapes[lab] += 1
        lab_images[lab].add(p['name'])
        shape_types[s.get('shape_type', 'polygon')] += 1
print('\nEtiquetas en las anotaciones (formas | imágenes que la contienen):')
for lab, n in lab_shapes.most_common():
    print(f'  {lab:12s} {n:6d} formas | {len(lab_images[lab]):4d} imágenes')
print('shape_type:', dict(shape_types))

# ------------------------------------------------------------
# PARTE 1 — Predicciones por imagen y clase
# ------------------------------------------------------------
def _rows_from_logits(logits, masks, pairs, kind, mode, fold_tag):
    pred = (torch.sigmoid(logits) > EVAL_THRESHOLD)
    gt = (masks > 0.5)
    tp = (pred & gt).sum((2, 3)).numpy()
    fp = (pred & ~gt).sum((2, 3)).numpy()
    fn = (~pred & gt).sum((2, 3)).numpy()
    pred_np, gt_np = pred.numpy(), gt.numpy()
    recs = []
    for b, p in enumerate(pairs):
        pid = get_patient_id(p['image_path'])
        for c, cname in enumerate(CLASSES):
            ga, pa = int(gt_np[b, c].sum()), int(pred_np[b, c].sum())
            if ga == 0 and pa == 0:
                status, hd = 'both_empty', np.nan
            elif ga > 0 and pa == 0:
                status, hd = 'miss', np.nan
            elif ga == 0 and pa > 0:
                status, hd = 'false_alarm', np.nan
            else:
                status = 'ok'
                hd = _hd95_single(pred_np[b, c].astype(np.float32), gt_np[b, c].astype(np.float32))
            recs.append(dict(kind=kind, mode=mode, fold=fold_tag, image=p['name'], patient=pid,
                             cls=cname, gt_area=ga, pred_area=pa, tp=int(tp[b, c]),
                             fp=int(fp[b, c]), fn=int(fn[b, c]), status=status, hd95=hd))
    return recs


def collect(kind='best_ema'):
    """kind: 'best_ema' (checkpoint reportado en el paper) o 'swa' (no seleccionado por Dice de validación)."""
    recs = []
    for fold_idx, (_, va_idx) in enumerate(FOLD_SPLITS):
        h = all_fold_histories[f'fold_{fold_idx}']
        pairs = [data_pairs[i] for i in va_idx]
        path = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME}_fold{fold_idx}_{kind}.pth')
        if not os.path.exists(path):
            print(f'  SKIP (no existe): {path}')
            continue
        model = build_model().to(DEVICE)
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        model.eval()
        loader = DataLoader(UBPDDataset(pairs, get_val_transform(IMAGE_SIZE, h['fold_mean'], h['fold_std'])),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        for mode in ['base', 'tta']:
            if mode == 'base':
                logits, masks = _predict_loader(model, loader, DEVICE)
            else:
                logits, masks = _predict_loader_tta(model, loader, DEVICE)
            recs += _rows_from_logits(logits, masks, pairs, kind, mode, fold_idx + 1)
        del model
        torch.cuda.empty_cache()
        print(f'  [{kind}] fold {fold_idx + 1} listo')
    return pd.DataFrame(recs)


print('\n' + '=' * 70)
print('PARTE 1 — INFERENCIA POR IMAGEN')
print('=' * 70)
frames = [collect('best_ema'), collect('swa')]
df = pd.concat([f for f in frames if len(f)], ignore_index=True)
csv_path = os.path.join(RESULTS_DIR, 'image_level_v34.csv')
df.to_csv(csv_path, index=False)
print('Guardado:', csv_path, '| filas:', len(df))

# ------------------------------------------------------------
# PARTE 2 — Análisis
# ------------------------------------------------------------
def _dice(tp, fp, fn):
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), np.nan)


def analyze(df, kind, mode, verbose=True):
    d = df[(df.kind == kind) & (df.mode == mode)]
    out = {'kind': kind, 'mode': mode}

    # 2.1 Presencia de estructuras (imágenes y pacientes con GT presente)
    pres = d[d.gt_area > 0].groupby('cls').agg(n_imagenes=('image', 'nunique'), n_pacientes=('patient', 'nunique'))
    pres['n_imagenes_totales'] = d.image.nunique()
    pres['n_pacientes_totales'] = d.patient.nunique()
    out['presencia'] = pres.reindex(CLASSES).to_dict('index')

    # 2.2 Dice agregado por fold (reproduce la Tabla del paper) y macro
    s = d.groupby(['fold', 'cls'])[['tp', 'fp', 'fn']].sum()
    s['dice'] = _dice(s.tp.values, s.fp.values, s.fn.values)
    fold_macro = s['dice'].unstack()[CLASSES].mean(axis=1)
    out['dice_macro_por_fold'] = fold_macro.round(4).to_dict()
    out['dice_macro_media_folds'] = float(fold_macro.mean())
    out['dice_macro_sd_ddof0'] = float(fold_macro.std(ddof=0))
    out['dice_macro_sd_ddof1'] = float(fold_macro.std(ddof=1))

    # 2.3 Agregado fuera de muestra sobre los 88 pacientes + IC 95 % bootstrap por paciente
    patients = sorted(d.patient.unique())
    agg = d.groupby(['patient', 'cls'])[['tp', 'fp', 'fn']].sum()
    T = np.zeros((len(patients), len(CLASSES), 3))
    for i, pt in enumerate(patients):
        for j, c in enumerate(CLASSES):
            T[i, j] = agg.loc[(pt, c)].values

    def metrics(S):
        tp, fp, fn = S[:, 0], S[:, 1], S[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            return _dice(tp, fp, fn), tp / (tp + fp), tp / (tp + fn)

    dice, prec, rec = metrics(T.sum(0))
    obs = np.concatenate([dice, [np.nanmean(dice)], prec, [np.nanmean(prec)], rec, [np.nanmean(rec)]])
    boots = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, len(patients), len(patients))
        dd, pp, rr = metrics(T[idx].sum(0))
        boots.append(np.concatenate([dd, [np.nanmean(dd)], pp, [np.nanmean(pp)], rr, [np.nanmean(rr)]]))
    lo, hi = np.nanpercentile(np.array(boots), [2.5, 97.5], axis=0)
    names = [f'{m}_{c}' for m in ('dice', 'precision', 'recall') for c in CLASSES + ['macro']]
    out['agregado_88_pacientes'] = {n: {'valor': round(float(o), 4), 'ic95': [round(float(l), 4), round(float(h), 4)]}
                                    for n, o, l, h in zip(names, obs, lo, hi)}

    # 2.4 Dice por paciente (pooled dentro del paciente; solo pacientes con la estructura presente)
    pa = d.groupby(['patient', 'cls'])[['tp', 'fp', 'fn', 'gt_area']].sum()
    pa['dice'] = _dice(pa.tp.values, pa.fp.values, pa.fn.values)
    pp_ = pa[pa.gt_area > 0]
    by_cls = pp_.groupby('cls').dice.agg(n='count', media='mean', sd='std', mediana='median',
                                         q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75))
    out['dice_por_paciente_por_clase'] = by_cls.reindex(CLASSES).round(4).to_dict('index')
    pm = pp_.groupby('patient').dice.mean()
    out['dice_macro_por_paciente'] = {'n': int(pm.count()), 'media': round(float(pm.mean()), 4),
                                      'sd': round(float(pm.std()), 4), 'mediana': round(float(pm.median()), 4),
                                      'q25': round(float(pm.quantile(.25)), 4), 'q75': round(float(pm.quantile(.75)), 4),
                                      'min': round(float(pm.min()), 4), 'max': round(float(pm.max()), 4)}

    # 2.5 Estados por imagen y HD95 convencional vs con penalización (como en el paper)
    st = d.groupby(['cls', 'status']).size().unstack(fill_value=0).reindex(CLASSES)
    for col in ['ok', 'miss', 'false_alarm', 'both_empty']:
        if col not in st:
            st[col] = 0
    st['n_imagenes'] = st[['ok', 'miss', 'false_alarm', 'both_empty']].sum(axis=1)
    st['tasa_omision_sobre_GT_presente_%'] = (100 * st['miss'] / (st['miss'] + st['ok'])).round(1)
    st['tasa_falsa_alarma_sobre_GT_ausente_%'] = (100 * st['false_alarm'] / (st['false_alarm'] + st['both_empty'])).round(1)
    out['estados_por_clase'] = st.to_dict('index')
    hd_pen = d.assign(hd=np.where(d.status == 'ok', d.hd95, np.where(d.status == 'both_empty', 0.0, 300.0)))
    hd_conv = d[d.status == 'ok'].groupby('cls').hd95.agg(n='count', media='mean', mediana='median',
                                                           p95=lambda x: x.quantile(.95))
    out['hd95_convencional_px(ambas_no_vacias)'] = hd_conv.reindex(CLASSES).round(2).to_dict('index')
    out['hd95_con_penalizacion_px(todas_las_imagenes)'] = hd_pen.groupby('cls').hd.mean().reindex(CLASSES).round(2).to_dict()

    # 2.6 Dice por imagen solo con GT presente
    di = d[d.gt_area > 0].copy()
    di['dice_img'] = _dice(di.tp.values, di.fp.values, di.fn.values)
    out['dice_por_imagen_GT_presente'] = di.groupby('cls').dice_img.agg(
        n='count', media='mean', mediana='median', q25=lambda x: x.quantile(.25),
        q75=lambda x: x.quantile(.75)).reindex(CLASSES).round(4).to_dict('index')

    if verbose:
        print('\n' + '#' * 70)
        print(f'# kind={kind} | mode={mode}')
        print('#' * 70)
        for k, v in out.items():
            if k in ('kind', 'mode'):
                continue
            print(f'\n--- {k} ---')
            print(json.dumps(v, indent=1, ensure_ascii=False, default=float))
    return out


print('\n' + '=' * 70)
print('PARTE 2 — ANÁLISIS')
print('=' * 70)
results = {}
for kind in df.kind.unique():
    for mode in ['base', 'tta']:
        results[f'{kind}|{mode}'] = analyze(df, kind, mode, verbose=(mode == 'tta'))

if 'swa' in df.kind.unique():
    a = results['best_ema|tta']['dice_macro_media_folds']
    b = results['swa|tta']['dice_macro_media_folds']
    print(f'\nSENSIBILIDAD AL CHECKPOINT (TTA): best_ema={a:.4f} | swa={b:.4f} | Δ(best_ema − swa)={a - b:+.4f}')
    print('Por fold:', {k: (results["best_ema|tta"]["dice_macro_por_fold"][k],
                            results["swa|tta"]["dice_macro_por_fold"][k])
                       for k in results['best_ema|tta']['dice_macro_por_fold']})

out_json = os.path.join(RESULTS_DIR, 'revision_analysis_v34.json')
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=float)
print('Guardado:', out_json)

# ------------------------------------------------------------
# PARTE 3 (opcional) — Partición independiente (evaluación UNA sola vez, sin ajustar nada)
# ------------------------------------------------------------
if TEST_IMAGES_DIR and TEST_ANNOTATIONS_DIR:
    print('\n' + '=' * 70)
    print('PARTE 3 — PARTICIÓN INDEPENDIENTE')
    print('=' * 70)
    t_imgs = sorted(f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith('.jpg'))
    test_pairs = []
    for f in t_imgs:
        j = os.path.join(TEST_ANNOTATIONS_DIR, f.replace('.jpg', '.json'))
        if os.path.exists(j):
            test_pairs.append({'image_path': os.path.join(TEST_IMAGES_DIR, f), 'annotation_path': j,
                               'name': f.replace('.jpg', '')})
    t_pat = {get_patient_id(p['image_path']) for p in test_pairs}
    tr_pat = {get_patient_id(p['image_path']) for p in data_pairs}
    print(f'Pares: {len(test_pairs)} | pacientes: {len(t_pat)} | '
          f'SOLAPAMIENTO con los 88 de entrenamiento/validación: {len(t_pat & tr_pat)} (debe ser 0)')

    per_model, ens_logits, t_masks = [], [], None
    for fold_idx in range(len(FOLD_SPLITS)):
        h = all_fold_histories[f'fold_{fold_idx}']
        path = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME}_fold{fold_idx}_best_ema.pth')
        model = build_model().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False)['model_state_dict'])
        model.eval()
        loader = DataLoader(UBPDDataset(test_pairs, get_val_transform(IMAGE_SIZE, h['fold_mean'], h['fold_std'])),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        logits, t_masks = _predict_loader_tta(model, loader, DEVICE)
        ens_logits.append(logits)
        rows = pd.DataFrame(_rows_from_logits(logits, t_masks, test_pairs, 'best_ema', 'tta', fold_idx + 1))
        g = rows.groupby('cls')[['tp', 'fp', 'fn']].sum().reindex(CLASSES)
        dd = _dice(g.tp.values, g.fp.values, g.fn.values)
        per_model.append(dd)
        print(f'  modelo fold {fold_idx + 1}: Dice por clase {np.round(dd, 4)} | macro {np.nanmean(dd):.4f}')
        del model
        torch.cuda.empty_cache()
    pm_ = np.array(per_model)
    print(f'\nMedia ± SD entre los 5 modelos (macro): {np.nanmean(pm_, axis=1).mean():.4f} ± {np.nanmean(pm_, axis=1).std(ddof=0):.4f}')
    ens = torch.stack(ens_logits).mean(0)
    rows = pd.DataFrame(_rows_from_logits(ens, t_masks, test_pairs, 'ensemble', 'tta', 0))
    g = rows.groupby('cls')[['tp', 'fp', 'fn']].sum().reindex(CLASSES)
    dd = _dice(g.tp.values, g.fp.values, g.fn.values)
    print(f'Ensamble de 5 modelos: Dice por clase {np.round(dd, 4)} | macro {np.nanmean(dd):.4f}')
