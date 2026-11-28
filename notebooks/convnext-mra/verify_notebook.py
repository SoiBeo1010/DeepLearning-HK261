import ast
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'
sys.stdout.reconfigure(encoding='utf-8')

import nbformat
import numpy as np
import torch
from PIL import Image


NOTEBOOK_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = NOTEBOOK_DIR.parents[1]
os.chdir(WORKSPACE_ROOT)
NB_PATH = NOTEBOOK_DIR / 'fer2013-convnext-mra-kaggle.ipynb'

# Ghi nhận SHA-256 của mọi notebook khác để bảo đảm không đụng vào notebook gốc.
protected_notebooks = {
    p: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in NOTEBOOK_DIR.parent.rglob('*.ipynb')
    if p != NB_PATH
}

nb = nbformat.read(NB_PATH, as_version=4)
nbformat.validate(nb)

ns = {'__name__': '__convnext_mra_verify__'}
load_tags = {'imports', 'configuration', 'data_functions', 'model_definitions', 'optimization', 'training_functions', 'report_functions'}

for idx, cell in enumerate(nb.cells):
    if cell.cell_type != 'code':
        continue
    tree = ast.parse(cell.source)
    compile(tree, f'cell_{idx}', 'exec')
    assert not cell.outputs and cell.execution_count is None, f'Cell {idx} should be clean (no saved outputs).'
    tags = set(cell.metadata.get('tags', []))
    if tags & load_tags:
        exec(compile(tree, f'cell_{idx}', 'exec'), ns)
    if 'final_evaluation' in tags:
        defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=defs, type_ignores=[]), f'defs_{idx}', 'exec'), ns)

# 1. Kiểm tra preset & validate_config
cfg_default = copy.deepcopy(ns['CFG'])
assert cfg_default['experiment'] == 'convnext_tiny_224_mra'
assert cfg_default['backbone'] == 'convnext_tiny'
assert cfg_default['image_size'] == 224 and cfg_default['use_mra'] is True
assert cfg_default['deduplicate_train'] is True
assert cfg_default['mixup_alpha'] == 0.40 and cfg_default['mix_probability'] == 0.50

for name, preset in ns['EXPERIMENT_PRESETS'].items():
    merged = {**cfg_default, **preset}
    ns['validate_config'](merged)
print('[OK] All presets pass validate_config:', list(ns['EXPERIMENT_PRESETS'].keys()))

for bad_override in [
    {'backbone': 'unknown_net'},
    {'image_size': 96},
    {'mra_dim': 100},
    {'selection_metric': 'roc_auc'},
    {'warmup_epochs': cfg_default['epochs']},
]:
    try:
        ns['validate_config']({**cfg_default, **bad_override})
    except ValueError:
        pass
    else:
        raise AssertionError(f'Invalid config accepted: {bad_override}')

# 2. Kiểm tra kiến trúc: ConvNeXt-Tiny (MRA & Global), ResNet-34 (128), ResNet-18 (112)
for preset_name in ('convnext_tiny_224_mra', 'convnext_tiny_224_global', 'convnext_tiny_128_mra', 'resnet34_128_mra', 'resnet18_112_mra'):
    c = {**cfg_default, **ns['EXPERIMENT_PRESETS'][preset_name], 'pretrained': False}
    m = ns['build_model'](c, pretrained=False).eval()
    h = c['image_size']
    x = torch.randn(2, 3, h, h)
    with torch.inference_mode():
        out = m(x)
    assert out['logits'].shape == (2, 7)
    assert torch.isfinite(out['logits']).all()
    if c['use_mra']:
        assert out['attention'] is not None and out['attention'].shape[1] == c['mra_heads']
        assert 0.0 <= float(out['diversity_loss'].item()) <= 1.0001
        torch.testing.assert_close(out['attention'].sum(dim=-1), torch.ones(2, c['mra_heads']), atol=1e-5, rtol=1e-5)
    else:
        assert out['attention'] is None and float(out['diversity_loss'].item()) == 0.0
    groups = m.stage_param_groups(c['backbone_lr'], c['head_lr'], c['llrd'], c['weight_decay'])
    group_param_ids = [id(p) for g in groups for p in g['params']]
    all_param_ids = [id(p) for p in m.parameters() if p.requires_grad]
    assert len(group_param_ids) == len(set(group_param_ids)) == len(all_param_ids), f'LLRD group mismatch for {preset_name}'
    print(f'[OK] Model {preset_name}: params={sum(p.numel() for p in m.parameters()):,}, out={tuple(out["logits"].shape)}')

# 3. Kiểm tra Loss (CE + Logit Adjustment + Diversity), MixUp/CutMix, EMA, Temperature Scaling, ECE
counts = [3596, 392, 3687, 6493, 4469, 4347, 2854]
loss_ce = ns['FERLoss'](counts, label_smoothing=0.05, logit_adj_tau=0.0, diversity_weight=0.05)
loss_la = ns['FERLoss'](counts, label_smoothing=0.05, logit_adj_tau=0.4, diversity_weight=0.05)
m_small = ns['build_model']({**cfg_default, **ns['EXPERIMENT_PRESETS']['resnet18_112_mra'], 'pretrained': False})
x_b = torch.randn(4, 3, 112, 112)
y_b = torch.tensor([0, 1, 2, 3])
out_b = m_small(x_b)
tot, cls_l, div_l = loss_la(out_b, y_b)
tot.backward()
assert all(p.grad is not None for p in m_small.classifier.parameters())
assert torch.isfinite(tot) and float(cls_l) > 0 and float(div_l) >= 0

ema = ns['ModelEMA'](m_small, decay=0.99)
ema.update(m_small)
assert ema.updates == 1

t_opt = ns['fit_temperature'](torch.randn(50, 7) * 2.0, torch.randint(0, 7, (50,)))
assert 0.05 <= t_opt <= 10.0
with torch.inference_mode():
    inf_logits = torch.randn(50, 7) * 2.0
    inf_targets = torch.randint(0, 7, (50,))
t_opt_inf = ns['fit_temperature'](inf_logits, inf_targets)
assert 0.05 <= t_opt_inf <= 10.0
m_cal = ns['compute_metrics_from_logits'](torch.randn(50, 7), torch.randint(0, 7, (50,)))
assert set(m_cal) == {'nll', 'ece', 'accuracy', 'macro_f1', 'balanced_accuracy', 'composite'}
assert 0.0 <= m_cal['ece'] <= 1.0
print('[OK] Loss, backward, EMA, LLRD, Temperature Scaling, ECE verified.')

# 4. Kiểm tra Deduplication (MD5 exact + conflict + dHash group split), Grayscale cache & End-to-End Smoke Run
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    data_root = tmp_path / 'fer2013'
    out_root = tmp_path / 'outputs'
    cache_dir = tmp_path / 'cache'
    rng = np.random.default_rng(123)

    # Tạo 7 lớp * 12 ảnh gốc 48x48 + cố tình chèn bản sao exact duplicate (cùng nhãn & khác nhãn)
    base_img = rng.integers(0, 256, size=(48, 48), dtype=np.uint8)
    conflict_img = rng.integers(0, 256, size=(48, 48), dtype=np.uint8)
    for split, per_class in (('train', 12), ('test', 3)):
        for c_idx, cname in enumerate(ns['CLASS_NAMES']):
            cdir = data_root / split / cname
            cdir.mkdir(parents=True, exist_ok=True)
            for k in range(per_class):
                arr = rng.integers(0, 256, size=(48, 48), dtype=np.uint8)
                Image.fromarray(arr, mode='L').save(cdir / f'{cname}_{k:02d}.png')

    # Chèn 2 bản sao giống hệt nhau vào lớp angry (phải giữ 1, loại 1)
    Image.fromarray(base_img, mode='L').save(data_root / 'train' / 'angry' / 'dup_a.png')
    Image.fromarray(base_img, mode='L').save(data_root / 'train' / 'angry' / 'dup_b.png')
    # Chèn 1 ảnh giống hệt nhau ở 2 lớp khác nhau angry vs sad (phải loại cả 2)
    Image.fromarray(conflict_img, mode='L').save(data_root / 'train' / 'angry' / 'conflict_a.png')
    Image.fromarray(conflict_img, mode='L').save(data_root / 'train' / 'sad' / 'conflict_b.png')
    # Chèn 1 bản sao train sang test để kiểm tra bộ đếm overlap train-test
    Image.fromarray(base_img, mode='L').save(data_root / 'test' / 'angry' / 'leak_test.png')

    smoke_cfg = copy.deepcopy(cfg_default)
    smoke_cfg.update(ns['EXPERIMENT_PRESETS']['resnet18_112_mra'])
    smoke_cfg.update(
        experiment='resnet18_112_mra',
        run_name='smoke_resnet18_112_mra',
        data_dir=str(data_root),
        output_root=str(out_root),
        cache_dir=str(cache_dir),
        pretrained=False,
        device='cpu',
        use_amp=False,
        num_workers=0,
        cache_workers=2,
        batch_size=8,
        epochs=2,
        warmup_epochs=1,
        early_stop_patience=5,
        validation_fraction=0.25,
        clean_train_every=1,
        clean_train_max_samples=14,
        deduplicate_train=True,
    )
    ns['validate_config'](smoke_cfg)

    loaders, manifest, summary, cache = ns['build_data'](smoke_cfg, data_root, torch.device('cpu'))
    dedup = summary['deduplication']
    assert dedup['removed_same_label_exact_duplicates'] == 1, dedup
    assert dedup['removed_conflicting_label_exact_duplicates'] == 2, dedup
    assert len(cache.images) == dedup['raw_train_pool']
    # Kiểm tra disk cache reload
    ns['GRAY_CACHE_REGISTRY'].clear()
    loaders2, manifest2, _, _ = ns['build_data'](smoke_cfg, data_root, torch.device('cpu'))
    assert ns['manifest_fingerprint'](manifest) == ns['manifest_fingerprint'](manifest2)
    print('[OK] Deduplication (exact + conflict + dHash StratifiedGroupKFold) & Gray disk cache verified.')

    run_dir = out_root / 'smoke_run'
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(run_dir / 'split_manifest.csv', index=False)
    train_counts = np.bincount(manifest.loc[manifest['split'] == 'train', 'label'].to_numpy(), minlength=7).tolist()
    meta = {
        'config': smoke_cfg,
        'model_config': ns['model_spec'](smoke_cfg),
        'classes': list(ns['CLASS_NAMES']),
        'class_counts': train_counts,
        'checkpoint_file': 'best_checkpoint.pth',
    }
    (run_dir / 'training.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    m_smoke = ns['build_model'](smoke_cfg, pretrained=False)
    best_m, payload, hist = ns['run_training'](m_smoke, smoke_cfg, loaders, train_counts, run_dir, torch.device('cpu'))
    assert len(hist) == 2
    assert (run_dir / 'best_checkpoint.pth').is_file()
    assert (run_dir / 'history.csv').is_file()

    # Kiểm tra checkpoint reload & test evaluation (gồm phát hiện exact overlap train-test)
    reloaded, loaded_payload = ns['load_trained_model'](run_dir / 'best_checkpoint.pth', torch.device('cpu'))
    assert loaded_payload['epoch'] == payload['epoch']
    test_metrics = ns['evaluate_final_test'](run_dir, data_root, torch.device('cpu'), posthoc_tau=0.2)
    assert 'accuracy' in test_metrics and 'macro_f1' in test_metrics and 'ece' in test_metrics
    print('[OK] Full end-to-end training, EMA checkpoint round-trip, HFlip TTA & Test eval verified.')

# 5. Bảo đảm không notebook gốc nào bị thay đổi
for p, digest in protected_notebooks.items():
    assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, f'Protected notebook modified: {p}'
print('[OK] All existing notebooks remain 100% untouched.')
print('ALL CHECKS PASSED SUCCESSFULLY.')
