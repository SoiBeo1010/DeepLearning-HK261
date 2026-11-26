import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

os.environ['MPLBACKEND'] = 'Agg'
sys.stdout.reconfigure(encoding='utf-8')

import nbformat
import numpy as np
from PIL import Image
import torch
from torch import nn


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
os.chdir(WORKSPACE_ROOT)
path = Path('Notebook/PairCorrection/fer2013-pair-correction-kaggle.ipynb')
protected = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('Notebook').rglob('*.ipynb') if p != path}
nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
ns = {'__name__': '__pair_notebook_smoke__'}
for index, cell in enumerate(nb.cells):
    if cell.cell_type != 'code':
        continue
    tree = ast.parse(cell.source)
    compile(tree, f'cell_{index}', 'exec')
    assert not cell.outputs and cell.execution_count is None
    if set(cell.metadata.get('tags', [])) & {'imports','configuration','data_functions','model_definitions','optimization','training_functions','report_functions'}:
        exec(compile(tree, f'cell_{index}', 'exec'), ns)
    if 'pair_selection' in cell.metadata.get('tags', []) or 'final_evaluation' in cell.metadata.get('tags', []):
        defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=defs, type_ignores=[]), f'definitions_{index}', 'exec'), ns)

cfg = copy.deepcopy(ns['CFG'])
assert cfg['experiment'] == 'pair_balanced' and cfg['num_pairs'] == 3
assert cfg['pilot_epochs'] + cfg['main_epochs'] == 50
cfg.update(pretrained=False, device='cpu', batch_size=7, pilot_epochs=2, main_epochs=3,
           pilot_warmup_epochs=0, main_warmup_epochs=1, use_amp=False,
           num_workers=0, clean_train_max_samples=7, clean_train_every=1, validation_fraction=.25)
device = torch.device('cpu')
pairs = [[2,5],[4,5],[0,5]]
counts = [10,2,12,30,15,18,8]
for invalid in [[[1,1]], [[0,7]], [[2,5],[5,2]], [[False,1]]]:
    try: ns['validate_pairs'](invalid)
    except ValueError: pass
    else: raise AssertionError(f'Bad pairs accepted: {invalid}')
weights = ns['effective_class_weights'](counts, cfg, True)
assert torch.isfinite(weights).all() and (weights > 0).all()
assert math.isclose(weights.mean().item(), 1.0, abs_tol=1e-6)
assert weights[1] > weights[3]
torch.testing.assert_close(ns['effective_class_weights'](counts,{**cfg,'balance_beta':0},True),torch.ones(7))

model = ns['build_model'](cfg, pairs).eval()
baseline = ns['build_model'](cfg, []).eval()
assert sum(p.numel() for p in model.parameters()) == 933805
assert sum(p.numel() for p in baseline.parameters()) == 931047
with torch.inference_mode():
    output = model(torch.randn(3,3,112,112))
assert set(output) == {'logits','global_logits','pair_logits','gates'}
assert output['pair_logits'].shape == (3,3) and output['logits'].shape == (3,7)
assert (output['gates'] >= 0).all() and (output['gates'] <= 1).all()
torch.testing.assert_close(output['logits'],output['global_logits'],atol=0,rtol=0)
# A controlled positive delta shifts only the relevant classes, with opposite signs.
with torch.no_grad(): model.pair_bias.copy_(torch.tensor([2.,0.,0.]))
with torch.inference_mode(): output = model(torch.randn(3,3,112,112))
delta = output['logits']-output['global_logits']
torch.testing.assert_close(delta[:,2],2*output['gates'][:,0])
torch.testing.assert_close(delta[:,5],-2*output['gates'][:,0])
torch.testing.assert_close(delta[:,[0,1,3,4,6]],torch.zeros(3,5),atol=1e-7,rtol=0)
torch.testing.assert_close(delta.sum(1),torch.zeros(3),atol=1e-6,rtol=0)

criterion = ns['PairTrainingLoss'](counts,pairs,cfg,balanced=True)
opt,_ = ns['build_optimizer'](model,cfg,'main')
head_ids = {id(p) for p in opt.param_groups[1]['params']}
backbone_ids = {id(p) for p in opt.param_groups[0]['params']}
assert all(id(p) in head_ids for p in model.local_projection.parameters())
assert not head_ids & backbone_ids
model.train()
for _ in range(2):
    opt.zero_grad(set_to_none=True)
    outputs = model(torch.randn(7,3,112,112))
    loss,classification,auxiliary = criterion(outputs,torch.arange(7))
    assert torch.isfinite(loss) and auxiliary > 0
    loss.backward()
    assert model.pair_weights.grad is not None and model.pair_weights.grad.abs().sum() > 0
    assert model.local_projection[0].weight.grad is not None
    assert torch.isfinite(model.local_projection[0].weight.grad).all()
    opt.step()
assert model.local_projection[0].weight.grad.abs().sum() > 0
assert model.attention_maps.weight.grad.abs().sum() > 0
print('PASS: parameter counts, zero initial correction, gated opposite-sign pair correction and full gradients.')

# No examples in either side of a pair: the auxiliary is a differentiable zero.
empty_logits = torch.randn(4,1,requires_grad=True)
pair_loss = ns['PairTrainingLoss'](counts,[[2,5]],cfg,True)
missing = pair_loss.pair_auxiliary(empty_logits,torch.zeros(4,dtype=torch.long),torch.ones(4,dtype=torch.long),.4)
assert missing.item() == 0
missing.backward()
torch.testing.assert_close(empty_logits.grad,torch.zeros_like(empty_logits))
# Numeric check for mixed labels including an out-of-pair label.
logits = torch.tensor([[1.2],[-.3]],requires_grad=True)
ya = torch.tensor([2,0]); yb = torch.tensor([5,2]); lam = .25
actual = pair_loss.pair_auxiliary(logits,ya,yb,lam)
wi,wj = counts[5]/(counts[2]+counts[5]),counts[2]/(counts[2]+counts[5])
expected = (lam*wi*nn.functional.softplus(-logits[0,0]) + (1-lam)*wj*nn.functional.softplus(logits[0,0])
            + (1-lam)*wi*nn.functional.softplus(-logits[1,0])) / (lam*wi+(1-lam)*wj+(1-lam)*wi)
torch.testing.assert_close(actual,expected)
actual.backward(); assert torch.isfinite(logits.grad).all()
plain_criterion = ns['PairTrainingLoss'](counts,[],cfg,False)
fake = {'logits':torch.randn(4,7,requires_grad=True),'pair_logits':torch.zeros(4,0)}
targets = torch.tensor([0,1,2,3])
ce,_,_ = plain_criterion(fake,targets)
torch.testing.assert_close(ce,nn.functional.cross_entropy(fake['logits'],targets,label_smoothing=.05))
print('PASS: effective-number weights from train, weighted pair BCE/MixUp numeric identity, empty-pair batches and global CE equivalence.')

with tempfile.TemporaryDirectory(prefix='fer-pair-notebook-smoke-') as directory:
    folder = Path(directory)
    model.eval()
    sample = torch.randn(2,3,112,112)
    with torch.inference_mode(): expected = model(sample)['logits']
    checkpoint = folder/'checkpoint.pth'
    wrapped = nn.DataParallel(model)
    ns['save_checkpoint'](wrapped,cfg,2,{'loss':1.0,'accuracy':.7,'macro_f1':.69,'balanced_accuracy':.68},checkpoint)
    payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert not any(key.startswith('module.') for key in payload['state_dict'])
    restored = ns['load_trained_model'](checkpoint,device)
    with torch.inference_mode(): torch.testing.assert_close(restored(sample)['logits'],expected)
    damaged = copy.deepcopy(payload)
    damaged['state_dict']['pair_indices'][0,0]=1
    torch.save(damaged,folder/'damaged.pth')
    try: ns['load_trained_model'](folder/'damaged.pth',device)
    except ValueError: pass
    else: raise AssertionError('Mismatched pair metadata/buffers accepted.')
    print('PASS: DataParallel-safe checkpoint export, exact model reconstruction and corrupted-pair rejection.')

    true = list(range(7))*2
    pred = true.copy(); pred[2]=5; pred[9]=5; pred[5]=2
    ranking,matrix=ns['rank_confusion_pairs'](true,pred)
    assert (ranking[0]['i'],ranking[0]['j']) == (2,5)
    val={'targets':true,'predictions':pred}
    selected,_=ns['choose_pairs'](cfg,val,'abc')
    assert selected['pairs'][0]==[2,5] and len(selected['pairs'])==3
    pairfile=folder/'pairs.json';pairfile.write_text(json.dumps(selected),encoding='utf-8')
    reused,_=ns['choose_pairs']({**cfg,'pairs_file':str(pairfile)},val,'abc')
    assert reused['pairs']==selected['pairs']
    try: ns['choose_pairs']({**cfg,'pairs_file':str(pairfile)},val,'wrong')
    except ValueError: pass
    else: raise AssertionError('Wrong split pair artifact accepted.')
    random1,_=ns['choose_pairs']({**cfg,'pair_selection':'random'},val,'abc')
    random2,_=ns['choose_pairs']({**cfg,'pair_selection':'random'},val,'abc')
    assert random1['pairs']==random2['pairs']
    assert ns['choose_pairs']({**cfg,'use_pair_correction':False},val,'abc')[0]['pairs']==[]

    root=folder/'data'
    for split,count in [('train',4),('test',1)]:
        for label_id,label in enumerate(ns['CLASS_NAMES']):
            target=root/split/label; target.mkdir(parents=True)
            for i in range(count):
                image=np.random.default_rng(label_id*10+i).integers(0,256,(48,48),dtype=np.uint8)
                Image.fromarray(image).save(target/f'{i}.png')
    (root/'test').rename(root/'held_out')
    loaders,manifest,summary=ns['build_data'](cfg,root,device)
    other,other_manifest,_=ns['build_data']({**cfg,'seed':43},root,device)
    assert manifest.equals(other_manifest)
    assert len(loaders['train'].dataset)==21 and len(loaders['val'].dataset)==7
    assert set(manifest[manifest.split=='train'].path).isdisjoint(manifest[manifest.split=='val'].path)
    fp=ns['manifest_fingerprint'](manifest)
    assert fp==ns['manifest_fingerprint'](other_manifest)
    for loader in loaders.values():
        data,labels=next(iter(loader));assert data.shape[1:]==(3,112,112)
    print('PASS: normalized confusion ranking, reproducible random/reused pairs, fingerprint guard, fixed split and no test reads.')

    # Execute every cell in order, including pair mining and full pilot/main train.
    # Test stays physically absent until the final evaluation cell.
    run={'__name__':'__pair_run_all__'}
    test_reached=False
    with patch('matplotlib.pyplot.show'):
        for i,cell in enumerate(nb.cells):
            if cell.cell_type!='code':continue
            if 'final_evaluation' in cell.metadata.get('tags',[]):
                (root/'held_out').rename(root/'test')
                test_reached=True
            exec(compile(cell.source,f'run_all_cell_{i}','exec'),run)
            if 'configuration' in cell.metadata.get('tags',[]):
                run['CFG'].update(cfg)
                run['CFG']['data_dir']=str(root)
                run['CFG']['output_root']=str(folder/'outputs')
                run['DEVICE']=device
    assert test_reached
    assert len(run['pilot_history'])==2 and len(run['history'])==3
    assert all(np.isfinite(run['history'][column]).all() for column in ['train_loss','pair_loss','val_accuracy','val_macro_f1'])
    assert run['model'].pair_indices.cpu().tolist()==run['PAIRS']
    assert len(run['PAIRS'])==3
    run_dir=run['RUN_DIR']
    metadata=json.loads((run_dir/'training.json').read_text())
    assert metadata['parameter_count']==933805
    assert metadata['class_counts']==[3]*7
    assert metadata['pair_info']['source']=='pilot_validation_confusion'
    best=json.loads((run_dir/'main_best_validation.json').read_text())
    assert best['accuracy']==run['history'].val_accuracy.max()
    saved=torch.load(run_dir/'pair_correction_best_checkpoint.pth',map_location='cpu',weights_only=True)
    assert saved['epoch']==best['epoch']
    assert all(torch.equal(value,run['model'].state_dict()[key]) for key,value in saved['state_dict'].items())
    assert run['test_metrics']['sample_count']==7
    assert math.isclose(run['validation_metrics']['accuracy'],best['accuracy'],abs_tol=1e-10)
    for artifact in ['pairs.json','pilot_pair_ranking.csv','pilot_validation_confusion.csv','class_weights.csv',
                     'main_history.csv','pilot_history.csv','training.json','model_config.json','split_manifest.csv',
                     'validation_predictions.csv','validation_pair_diagnostics.csv','test_report.txt','test_confusion.png',
                     'test_metrics.json','test_predictions.csv','training_history.png']:
        assert (run_dir/artifact).is_file(),artifact
    print('PASS: entire notebook executed in order, 2 pilot + 3 main epochs, frozen validation-mined pairs, strict pilot transfer, reports and test.')

    # A global-only run must use no auxiliary branch and no pair-loss terms.
    global_cfg={**cfg,**ns['EXPERIMENT_PRESETS']['global_balanced'],'main_epochs':2,'main_warmup_epochs':0}
    pilot=ns['load_trained_model'](run_dir/'pilot_global_best_checkpoint.pth',device)
    global_model=ns['build_model'](global_cfg,[])
    global_model.features.load_state_dict(pilot.features.state_dict(),strict=True)
    global_model.global_classifier.load_state_dict(pilot.global_classifier.state_dict(),strict=True)
    global_run=folder/'global_run';global_run.mkdir()
    global_model,global_history=ns['run_phase'](global_model,global_cfg,loaders,[3]*7,global_run,device,'main')
    assert (global_history.pair_loss==0).all()
    assert sum(p.numel() for p in global_model.parameters())==931047
    assert global_model.pair_indices.shape==(0,2)
    try:
        ns['run_phase'](global_model,global_cfg,loaders,[3]*7,global_run,device,'main')
    except FileExistsError:
        pass
    else:
        raise AssertionError('An existing run must not be overwritten by rerunning a phase.')
    print('PASS: global-only balanced preset train/export path has 931,047 parameters and zero pair auxiliary.')

assert all(hashlib.sha256(p.read_bytes()).hexdigest()==value for p,value in protected.items())
print('ALL AVAILABLE PAIR-CORRECTION CHECKS PASSED. Prior notebooks unchanged; FER2013/GPU accuracy not measured locally.')
