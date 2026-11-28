import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
from unittest.mock import Mock, patch

os.environ['MPLBACKEND'] = 'Agg'
sys.stdout.reconfigure(encoding='utf-8')

import nbformat
import numpy as np
import pandas as pd
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
assert cfg['experiment'] == 'global_ce_224' and cfg['num_pairs'] == 3
assert cfg['image_size'] == 224 and cfg['head_type'] == 'linear' and cfg['mixup_probability'] == .3
assert not cfg['use_pair_correction'] and not cfg['balanced_loss']
assert cfg['backbone_lr'] == 1e-4 and cfg['pair_mining_window'] == 3
assert cfg['pilot_epochs'] == 15 and cfg['main_epochs'] == 35
assert ns['EXPERIMENT_PRESETS']['global_ce_lr3e5']['backbone_lr'] == 3e-5
assert cfg['pilot_epochs'] + cfg['main_epochs'] == 50
assert ns['tqdm'].__module__ == 'tqdm.std'
baseline_preset=ns['EXPERIMENT_PRESETS']['global_ce']
assert {k for k in baseline_preset if baseline_preset[k]!=ns['EXPERIMENT_PRESETS']['global_ce_224'][k]}=={'image_size'}
assert {k for k in baseline_preset if baseline_preset[k]!=ns['EXPERIMENT_PRESETS']['global_ce_pretrained_head'][k]}=={'head_type'}
assert {k for k in baseline_preset if baseline_preset[k]!=ns['EXPERIMENT_PRESETS']['global_ce_mixup50'][k]}=={'mixup_probability'}
for preset in ns['EXPERIMENT_PRESETS'].values():
    ns['validate_config']({**cfg,**preset})
try:ns['validate_config']({**cfg,'head_type':'unsupported'})
except ValueError:pass
else:raise AssertionError('Unknown classifier type accepted.')
cfg.update(ns['EXPERIMENT_PRESETS']['pair_balanced'])
cfg.update(experiment='pair_balanced', run_name='smoke_pair_balanced')
cfg.update(pretrained=False, device='cpu', batch_size=7, pilot_epochs=2, main_epochs=3,
           pilot_warmup_epochs=0, main_warmup_epochs=1, use_amp=False,
           num_workers=0, clean_train_max_samples=7, clean_train_every=1, validation_fraction=.25,
           pair_mining_window=2)
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
    legacy = copy.deepcopy(payload)
    legacy['model_config'].pop('image_size')
    legacy['model_config'].pop('head_type')
    torch.save(legacy,folder/'legacy112.pth')
    legacy_restored = ns['load_trained_model'](folder/'legacy112.pth',device)
    with torch.inference_mode(): torch.testing.assert_close(legacy_restored(sample)['logits'],expected)
    large_cfg={**cfg,'image_size':224}
    ns['validate_config'](large_cfg)
    large_model=ns['build_model'](large_cfg,pairs).eval()
    large_input=torch.randn(2,3,224,224)
    with torch.inference_mode(): large_expected=large_model(large_input)['logits']
    ns['save_checkpoint'](large_model,large_cfg,1,{'loss':1.,'accuracy':.6,'macro_f1':.5,'balanced_accuracy':.5},folder/'224.pth')
    large_restored=ns['load_trained_model'](folder/'224.pth',device)
    assert large_restored.image_size==224
    with torch.inference_mode(): torch.testing.assert_close(large_restored(large_input)['logits'],large_expected)
    try: large_restored(sample)
    except ValueError: pass
    else: raise AssertionError('Checkpoint resolution must be enforced.')
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
    selected,_,_=ns['choose_pairs']({**cfg,'pair_mining_window':1},val,'abc')
    assert selected['pairs'][0]==[2,5] and len(selected['pairs'])==3
    pairfile=folder/'pairs.json';pairfile.write_text(json.dumps(selected),encoding='utf-8')
    reused,_,_=ns['choose_pairs']({**cfg,'pairs_file':str(pairfile)},val,'abc')
    assert reused['pairs']==selected['pairs']
    try: ns['choose_pairs']({**cfg,'pairs_file':str(pairfile)},val,'wrong')
    except ValueError: pass
    else: raise AssertionError('Wrong split pair artifact accepted.')
    random1,_,_=ns['choose_pairs']({**cfg,'pair_selection':'random'},val,'abc')
    random2,_,_=ns['choose_pairs']({**cfg,'pair_selection':'random'},val,'abc')
    assert random1['pairs']==random2['pairs']
    assert ns['choose_pairs']({**cfg,'use_pair_correction':False},val,'abc')[0]['pairs']==[]

    # Pretrained hidden layer must be reused exactly, not reinitialized as a new head.
    # Use a deterministic local torchvision state dict as a network-free ImageNet-file fixture.
    fixture_backbone=ns['models'].mobilenet_v3_small(weights=None)
    with torch.no_grad():
        fixture_backbone.classifier[0].weight.fill_(.0123)
        fixture_backbone.classifier[0].bias.fill_(.0045)
    backbone_path=folder/'mobilenet_fixture.pth'
    torch.save(fixture_backbone.state_dict(),backbone_path)
    hidden_cfg={**cfg,'head_type':'pretrained_mlp','pretrained':True,'backbone_weights_path':str(backbone_path)}
    hidden=ns['build_model'](hidden_cfg,[],pretrained=True).eval()
    assert sum(p.numel() for p in hidden.parameters())==1525031
    assert hidden.global_classifier.in_features==1024
    assert sum(p.numel() for p in hidden.global_projection.parameters())==590848
    torch.testing.assert_close(hidden.global_projection[0].weight,fixture_backbone.classifier[0].weight,atol=0,rtol=0)
    torch.testing.assert_close(hidden.global_projection[0].bias,fixture_backbone.classifier[0].bias,atol=0,rtol=0)
    # Also check the online weights argument without accessing the network.
    with patch.object(ns['models'],'mobilenet_v3_small',return_value=fixture_backbone) as constructor:
        online=ns['build_model']({**hidden_cfg,'backbone_weights_path':None},[],pretrained=True)
    constructor.assert_called_once_with(weights=ns['models'].MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    torch.testing.assert_close(online.global_projection[0].weight,hidden.global_projection[0].weight)
    transferred=ns['build_model'](hidden_cfg,pairs,pretrained=False)
    assert sum(p.numel() for p in transferred.parameters())==1527789
    with torch.no_grad():
        hidden.global_projection[0].bias.add_(.03)
        hidden.global_classifier.bias.add_(.04)
    ns['transfer_global_state'](hidden,transferred)
    for part in ('features','global_projection','global_classifier'):
        first=getattr(hidden,part).state_dict();second=getattr(transferred,part).state_dict()
        assert first.keys()==second.keys()
        assert all(torch.equal(value,second[key]) for key,value in first.items())
    hidden_optimizer,_=ns['build_optimizer'](nn.DataParallel(transferred),hidden_cfg,'main')
    low_lr={id(p) for p in hidden_optimizer.param_groups[0]['params']}
    high_lr={id(p) for p in hidden_optimizer.param_groups[1]['params']}
    assert not low_lr & high_lr
    assert low_lr | high_lr == {id(p) for p in transferred.parameters()}
    assert all(id(p) in low_lr for p in transferred.global_projection.parameters())
    assert all(id(p) in high_lr for p in transferred.global_classifier.parameters())
    assert all(id(p) in high_lr for p in transferred.local_projection.parameters())
    for image_size in (112,224):
        candidate_cfg={**hidden_cfg,'image_size':image_size}
        candidate=ns['build_model'](candidate_cfg,pairs,pretrained=False)
        pilot=ns['build_model'](candidate_cfg,[],pretrained=False)
        ns['transfer_global_state'](pilot,candidate)
        candidate.train()
        batch=torch.randn(2,3,image_size,image_size)
        result=candidate(batch)
        assert result['logits'].shape==(2,7)
        gradient_loss=ns['PairTrainingLoss'](counts,pairs,candidate_cfg,balanced=True)(result,torch.tensor([2,5]))[0]
        gradient_loss.backward()
        assert candidate.global_projection[0].weight.grad is not None
        assert candidate.global_projection[0].weight.grad.abs().sum()>0
        assert torch.isfinite(candidate.global_projection[0].weight.grad).all()
        candidate.eval()
        with torch.inference_mode():expected_hidden=candidate(batch)['logits']
        filename=folder/f'hidden_{image_size}.pth'
        ns['save_checkpoint'](candidate,candidate_cfg,1,{'loss':1.,'accuracy':.6,'macro_f1':.5,'balanced_accuracy':.5},filename)
        with patch.object(ns['models'],'mobilenet_v3_small',wraps=ns['models'].mobilenet_v3_small) as constructor:
            from_saved=ns['load_trained_model'](filename,device)
        constructor.assert_called_once_with(weights=None)
        assert from_saved.head_type=='pretrained_mlp' and from_saved.image_size==image_size
        with torch.inference_mode():torch.testing.assert_close(from_saved(batch)['logits'],expected_hidden)
        wrong_head=ns['build_model']({**candidate_cfg,'head_type':'linear'},[],pretrained=False)
        try:ns['transfer_global_state'](wrong_head,candidate)
        except ValueError:pass
        else:raise AssertionError('Mismatched pilot/main head type accepted.')
    print('PASS: pretrained hidden weights preserved, exact pilot transfer, low-LR optimizer group, 112/224 backward and metadata checkpoints.')

    # Old/early confusion must not influence ranking of the last two snapshots.
    early=np.eye(7,dtype=int)*10;early[0,0]=0;early[0,1]=10;early[1,1]=0;early[1,0]=10
    late1=np.eye(7,dtype=int)*10;late1[2,2]=4;late1[2,5]=6;late1[5,5]=5;late1[5,2]=5
    late2=np.eye(7,dtype=int)*10;late2[2,2]=5;late2[2,5]=5;late2[5,5]=4;late2[5,2]=6
    snapshots=[{'epoch':i+1,'accuracy':.58+i*.02,'confusion':matrix.tolist()} for i,matrix in enumerate([early,late1,late2])]
    ranking,average,chosen,stability=ns['aggregate_pilot_rankings'](snapshots,2,3)
    assert (ranking[0]['i'],ranking[0]['j'])==(2,5)
    assert math.isclose(ranking[0]['score'],.55)
    assert ranking[0]['top_k_count']==2 and chosen[0]['epoch']==2
    np.testing.assert_array_equal(average,(late1+late2)/2)
    assert all(row['jaccard_with_aggregate']==1 for row in stability)
    selected_window,_,mining=ns['choose_pairs'](cfg,val,'abc',snapshots)
    assert selected_window['pairs'][0]==[2,5]
    assert selected_window['mining_epochs']==[2,3]
    assert selected_window['source']=='pilot_validation_window'
    try: ns['choose_pairs'](cfg,val,'abc')
    except ValueError: pass
    else: raise AssertionError('Window mining must not silently fall back to a single snapshot.')
    wrong_support=copy.deepcopy(snapshots);wrong_support[-1]['confusion'][0][0]+=1
    try: ns['aggregate_pilot_rankings'](wrong_support,2,3)
    except ValueError: pass
    else: raise AssertionError('Snapshot supports must match.')
    low=copy.deepcopy(snapshots)
    for row in low:row['accuracy']=.2
    assert ns['choose_pairs'](cfg,val,'abc',low)[0]['warnings']
    print('PASS: tail-window mining ignores early confusion, preserves support, exports stability and warns on a weak pilot.')

    # Diagnostics must distinguish a helpful and harmful correction, per class.
    class FixedCorrection(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('pair_indices',torch.tensor([[0,1]]))
            self.register_buffer('pair_incidence',torch.tensor([[1.,-1.,0,0,0,0,0]]))
            self.register_buffer('z',torch.tensor([[0.,.2,-10,-10,-10,-10,-10],
                                                 [.4,0.,-10,-10,-10,-10,-10],
                                                 [.2,0.,-10,-10,-10,-10,-10],
                                                 [0.,.2,-10,-10,-10,-10,-10]]))
            self.register_buffer('g',torch.tensor([[.1],[.5],[.9],[.3]]))
            self.register_buffer('d',torch.tensor([[5.],[-5.],[-3.],[3.]]))
        def forward(self,inputs):
            ix=inputs[:,0].long()
            global_logits=self.z[ix];gates=self.g[ix];delta=self.d[ix]
            return {'global_logits':global_logits,'gates':gates,'pair_logits':delta,
                    'logits':global_logits+(gates*delta)@self.pair_incidence}
    fixture_loader=[(torch.arange(4).float().unsqueeze(1),torch.tensor([0,1,0,1]))]
    diagnostic_result=ns['evaluate'](FixedCorrection(),fixture_loader,device,cfg,collect_pair_details=True)
    diagnostic_pairs,by_class=ns['pair_diagnostics'](diagnostic_result,[[0,1]])
    row=diagnostic_pairs.iloc[0]
    assert row.only_pair_helpful_all==2 and row.only_pair_harmful_all==2
    assert row.only_pair_net_correct_all==0 and row.marginal_net_correct_all==0
    assert math.isclose(row.mean_gate_on_pair,.45,abs_tol=1e-6)
    assert row.remove_pair_accuracy_all==.5
    assert len(by_class)==7 and by_class.samples.tolist()==[2,2,0,0,0,0,0]
    assert math.isclose(by_class.iloc[0].mean_gate,.5,abs_tol=1e-6)
    assert math.isclose(by_class.iloc[1].mean_delta,-1.,abs_tol=1e-6)
    with patch('matplotlib.pyplot.show'):
        ns['export_report'](diagnostic_result,folder,'fixture',[[0,1]])
    predicted=pd.read_csv(folder/'fixture_predictions.csv')
    assert predicted.pair_angry_disgust_gate.notna().all()
    assert (folder/'fixture_pair_by_class.csv').is_file()
    print('PASS: detailed gate/delta, helpful/harmful only-pair vs remove-pair diagnostics and conditional per-class summaries.')

    root=folder/'data'
    for split,count in [('train',4),('test',1)]:
        for label_id,label in enumerate(ns['CLASS_NAMES']):
            target=root/split/label; target.mkdir(parents=True)
            for i in range(count):
                image=np.random.default_rng(label_id*10+i).integers(0,256,(48,48),dtype=np.uint8)
                Image.fromarray(image).save(target/f'{i}.png')
    cfg['cache_dir']=str(folder/'cache_packs')
    # Threaded decode must not block replenishment on a slow first item.
    expected_arrays={str(i):np.full((2,2,3),i,dtype=np.uint8) for i in range(8)}
    reached_later=threading.Event()
    def delayed_decode(path):
        if path=='0':
            assert reached_later.wait(timeout=3),'Slow first item blocked queue replenishment.'
        if path=='4':reached_later.set()
        return expected_arrays[path].copy()
    with patch.dict(ns['iter_decoded_images'].__globals__,decode_rgb_array=delayed_decode):
        decoded=dict(ns['iter_decoded_images'](list(expected_arrays),2))
    assert reached_later.is_set() and set(decoded)==set(expected_arrays)
    for key in decoded:np.testing.assert_array_equal(decoded[key],expected_arrays[key])
    print('PASS: bounded rolling queue keeps reading beyond a slow first file.')

    image_paths=sorted(str(p.resolve()) for p in (root/'train').rglob('*.png'))
    cache_signature=ns['image_cache_signature'](image_paths,cfg)
    pack_path=folder/'standalone_cache.npz'
    cache=ns['CachedImageLoader'](image_paths,256,workers=4,cache_path=pack_path,signature=cache_signature)
    assert len(cache.images)==28 and cache.used_bytes==28*48*48*3 and pack_path.is_file()
    expected=np.asarray(ns['datasets'].folder.default_loader(image_paths[0])).copy()
    first=cache(image_paths[0]);first.putpixel((0,0),(0,0,0))
    np.testing.assert_array_equal(np.asarray(cache(image_paths[0])),expected)
    with patch.object(ns['datasets'].folder,'default_loader',side_effect=AssertionError('Disk pack must avoid source decode')):
        disk_cache=ns['CachedImageLoader'](image_paths,256,workers=4,cache_path=pack_path,signature=cache_signature)
        assert len(disk_cache.images)==28
        for key in image_paths:np.testing.assert_array_equal(np.asarray(disk_cache(key)),cache.images[key])
    # Cache must not store augmentation: fixed seed gives identical transformed input.
    augmentation=ns['build_transforms'](cfg,training=True)
    ns['seed_everything'](42);original_tensor=augmentation(ns['datasets'].folder.default_loader(image_paths[0]))
    ns['seed_everything'](42);cached_tensor=augmentation(disk_cache(image_paths[0]))
    torch.testing.assert_close(original_tensor,cached_tensor,atol=0,rtol=0)
    with np.load(pack_path,allow_pickle=False) as packed:
        values={key:packed[key] for key in packed.files}
    values['indices']=values['indices'].copy();values['indices'][0]=len(image_paths)+1
    with pack_path.open('wb') as handle:np.savez(handle,**values)
    repaired=ns['CachedImageLoader'](image_paths,256,workers=4,cache_path=pack_path,signature=cache_signature)
    assert len(repaired.images)==28
    limited=ns['CachedImageLoader'](image_paths,48*48*3/1024**2,workers=2)
    assert len(limited.images)==1 and limited.used_bytes==48*48*3
    missed=next(p for p in image_paths if p not in limited.images)
    np.testing.assert_array_equal(np.asarray(limited(missed)),np.asarray(ns['datasets'].folder.default_loader(missed)))
    try:ns['CachedImageLoader']([str(root/'missing.png')],256,workers=2)
    except FileNotFoundError:pass
    else:raise AssertionError('Source-image errors must propagate.')
    assert not any(t.name.startswith('ThreadPoolExecutor-') for t in threading.enumerate())
    print('PASS: exact RGB pixels, independent PIL copies, seeded transform identity, safe disk-pack round trip, corruption rebuild and bounded fallback.')

    (root/'test').rename(root/'held_out')
    loaders,manifest,summary=ns['build_data'](cfg,root,device)
    ram_cache=loaders['train'].dataset.dataset.loader
    with patch.object(ns['datasets'].folder,'default_loader',side_effect=AssertionError('RAM reuse must not decode sources')):
        repeated_loaders,repeated_manifest,_=ns['build_data'](cfg,root,device)
    assert repeated_loaders['train'].dataset.dataset.loader is ram_cache
    assert repeated_manifest.equals(manifest)
    # Re-executing the definitions cell must preserve the session registry.
    data_source=next(c.source for c in nb.cells if 'data_functions' in c.metadata.get('tags',[]))
    saved_registry=ns['IMAGE_CACHE_REGISTRY']
    exec(data_source,ns)
    assert ns['IMAGE_CACHE_REGISTRY'] is saved_registry
    ns['IMAGE_CACHE_REGISTRY'].clear()
    with patch.object(ns['datasets'].folder,'default_loader',side_effect=AssertionError('Reload from pack must not decode sources')):
        packed_loaders,packed_manifest,_=ns['build_data'](cfg,root,device)
    assert packed_manifest.equals(manifest)
    assert packed_loaders['train'].dataset.dataset.loader is not ram_cache
    assert ns['image_cache_signature'](image_paths,{**cfg,'cache_revision':'changed'})!=cache_signature
    assert ns['image_cache_signature'](image_paths,{**cfg,'cache_workers':1})==cache_signature
    original_decoder=ns['decode_rgb_array']
    with patch.dict(ns['iter_decoded_images'].__globals__,decode_rgb_array=Mock(wraps=original_decoder)):
        fresh_loaders,_,_=ns['build_data']({**cfg,'cache_reuse':False},root,device)
        assert ns['decode_rgb_array'].call_count==28
    assert fresh_loaders['train'].dataset.dataset.loader is not ram_cache
    assert not any(str(root/'test') in p for p in ram_cache.images)
    broken_pack=folder/'broken_zip.npz'
    broken_pack.write_bytes(b'not a numpy pack')
    rebuilt=ns['CachedImageLoader'](image_paths,256,workers=2,cache_path=broken_pack,signature=cache_signature)
    assert len(rebuilt.images)==28
    with patch.object(ns['datasets'].folder,'default_loader',side_effect=AssertionError('Rebuilt pack should be reusable')):
        ns['CachedImageLoader'](image_paths,256,workers=2,cache_path=broken_pack,signature=cache_signature)
    print('PASS: same-session RAM reuse, registry survives cell reruns, pack reload avoids source I/O, revision invalidates and reuse=False rebuilds.')
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
            source=cell.source
            if 'final_evaluation' in cell.metadata.get('tags',[]):
                assert 'RUN_FINAL_TEST = False' in source
                source=source.replace('RUN_FINAL_TEST = False','RUN_FINAL_TEST = True')
            exec(compile(source,f'run_all_cell_{i}','exec'),run)
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
    assert metadata['pair_info']['source']=='pilot_validation_window'
    assert metadata['pair_info']['mining_epochs']==[1,2]
    snapshots_saved=json.loads((run_dir/'pilot_validation_snapshots.json').read_text())
    assert len(snapshots_saved)==2
    ranking_csv=pd.read_csv(run_dir/'pilot_pair_ranking.csv')
    assert 'top_k_count' in ranking_csv and 'score_std' in ranking_csv
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
                     'test_metrics.json','test_predictions.csv','training_history.png',
                     'pilot_pair_stability.csv','pilot_mining_confusion_mean.csv',
                     'validation_pair_by_class.csv','test_pair_by_class.csv']:
        assert (run_dir/artifact).is_file(),artifact
    print('PASS: entire notebook executed in order, 2 pilot + 3 main epochs, frozen validation-mined pairs, strict pilot transfer, reports and test.')

    # Default global_ce_224 must finish Run All with its actual resolution and no test directory.
    (root/'test').rename(root/'held_out_again')
    default_run={'__name__':'__default_global_ce_smoke__'}
    default_cfg={**cfg,**ns['EXPERIMENT_PRESETS']['global_ce_224'],
                 'experiment':'global_ce_224','run_name':'smoke_global_ce_224',
                 'main_epochs':2,'main_warmup_epochs':0}
    with patch('matplotlib.pyplot.show'):
        for i,cell in enumerate(nb.cells):
            if cell.cell_type!='code':continue
            exec(compile(cell.source,f'default_cell_{i}','exec'),default_run)
            if 'configuration' in cell.metadata.get('tags',[]):
                default_run['CFG'].update(default_cfg)
                default_run['CFG']['data_dir']=str(root)
                default_run['CFG']['output_root']=str(folder/'default_outputs')
                default_run['DEVICE']=device
    assert default_run['PAIRS']==[] and len(default_run['pilot_history'])==2
    assert default_run['model'].image_size==224
    default_metadata=json.loads((default_run['RUN_DIR']/'training.json').read_text())
    assert default_metadata['model_config']['image_size']==224 and default_metadata['model_config']['head_type']=='linear'
    assert len(default_run['history'])==2 and (default_run['history'].pair_loss==0).all()
    assert 'test_result' not in default_run
    assert not (default_run['RUN_DIR']/'test_metrics.json').exists()
    default_predictions=pd.read_csv(default_run['RUN_DIR']/'validation_predictions.csv')
    assert (default_predictions.prediction==default_predictions.global_prediction).all()
    assert pd.read_csv(default_run['RUN_DIR']/'validation_pair_diagnostics.csv').empty
    print('PASS: default global_ce_224 Run All completes at 224 with no test directory and no pair branch.')

    # Combined option exercises real pilot/main training at 224 with the retained head,
    # and probability .5 (the numeric MixUp labels are checked below).
    option_cfg={**cfg,**ns['EXPERIMENT_PRESETS']['global_ce_224_pretrained_head_mixup50'],
                'experiment':'global_ce_224_pretrained_head_mixup50','run_name':'smoke_hidden224_mixup50',
                'pilot_epochs':1,'main_epochs':1,'pilot_warmup_epochs':0,'main_warmup_epochs':0,
                'pair_mining_window':1}
    assert option_cfg['mixup_probability']==.5
    option_run={'__name__':'__hidden224_run_all__'}
    with patch('matplotlib.pyplot.show'):
        for i,cell in enumerate(nb.cells):
            if cell.cell_type!='code':continue
            exec(compile(cell.source,f'hidden224_cell_{i}','exec'),option_run)
            if 'configuration' in cell.metadata.get('tags',[]):
                option_run['CFG'].update(option_cfg)
                option_run['CFG']['data_dir']=str(root)
                option_run['CFG']['output_root']=str(folder/'hidden_outputs')
                option_run['DEVICE']=device
    assert option_run['model'].image_size==224 and option_run['model'].head_type=='pretrained_mlp'
    assert option_run['history'].train_loss.notna().all()
    option_meta=json.loads((option_run['RUN_DIR']/'training.json').read_text())
    assert option_meta['parameter_count']==1525031 and option_meta['retained_projection_parameters']==590848
    assert not option_meta['retained_projection_initialized_from_imagenet']  # Random local smoke fixture, not a claim of pretraining.
    assert option_meta['config']['mixup_probability']==.5
    assert 'test_result' not in option_run
    print('PASS: combined 224 + retained-head + MixUp .5 Run All, strict phase transfer and full artifact export.')

    # Verify MixUp .5 gate takes mixed/unmixed branches on either side of threshold.
    class MixToy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc=nn.Linear(1,7,bias=False)
            with torch.no_grad():self.fc.weight.zero_();self.fc.weight[0,0]=-4;self.fc.weight[1,0]=4
        def forward(self,x):
            z=self.fc(x)
            return {'logits':z,'global_logits':z,'pair_logits':z.new_zeros((len(x),0)),'gates':z.new_zeros((len(x),0))}
    toy=MixToy();toy_optimizer=torch.optim.SGD(toy.parameters(),lr=0)
    toy_criterion=ns['PairTrainingLoss']([3]*7,[],option_cfg,False)
    toy_loader=[(torch.tensor([[-1.],[1.]]),torch.tensor([0,1]))]
    for coin,expected_accuracy,should_mix in [(.4,.75,True),(.6,1.0,False)]:
        with patch('numpy.random.random',return_value=coin),patch('numpy.random.beta',return_value=.25) as beta, \
             patch('torch.randperm',return_value=torch.tensor([1,0])):
            outcome=ns['train_one_epoch'](toy,toy_loader,toy_optimizer,torch.amp.GradScaler('cuda',enabled=False),
                toy_criterion,device,option_cfg,'main',1)
        assert outcome['accuracy']==expected_accuracy and beta.called==should_mix
    print('PASS: MixUp probability .5 activates only selected batches and preserves weighted two-label accuracy.')

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
