"""Software checks on artificial images/masks. No clinical performance claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch

import train
import summarize


def fixture(root):
    parts={}
    rng=np.random.default_rng(9)
    for split,n in [('train',8),('validation',8)]:
        records=[]
        for i in range(n):
            image_id=f'{split}_{i}'
            directory=root/image_id
            directory.mkdir()
            Image.fromarray(rng.integers(0,256,(224,224,3),dtype=np.uint8)).save(directory/'image_224.png')
            disc=np.zeros((64,64),dtype=np.uint8);disc[8:56,8:56]=255
            cup=np.zeros_like(disc);cup[20:44,20:44]=255
            Image.fromarray(disc).save(directory/'disc.png');Image.fromarray(cup).save(directory/'cup.png')
            records.append(dict(image_id=image_id,split=split,binary_label=i%2,
                                classification_image=f'{image_id}/image_224.png',
                                segmentation_eligible='1' if i!=0 else '0',
                                disc_mask=f'{image_id}/disc.png',cup_mask=f'{image_id}/cup.png',
                                dataset='chaksu',camera='Synthetic',patient_id='',analysis_group=image_id))
        parts[split]=pd.DataFrame(records)
    return parts


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ratios_and_invalid_masks(self):
        disc=np.ones((8,10));cup=np.zeros((8,10));cup[2:6,2:7]=1
        np.testing.assert_allclose(train.ratios_from_masks(disc,cup),[.5,.5,.25])
        with self.assertRaises(ValueError):train.ratios_from_masks(disc,np.zeros_like(disc))
        with self.assertRaises(ValueError):train.ratios_from_masks(cup,disc)

    def test_all_architecture_shapes_and_inference_inputs(self):
        for experiment in ['E0','E1','E2','E3','E4','E5']:
            config=train.configuration(experiment,2440)
            model=train.AnatomyModel(config,pretrained=False).eval()
            with torch.no_grad():logits,ratios=model(torch.randn(2,3,64,64))
            self.assertEqual(logits.shape,(2,))
            self.assertEqual(ratios is not None,config['auxiliary'])
            if ratios is not None:self.assertTrue(((ratios>=0)&(ratios<=1)).all())

    def test_frozen_backbone_and_batchnorm(self):
        model=train.AnatomyModel(train.configuration('E1',2440),False)
        model.set_phase(True)
        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        model.set_phase(False)
        for name,param in model.backbone.named_parameters():
            self.assertEqual(param.requires_grad,name.startswith('layer4.'))
        self.assertTrue(all(not m.training for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)))

    def test_paired_head_initialization(self):
        models={exp:train.AnatomyModel(train.configuration(exp,2440),False) for exp in ['E0','E3','E4']}
        torch.testing.assert_close(models['E0'].classifier.weight,models['E3'].classifier.weight)
        torch.testing.assert_close(models['E3'].anatomy.weight,models['E4'].anatomy.weight)

    def test_missing_anatomy_does_not_receive_fake_supervision(self):
        labels=torch.tensor([0.,1.]);logits=torch.tensor([.1,.2],requires_grad=True)
        ratios=torch.tensor([[.2,.3,.4],[.6,.7,.8]],requires_grad=True)
        total,_,_=train.losses(logits,ratios,labels,torch.zeros(2,3),torch.tensor([True,False]),.5)
        total.backward()
        self.assertEqual(float(ratios.grad[1].abs().sum()),0.)
        self.assertGreater(float(ratios.grad[0].abs().sum()),0.)

    def test_test_split_is_rejected(self):
        rows=pd.DataFrame([dict(split='external_test')])
        with self.assertRaises(ValueError):train.DevelopmentDataset(rows,Path('.'),train.configuration('E0',2440),False)
        with self.assertRaises(ValueError):train.anatomy_targets(rows,Path('.'))

    def test_tampered_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'all.csv';path.write_text('image_id,split\nx,train\n')
            with self.assertRaises(ValueError):train.load_development(path)

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):train.safe_mask_path(Path('.'),'../../private.png')

    def test_end_to_end_reference_and_fusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parts=fixture(root)
            for experiment in ['E0','E4']:
                config=train.configuration(experiment,2440)
                config.update(epochs=2,warmup_epochs=1,batch_size=4)
                out=root/experiment
                summary=train.run(config,parts,root,out,torch.device('cpu'),pretrained=False,
                                  manifest_hash='SYNTHETIC_TEST_ONLY')
                self.assertTrue(summary['synthetic'])
                self.assertEqual(summary['best_epoch'],2)
                self.assertEqual(json.loads((out/'status.json').read_text())['status'],'complete')
                self.assertEqual(len(pd.read_csv(out/'validation_predictions.csv')),8)
                self.assertFalse((out/'final_evaluation').exists())
                checkpoint=torch.load(out/'best.pt',weights_only=True,map_location='cpu')
                reloaded=train.AnatomyModel(config,False);reloaded.load_state_dict(checkpoint['model_state'])
                _,pred=train.epoch(reloaded,train.loader(parts['validation'],root,config,False,{}),torch.device('cpu'))
                saved=pd.read_csv(out/'validation_predictions.csv')
                np.testing.assert_allclose(pred.probability,saved.probability,rtol=0,atol=1e-7)
                if experiment=='E4':
                    self.assertEqual(json.loads((out/'validation_anatomy_mae.json').read_text())['n'],7)
                with self.assertRaises(FileExistsError):train.run(config,parts,root,out,torch.device('cpu'),False)

    def test_summary_excludes_synthetic_and_checks_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for i,(experiment,seed,synthetic) in enumerate([('E0',2440,False),('E0',2441,False),
                                                         ('E0',2442,False),('E4',2440,True)]):
                path=root/str(i);path.mkdir()
                metrics={key:.5 for key in ['auroc','average_precision','sensitivity','specificity','brier','log_loss']}
                entry=dict(config=train.configuration(experiment,seed),synthetic=synthetic,metrics=metrics,
                           best_epoch=8,threshold=.1,manifest_sha256='test',development_images_sha256='pixels',
                           source_fingerprint='testcode',mask_audit={},outcome='fixture')
                (path/'summary.json').write_text(json.dumps(entry))
                (path/'status.json').write_text(json.dumps(dict(status='complete')))
            runs,comparison,_=summarize.summarize(root)
            self.assertEqual(len(runs),3)
            self.assertTrue(comparison.iloc[0].complete_three_seed_comparison)
            self.assertEqual(comparison.iloc[0].auroc_sd,0)
            duplicate=root/'duplicate';duplicate.mkdir()
            (duplicate/'summary.json').write_text((root/'0/summary.json').read_text())
            (duplicate/'status.json').write_text((root/'0/status.json').read_text())
            with self.assertRaises(ValueError):summarize.summarize(root)


if __name__=='__main__':unittest.main()
