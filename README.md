# super-resolution

## Getting started
### Download Datasets
- for traning [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

```python
  from data import DIV2K
  from model.srgan import generator, discriminator
  from train import SrganGeneratorTrainer

  DIV2K_dataset = DIV2K

  #Get train dataset
  train_dataset = DIV2K(scale=4, subset='train', images_dir='div2k/images', caches_dir='div2k/caches')
  train_dataset = train_dataset.dataset(batch_size=16)
  val_dataset = DIV2K(scale=4, subset='valid', images_dir='div2k/images', caches_dir='div2k/caches')
  val_dataset = val_dataset.dataset(batch_size=16)
```
### Train
