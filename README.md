# super-resolution

## Getting started
### Download Datasets
- for traning [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) used

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
- You can train the model using train.py
```python
from data import DIV2K
from model.srgan import generator, discriminator
from train import SrganGeneratorTrainer
from train import SrganTrainer

# Pretrain generator
pre_trainer = SrganGeneratorTrainer(model=srgan_gen, checkpoint_dir='./checkpoint/SRGAN', learning_rate=1e-4)
pre_trainer.train(train_dataset=train_dataset, valid_dataset=val_dataset.take(10), steps=100000, evaluate_every=1000, save_best_only=True)
pre_trainer.model.save_weights('weights/srgan/pre_generator.h5')

# Load pretrained weights
srgan_gen.load_weights('weights/srgan/pre_generator.h5')

# trainer
srgan_trainer = SrganTrainer(generator=srgan_gen, discriminator=srgan_dis, content_loss='VGG54')

# Train SRGAN
srgan_trainer.train(train_dataset=train_dataset, steps=200000)

# Save weights
srgan_trainer.generator.save_weights('weights/srgan/gen_generator.h5')
srgan_trainer.discriminator.save_weights('weights/srgan/gen_discriminator.h5')
```
### Test
- load your weights
```python
srgan_model = generator()
srgan_model.load_weights('weights/srgan/gen_generator.h5')
```
- load your file & test
```python
lr = load_image(f'demo/{file}')
hr_srgan = resolve_single(srgan_model, lr)
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].imshow(lr)
axes[0].set_title('LR')
axes[1].imshow(hr_srgan)
axes[1].set_title('SR_SRGAN(x4)')
plt.tight_layout()
plt.show()
```
