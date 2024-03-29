import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tqdm.autonotebook import tqdm
from data import DIV2K
from model.srgan import generator, discriminator
from train import SrganGeneratorTrainer

DIV2K_dataset = DIV2K

# Get train dataset
train_dataset = DIV2K(scale=4, subset='train', images_dir='div2k/images', caches_dir='div2k/caches')
train_dataset = train_dataset.dataset(batch_size=16)
val_dataset = DIV2K(scale=4, subset='valid', images_dir='div2k/images', caches_dir='div2k/caches')
val_dataset = val_dataset.dataset(batch_size=16)


# num = 0
# for _ in tqdm(train_dataset.repeat(1)):
#     num += 1
# # Get models
# srgan_gen = generator()
# srgan_dis = discriminator()
# #
# # Pretrain generator
# pre_trainer = SrganGeneratorTrainer(model=srgan_gen, checkpoint_dir='./checkpoint/SRGAN', learning_rate=1e-4)
# pre_trainer.train(train_dataset=train_dataset, valid_dataset=val_dataset.take(10), steps=100000, evaluate_every=1000, save_best_only=True)
# pre_trainer.model.save_weights('weights/srgan/pre_generator.h5')
#
# # Load pretrained weights
# srgan_gen.load_weights('weights/srgan/pre_generator.h5')
#
# # trainer
# srgan_trainer = SrganTrainer(generator=srgan_gen, discriminator=srgan_dis, content_loss='VGG54')
#
# # Train SRGAN
# srgan_trainer.train(train_dataset=train_dataset, steps=200000)
#
# # Save weights
# srgan_trainer.generator.save_weights('weights/srgan/gen_generator.h5')
# srgan_trainer.discriminator.save_weights('weights/srgan/gen_discriminator.h5')