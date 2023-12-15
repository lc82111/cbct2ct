from pathlib import Path

from monai import transforms
from monai.data import CacheDataset, DataLoader
from monai.transforms import Transform, RandomizableTransform, MapTransform
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.utils import convert_to_tensor, ensure_tuple, ensure_tuple_rep
from monai.data.meta_obj import get_track_meta
from monai.utils import ensure_tuple

from typing import Callable, Hashable, Mapping, Sequence
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, LearningRateFinder, BatchSizeFinder
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from src import *

from collections import defaultdict
from skimage.draw import disk, ellipse, rectangle

import numpy as np
import torch


class RandomShapes(Transform):
    """
    Applies random shapes with varying sizes and locations to each image in a dictionary.
    Supports both NumPy arrays and PyTorch tensors.
    """

    def __init__(self,
                 prob: float = 1.0,
                 min_radius: int = 10,
                 max_radius: int = 60,
                 min_num_circles: int = 10,
                 max_num_circles: int = 30,
                 min_replace_value: float = 0,
                 max_replace_value: float = 1, 
                 ):
        super().__init__()
        self.prob = prob
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_num_circles = min_num_circles
        self.max_num_circles = max_num_circles or self.min_num_circles
        self.min_replace_value = min_replace_value
        self.max_replace_value = max_replace_value
        self.rng = np.random.RandomState()

    def _generate_circle_location(self, data_shape):
        max_dim = max(data_shape)
        while True:
            x = self.rng.randint(0, max_dim)
            y = self.rng.randint(0, max_dim)
            radius = self.rng.randint(self.min_radius, self.max_radius + 1)
            if x + radius <= max_dim and y + radius <= max_dim:
                return (x, y)

    def _generate_circles(self, data_shape):
        circles = []
        num_circles = self.rng.randint(self.min_num_circles, self.max_num_circles + 1)
        for _ in range(num_circles):
            if self.rng.random() < self.prob:
                center = self._generate_circle_location(data_shape)
                radius = self.rng.randint(self.min_radius, self.max_radius + 1)
                replace_value = self.rng.uniform(self.min_replace_value, self.max_replace_value)
                circles.append((center, radius, replace_value))
        return circles

    def _generate_mask(self, data_shape, center, radius):
        """
        Generates a mask randomly from disk, elipse, and rectangle  using skimage.draw.
        """
        r_radius = self.rng.randint(radius-5, radius+5)
        c_radius = self.rng.randint(radius-5, radius+5)

        # random shape
        rand_value = self.rng.randint(0, 3)
        if rand_value == 0:
            rr, cc = ellipse(center[0], center[1], r_radius, c_radius, shape=data_shape, rotation=self.rng.randint(-np.pi, np.pi))
        elif rand_value == 1:
            rr, cc = rectangle(start=(center[0]-c_radius, center[1]-r_radius), end=(center[0]+c_radius, center[1]+r_radius), shape=data_shape)
        else:
            rr, cc = disk(center, radius, shape=data_shape)

        mask = np.zeros(data_shape, dtype=bool)
        mask[rr, cc] = True

        return mask

    def __call__(self, data):
        transformed_data = defaultdict(dict)
        for key, image in data.items():
            if isinstance(image, torch.Tensor):
                image = image.cpu().detach().numpy()  # Convert to NumPy for mask generation

            image_shape = image.shape

            transformed_image = image.copy()
            for center, radius, replace_value in self._generate_circles(image_shape):
                mask = self._generate_circle_mask(image_shape, center, radius)
                transformed_image[mask] = replace_value

            if isinstance(transformed_image, np.ndarray):
                transformed_image = torch.from_numpy(transformed_image)  # Convert back to PyTorch tensor

            transformed_data[key] = transformed_image

        self._generated_circles = None
        return transformed_data

def get_dataset(path='./catphan/', cache_rate=1.0, img_shape=512, in_range=(0,65535), out_range=(0, 1), aug_prob=0.5):
    # Step 1: Create a list of image pairs
    ct_dir = Path(path)/'ct'
    cbct_dir = Path(path)/'cbct'

    image_fn_pairs = [{'ct': os.path.join(ct_dir, img), 'cbct': os.path.join(cbct_dir, img)} 
                for img in os.listdir(ct_dir) if '_' not in img]

    # Step 2: Define the transformations
    keys = ['ct', 'cbct']
    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=keys),
            transforms.EnsureChannelFirstd(keys=keys),
            transforms.EnsureTyped(keys=keys, dtype=torch.float32),
            # transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=(64, 64, 64)),
            # transforms.RandSpatialCropd(keys=["cbct", "ct"], roi_size=(64, 64, 1), random_size=False),
            transforms.ScaleIntensityRanged(keys=keys, a_min=in_range[0], a_max=in_range[1], b_min=out_range[0], b_max=out_range[1], clip=True),
            transforms.Resized(keys=keys, spatial_size=(img_shape, img_shape), mode="bilinear"),
            RandomShapes(prob=aug_prob, min_radius=10, max_radius=60, min_num_circles=10, max_num_circles=35, min_replace_value=0, max_replace_value=1),
            transforms.RandAffined(keys=keys, mode=("bilinear", "nearest"), prob=aug_prob, padding_mode="zeros", cache_grid=True,
                                   rotate_range=(np.pi, np.pi*2),
                                #    scale_range=(-0.1, 0.1),
                                   translate_range=(0, 50),
                                    ),
            # transforms.RandShiftIntensityd(keys=keys, offsets=(-0.1, 0.1), prob=aug_prob),
            # transforms.RandCoarseDropoutd(keys=keys, holes=50, spatial_size=20, fill_value=0, prob=aug_prob),
        ]
    )

    # Step 3: Create CacheDataset and DataLoader
    train_dataset = CacheDataset(data=image_fn_pairs, transform=train_transforms, cache_rate=cache_rate, num_workers=4)
    return train_dataset

class MyLatentDiffusionConditional(pl.LightningModule):
    def __init__(self,
                 data_path,
                 in_range=(0,65535),
                 out_range=(0,1),
                 aug_prob=0.5,
                 num_timesteps=1000,
                 batch_size=1,
                 lr=1e-4,
                 img_shape=256,
                 img_range=(0,1)
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.register_buffer('mean', torch.tensor(0.0))
        self.register_buffer('std', torch.tensor(1.0))
        
        self.ae=AutoEncoder()
        with torch.no_grad():
            self.latent_dim=self.ae.encode(torch.ones(1,3,img_shape,img_shape)).shape[1]

        self.diff_model = DenoisingDiffusionConditionalProcess(generated_channels=self.latent_dim,
                                                        condition_channels=self.latent_dim,
                                                        num_timesteps=num_timesteps)
        
    def input_T(self, input):
        # By default, let the model accept samples in [0,1] range, and transform them to [-1,1] automatically
        return (input.repeat(1,3,1,1).clip(0,1).mul_(2)).sub_(1)

    def output_T(self, input):
        # Inverse transform of model output from [-1,1] to [0,1] range with channel mean
        return (input.add_(1)).div_(2).mean(dim=1, keepdim=True)

    def on_train_start(self) -> None:
        with torch.no_grad():
            # get the all batches from the train_dataloader
            latents = []
            for batch in self.trainer.train_dataloader:
                ct = batch['ct'].to(self.device)
                latent = self.ae.encode(self.input_T(ct)).detach()
                latents.append(latent)

            # determin the mean and std so the latent code can be scaled to mean 0 and std 1
            latents = torch.cat(latents, dim=0)
            self.mean = latents.mean()
            self.std = latents.std()

    @torch.no_grad()
    def forward(self, condition, *args, **kwargs):
        condition_latent = self.ae.encode(self.input_T(condition.to(self.device))).detach()
        condition_latent = (condition_latent-self.mean) / self.std
        
        output_latent = self.diff_model(condition_latent, *args, **kwargs)
        output_latent = output_latent * self.std + self.mean

        return self.output_T(self.ae.decode(output_latent))

    @torch.no_grad()
    def forward_infer(self, condition, *args, **kwargs):
        condition_latent = self.ae.encode(self.input_T(condition.to('cuda:0'))).detach()
        condition_latent = (condition_latent-self.mean) / self.std
        
        output_latent = self.diff_model(condition_latent.to('cuda:1'), *args, **kwargs)
        output_latent = output_latent * self.std + self.mean

        return self.output_T(self.ae.decode(output_latent.to('cuda:0')))
    
    def training_step(self, batch, batch_idx):   
        condition, output = batch['cbct'], batch['ct']
        with torch.no_grad():
            latents = self.ae.encode(self.input_T(output)).detach()
            latents = (latents - self.mean) / self.std

            latents_condition = self.ae.encode(self.input_T(condition)).detach()
            latents_condition = (latents_condition - self.mean) / self.std

        loss = self.diff_model.p_loss(latents, latents_condition)
        
        self.log('train_loss',loss)
        return loss

    def on_train_epoch_end(self) -> None:
        if self.current_epoch % 100 == 0 and self.current_epoch >= 0: # Log every 100 epochs 
            self.log_images()

    @torch.no_grad()
    def log_images(self):
        # get a batch of data from the train_dataloader
        batch = next(iter(self.trainer.train_dataloader))
        ct = batch['ct'].to(self.device)[0:8]  # log 8 random slcie
        cbct = batch['cbct'].to(self.device)[0:8]

        sct = self.forward(cbct, verbose=True)

        tb_logger = self.logger.experiment
        tb_logger.add_images("CBCT", cbct, self.current_epoch)
        tb_logger.add_images("CT", ct, self.current_epoch)
        tb_logger.add_images("sCT", sct.clip(0,1), self.current_epoch)

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> Any:
        cbct = batch["cbct"] 
        ct = batch["ct"]

        sct = self.forward(cbct, ct, verbose=True)
        sct = sct.clip(0, 1)
        return sct, ct.meta['filename_or_obj']

    def train_dataloader(self):
        self.train_dataset = get_dataset(self.hparams.data_path, 1, img_shape=self.hparams.img_shape,
                                          aug_prob=self.hparams.aug_prob,
                                          in_range=self.hparams.in_range,
                                          out_range=self.hparams.out_range,
                                          )
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=4,
                          drop_last=False,
                          persistent_workers=True)
 
    def configure_optimizers(self):
        return  torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.diff_model.parameters())), lr=self.hparams.lr)


if __name__ == "__main__":
    data_path = './catphan_betterReg/'  
    max_epochs = 50000 
    lr = 1e-4  # 1e-4
    batch_size = 32
    img_shape = 256
    aug_prob = 0.5
    in_range = (29500,36000)
    out_range = (0,1)
    devices = [0,1,2,3]
    accumulate_grad_batches = 1

    model = MyLatentDiffusionConditional(data_path=data_path, aug_prob=aug_prob,
                                          lr=lr, batch_size=batch_size, in_range=in_range,
                                          out_range=out_range, img_shape=img_shape)
    
    # resume the training
    # model = MyLatentDiffusionConditional.load_from_checkpoint(
    #                                 './lightning_logs/version_1/checkpoints/epoch=9952-train_loss=0.001200.ckpt',
    #                                 lr=lr)

    ckp_cb = ModelCheckpoint(save_top_k=3, monitor='train_loss', mode='min',
                              filename='{epoch}-{train_loss:.6f}', save_on_train_epoch_end=True, verbose=True)
    lr_cb =  LearningRateMonitor(logging_interval='epoch') 

    # find maximum values for learning rate and batch size
    # bsf_cb = BatchSizeFinder(mode='binsearch', init_val=16)
    # lrf_cb = LearningRateFinder(min_lr=1e-6, max_lr=1e-2, num_training_steps=100, update_attr=True)

    ddp = DDPStrategy(process_group_backend='gloo', find_unused_parameters=True)
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[EMA(0.9999), ckp_cb, lr_cb],
                          accelerator='gpu', devices=devices, strategy=ddp,
                          log_every_n_steps=12, accumulate_grad_batches=accumulate_grad_batches,
                        )

    trainer.fit(model)