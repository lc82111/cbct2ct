from pathlib import Path
from monai import transforms
from monai.data import CacheDataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, LearningRateFinder, BatchSizeFinder
import pytorch_lightning as pl
from src import *


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
                          num_workers=0,
                          drop_last=True,)
 
    def configure_optimizers(self):
        return  torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.diff_model.parameters())), lr=self.hparams.lr)


if __name__ == "__main__":
    data_path = './catphan_betterReg/'  
    max_epochs = 50000 
    lr = 2.5e-5  # 1e-4
    batch_size = 24 
    img_shape = 512 
    aug_prob = 0.5
    in_range = (29500,36000)
    out_range = (0,1)
    devices = [0,1]
    accumulate_grad_batches = 2

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

    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[EMA(0.9999), ckp_cb, lr_cb],
                          accelerator='gpu', devices=devices, strategy='ddp_find_unused_parameters_true',
                          log_every_n_steps=12, accumulate_grad_batches=accumulate_grad_batches,
                        )

    trainer.fit(model)