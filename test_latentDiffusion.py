import glob, os
from itertools import chain
import numpy as np
from PIL import Image
import torch
from monai.data import DataLoader
from tqdm import tqdm
from src.DenoisingDiffusionProcess.samplers.DDIM import DDIM_Sampler

from train_latentDiffusion import MyLatentDiffusionConditional, get_dataset

# Define constants
BATCH_SIZE = 22 
SHAPE = 512
SAVE_PATH = './lightning_logs/version_2/predictions/'
ddim_STEPS = 500

def load_model(ckpt_path):
    """Load the model from checkpoint."""
    model = MyLatentDiffusionConditional.load_from_checkpoint(ckpt_path, map_location='cpu')
    # model.eval()
    model.ae.to('cuda:0')
    model.diff_model.to('cuda:1')
    assert model.hparams.img_shape == SHAPE
    return model

def get_data_loader(dataset, batch_size):
    """Get data loader."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

def process_batches(dl, model):
    """Process batches."""
    scts, cts, cbcts, paths = [], [], [], []
    with torch.no_grad():
        i = 0
        for batch in tqdm(dl, desc='Processing batches', leave=False):
            cbct = batch['cbct'].to('cuda:0')
            ct = batch['ct']

            model.ae.to('cpu')
            torch.cuda.empty_cache()
            model.ae.to('cuda:0')

            ddim_sampler = DDIM_Sampler(ddim_STEPS, model.diff_model.num_timesteps)
            sct = model.forward_infer(cbct, sampler=ddim_sampler, verbose=True)

            cbcts.append(cbct)
            cts.append(ct)
            scts.append(sct)
            paths.append(ct.meta['filename_or_obj'])

        cbcts = torch.cat(cbcts, dim=0).squeeze()
        cts = torch.cat(cts, dim=0).squeeze()
        scts = torch.cat(scts, dim=0).squeeze()
        paths = list(chain.from_iterable(paths))

    return scts, cbcts, cts, paths

def save_images(scts, cbcts, cts, paths, save_path):
    """Save images to png files."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(scts.shape[0]):
        filename = paths[i].split('/')[-1]

        # sct: Add 0.5 after unnormalizing to [0, 65535] to round to the nearest integer
        # sct =  (scts[i].mul(65535).add(0.5).clamp(0, 65535).to('cpu').numpy().squeeze()).astype(np.uint16)
        # cbct = (cbcts[i].mul(65535).clamp(0, 65535).to('cpu').numpy().squeeze()).astype(np.uint16)
        # ct =   (cts[i].mul(65535).clamp(0, 65535).to('cpu').numpy().squeeze()).astype(np.uint16)

        ndarr = np.hstack([cbcts[i], cts[i], scts[i].clip(0,1)])
        ndarr = (ndarr.squeeze() * 255).astype(np.uint8)
        im = Image.fromarray(ndarr)
        im.save(f"{save_path}/{filename}") 

if __name__ == "__main__":
    # glob the checkpoint files
    # ckpts = glob.glob(LOAD_PATH)
    # ckpts = sorted(ckpts)
    # ckpt = ckpts[-1]
    ckpt = './lightning_logs/version_2/checkpoints/epoch=9569-train_loss=0.000317.ckpt'
    print('Loading checkpoint: ', ckpt)

    model = load_model(ckpt)
    ds = get_dataset('../catphan/', cache_rate=0.0, img_shape=SHAPE, img_range=(0,1))
    dl = get_data_loader(ds, BATCH_SIZE)
    sct, cbct, ct, paths = process_batches(dl, model)
    save_images(sct, cbct, ct, paths, SAVE_PATH)