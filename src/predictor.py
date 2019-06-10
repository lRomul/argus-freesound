import multiprocessing as mp
import torch
from torch.utils.data import DataLoader

from argus import load_model

from src.tiles import ImageSlicer


N_WORKERS = mp.cpu_count()


@torch.no_grad()
def tile_prediction(model, image, transforms,
                    tile_size, tile_step, batch_size):
    tiler = ImageSlicer(image.shape,
                        tile_size=tile_size,
                        tile_step=tile_step)

    tiles = tiler.split(image, value=float(image.min()))
    tiles = [transforms(tile) for tile in tiles]

    loader = DataLoader(tiles, batch_size=batch_size)

    preds_lst = []

    for tiles_batch in loader:
        pred_batch = model.predict(tiles_batch)
        preds_lst.append(pred_batch)

    pred = torch.cat(preds_lst, dim=0)

    return pred.cpu().numpy()


class Predictor:
    def __init__(self, model_path, transforms,
                 batch_size, tile_size, tile_step,
                 device='cuda'):
        self.model = load_model(model_path, device=device)
        self.transforms = transforms
        self.tile_size = tile_size
        self.tile_step = tile_step
        self.batch_size = batch_size

    def predict(self, image):
        pred = tile_prediction(self.model, image, self.transforms,
                               self.tile_size,
                               self.tile_step,
                               self.batch_size)
        return pred

    def tile_image(self, image):
        tiler = ImageSlicer(image.shape,
                            tile_size=self.tile_size,
                            tile_step=self.tile_step)

        tiles = tiler.split(image, value=float(image.min()))
        return tiles

    def predict_tiles(self, tiles):
        tiles = [self.transforms(tile) for tile in tiles]

        loader = DataLoader(tiles, batch_size=self.batch_size)

        preds_lst = []

        for tiles_batch in loader:
            pred_batch = self.model.predict(tiles_batch)
            preds_lst.append(pred_batch)

        pred = torch.cat(preds_lst, dim=0)

        return pred.cpu().numpy()
