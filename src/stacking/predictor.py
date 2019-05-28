import torch
import numpy as np
from torch.utils.data import DataLoader

from argus import load_model


class StackPredictor:
    def __init__(self, model_path,
                 batch_size, device='cuda'):
        self.model = load_model(model_path, device=device)
        self.batch_size = batch_size

    def predict(self, probs):
        probs = probs.copy()
        stack_tensors = [torch.from_numpy(prob.astype(np.float32))
                         for prob in probs]

        loader = DataLoader(stack_tensors, batch_size=self.batch_size)

        preds_lst = []
        for probs_batch in loader:
            pred_batch = self.model.predict(probs_batch)
            preds_lst.append(pred_batch)

        pred = torch.cat(preds_lst, dim=0)

        return pred.cpu().numpy()
