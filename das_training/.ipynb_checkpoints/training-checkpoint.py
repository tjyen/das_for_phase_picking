# das_training/training.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import sys
from tqdm import tqdm

def train_model(model, train_data_loader, device, lr, no_iter=10):
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0)
    
    model.train()
    
    for (i, batch) in enumerate(train_data_loader):
        x = batch['data'].to(device)
        y = batch['phase_pick'].to(device)
        
        result = model({'data':x, 'phase_pick':y})
        h = result['phase'].to(device)
        loss = torch.sum(-y * F.log_softmax(h, dim=1), dim=1).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #print(round(loss.item(), 6))
        
        print(('iteration {}; train loss: {:.6f}').format(i, round(loss.item(), 6)), end="\r", file=sys.stdout, flush=True)
        
        del x, y, result, h, loss
    
        if i % 5 == 0:
            torch.cuda.empty_cache()
            
        if i >= (no_iter-1):
            break
            
    #return model