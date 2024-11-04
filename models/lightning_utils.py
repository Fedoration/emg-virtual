import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


import torch
def upload_weights_pl(model, path):
    class Lit_Wrapper(pl.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            x = self.model(x)
            return x
        
    ckpt = torch.load(path)
    model_pl = Lit_Wrapper(model)
    model_pl.load_state_dict(ckpt['state_dict'])
    
    return model_pl.model


class LitHVATNet_v2(pl.LightningModule):
    def __init__(self, model, loss_function, lr, wd):
        """
        Wrapper of model with loss function calculatino and initing optimizer.
        """
        super().__init__()
        self.model = model
        self.lr = lr
        self.wd = wd
        self.loss_function = loss_function # should compare quats. 
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.lr, 
                                      weight_decay = self.wd)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        multi_scale_quats = self.model(x)
        
        # loss on full size
        full_size_pred = multi_scale_quats[-1]
        loss_dict = self.loss_function(full_size_pred, y)
        
        # average loss on multi scale outputs
        # for interpolation purpose 
        batch, n_bones, n_quats, time = y.shape
        y_3d = y.reshape(batch, -1, time)
        
        ms_losses = []
        for feat in multi_scale_quats[:-1]:  
            
            new_time_size = feat.shape[-1]
            y_ = F.interpolate(y_3d, size=new_time_size, mode='linear')
            y_ = y_.reshape(batch, n_bones, n_quats, new_time_size)
            
            loss_dict_tmp = self.loss_function(feat, y_)
            ms_losses.append(loss_dict_tmp['total_loss'])
        
        ms_loss = torch.mean(torch.stack(ms_losses))
        loss_dict['total_loss'] += ms_loss
        
        for k, v in loss_dict.items():
            self.log("train_" + str(k), v, on_step=False, on_epoch=True,  sync_dist=True)
        return loss_dict['total_loss']
            
    def validation_step(self, val_batch, batch_idx):
        if trainer.global_step == 0: 
            wandb.define_metric('val_angle_degree', summary='min')
        
        x, y = val_batch
        full_size_pred = self.model(x)
        loss_dict = self.loss_function(full_size_pred, y)
                
        for k, v in loss_dict.items():
            self.log("val_" + str(k), v, on_step=False, on_epoch=True,  sync_dist=True)
        return loss_dict['angle_degree']
    
    
    def validation_epoch_end(self, validation_step_outputs):
        val_current_loss = torch.mean(torch.stack(validation_step_outputs))
        self.val_current_loss = val_current_loss
        print(f'current step {self.current_epoch} val_current_loss {self.val_current_loss}')
            
        
        
        
    