"""
Self-supervised pre-training (k-fold) with Lightly.

Supports 8 methods: SimCLR, MoCo, SimSiam, MAE, NNCLR, SimMIM, BYOL, DINOv2.
Pre-training uses only the `train` split of each fold (labels are ignored);
the `val` split is used to track the SSL loss and select the best checkpoint.

Example
-------
    python -m selfsupervised.train --model_type simclr --data_path /data/k_fold \
        --checkpoint_dir ./checkpoints/selfsupervised --folds 1,2,3,4,5

Model definitions live in selfsupervised/models.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import copy
from functools import partial

# Lightly imports
from lightly.loss import NTXentLoss as LightlyNTXentLoss
from lightly.loss import NegativeCosineSimilarity, DINOLoss, IBOTPatchLoss, KoLeoLoss
from lightly.models.utils import (
    deactivate_requires_grad, 
    update_momentum, 
    random_block_mask,
    update_drop_path_rate
)
from lightly.models import utils as lightly_utils
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.transforms import SimSiamTransform, MAETransform
from lightly.transforms.byol_transform import BYOLTransform, BYOLView1Transform, BYOLView2Transform
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule, linear_warmup_schedule
from lightly.data import LightlyDataset

# Model classes (see selfsupervised/models.py)
from selfsupervised.models import (
    SimCLRModel,
    MoCoModel,
    SimSiamModel,
    MAEModel,
    NNCLRModel,
    SimMIMModel,
    BYOLModel,
    DINOv2Model
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# Loss Functions
# ============================================================================

class NTXentLoss(nn.Module):
    """NT-Xent loss for SimCLR"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.loss_fn = LightlyNTXentLoss(temperature=temperature)
    
    def forward(self, z_i, z_j):
        return self.loss_fn(z_i, z_j)


class MoCoLoss(nn.Module):
    """MoCo Loss with memory bank"""
    def __init__(self, temperature=0.2, memory_bank_size=(4096, 128)):
        super().__init__()
        self.loss_fn = LightlyNTXentLoss(
            temperature=temperature,
            memory_bank_size=memory_bank_size
        )
    
    def forward(self, query, key):
        return self.loss_fn(query, key)


class SimSiamLoss(nn.Module):
    """SimSiam Loss using Lightly's NegativeCosineSimilarity"""
    def __init__(self):
        super(SimSiamLoss, self).__init__()
        self.criterion = NegativeCosineSimilarity()
    
    def forward(self, z0, p1, z1, p0):
        # Lightly's loss: 0.5 * (D(z0, p1) + D(z1, p0))
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return loss


class MAELoss(nn.Module):
    """MAE Loss"""
    def __init__(self, patch_size=16):
        super(MAELoss, self).__init__()
        self.patch_size = patch_size
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        return self.mse(predictions, targets)


class NNCLRLoss(nn.Module):
    """NNCLR Loss"""
    def __init__(self, temperature=0.2):
        super(NNCLRLoss, self).__init__()
        self.temperature = temperature
        self.criterion = LightlyNTXentLoss(temperature=temperature)
    
    def forward(self, z1_nn, z2_nn, p1, p2):
        # Loss = 0.5 * (criterion(z1_nn, p2) + criterion(z2_nn, p1))
        loss = 0.5 * (self.criterion(z1_nn, p2) + self.criterion(z2_nn, p1))
        return loss


class SimMIMLoss(nn.Module):
    """SimMIM Loss"""
    def __init__(self):
        super(SimMIMLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        return self.mse(predictions, targets)


class BYOLLoss(nn.Module):
    """BYOL Loss using Negative Cosine Similarity"""
    def __init__(self):
        super(BYOLLoss, self).__init__()
        self.criterion = NegativeCosineSimilarity()
    
    def forward(self, p0, z1, p1, z0):
        """
        Args:
            p0: prediction from view 0
            z1: projection from view 1 (target)
            p1: prediction from view 1
            z0: projection from view 0 (target)
        """
        # Lightly's loss: 0.5 * (criterion(p0, z1) + criterion(p1, z0))
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        return loss


class DINOv2Loss(nn.Module):
    """DINOv2 Combined Loss (DINO + iBOT + KoLeo)"""
    def __init__(self):
        super(DINOv2Loss, self).__init__()
        self.dino_criterion = DINOLoss()
        self.ibot_criterion = IBOTPatchLoss()
        self.koleo_criterion = KoLeoLoss()
    
    def forward(self, teacher_cls_out, student_cls_out, 
                teacher_masked_out, student_masked_out,
                block_mask, student_global_cls_token,
                teacher_temp, num_views):
        """
        Compute combined DINOv2 loss
        
        Args:
            teacher_cls_out: Teacher CLS token outputs
            student_cls_out: Student CLS token outputs (global + local)
            teacher_masked_out: Teacher masked patch outputs
            student_masked_out: Student masked patch outputs
            block_mask: Block mask for iBOT
            student_global_cls_token: Student global CLS tokens (for KoLeo)
            teacher_temp: Teacher temperature
            num_views: Number of views (for chunking)
        """
        # DINO loss
        dino_loss = self.dino_criterion(
            teacher_out=teacher_cls_out.chunk(2),
            student_out=student_cls_out.chunk(num_views),
            teacher_temp=teacher_temp,
        )
        
        # iBOT loss
        ibot_loss = self.ibot_criterion(
            teacher_out=teacher_masked_out,
            student_out=student_masked_out,
            mask=block_mask,
            teacher_temp=teacher_temp,
        )
        
        # KoLeo regularization loss
        koleo_loss = 0.1 * sum(
            self.koleo_criterion(t) for t in student_global_cls_token.chunk(2)
        )
        
        total_loss = dino_loss + ibot_loss + koleo_loss
        
        return total_loss, dino_loss, ibot_loss, koleo_loss


# ============================================================================
# Data Transformations
# ============================================================================

def get_transform(model_type, size=224):
    """Get transformation based on model type"""
    if model_type == 'simclr':
        return SimCLRTransform(input_size=size, gaussian_blur=0.0)
    elif model_type == 'moco':
        return MoCoV2Transform(input_size=size)
    elif model_type == 'simsiam':
        return SimSiamTransform(input_size=size)
    elif model_type in ['mae', 'simmim']:
        return MAETransform(input_size=size)
    elif model_type == 'nnclr':
        return SimCLRTransform(input_size=size, gaussian_blur=0.0)
    elif model_type == 'byol':
        return BYOLTransform(
            view_1_transform=BYOLView1Transform(input_size=size, gaussian_blur=0.0),
            view_2_transform=BYOLView2Transform(input_size=size, gaussian_blur=0.0)
        )
    elif model_type == 'dinov2':
        return DINOTransform(
            global_crop_scale=(0.32, 1.0),
            local_crop_scale=(0.05, 0.32),
            n_local_views=8,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch_simclr(model, train_loader, criterion, optimizer, epoch):
    """Train SimCLR for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        (x_i, x_j), _ = batch
        
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        
        _, z_i = model(x_i)
        _, z_j = model(x_j)
        
        loss = criterion(z_i, z_j)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def train_epoch_moco(model, train_loader, criterion, optimizer, epoch, total_epochs):
    """Train MoCo for one epoch"""
    model.train()
    total_loss = 0
    
    # Calculate momentum value for this epoch
    momentum_val = cosine_schedule(epoch - 1, total_epochs, 0.996, 1)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        (x_query, x_key), _ = batch
        
        x_query = x_query.to(device)
        x_key = x_key.to(device)
        
        # Update momentum encoder
        update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
        update_momentum(model.projection_head, model.projection_head_momentum, m=momentum_val)
        
        # Forward pass
        query = model(x_query)
        key = model.forward_momentum(x_key)
        
        loss = criterion(query, key)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def train_epoch_simsiam(model, train_loader, criterion, optimizer, epoch):
    """Train SimSiam for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        (x1, x2), _ = batch
        x1 = x1.to(device)
        x2 = x2.to(device)
        
        _, z1, z2, p1, p2 = model(x1, x2)
        
        # Loss: 0.5 * (D(z1, p2) + D(z2, p1))
        loss = criterion(z1, p2, z2, p1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def train_epoch_mae(model, train_loader, criterion, optimizer, epoch):
    """Train MAE for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # MAETransform returns (views, target)
        views, _ = batch
        imgs = views[0].to(device)  # Single view
        
        # MAE returns (predictions, targets, encoded)
        predictions, targets, _ = model(imgs)
        loss = criterion(predictions, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def train_epoch_nnclr(model, train_loader, criterion, optimizer, epoch):
    """Train NNCLR for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        (x1, x2), _ = batch
        x1 = x1.to(device)
        x2 = x2.to(device)
        
        # Returns (h, z_nn1, z_nn2, p1, p2)
        _, z1_nn, z2_nn, p1, p2 = model(x1, x2)
        # Loss: 0.5 * (criterion(z1_nn, p2) + criterion(z2_nn, p1))
        loss = criterion(z1_nn, z2_nn, p1, p2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def train_epoch_simmim(model, train_loader, criterion, optimizer, epoch):
    """Train SimMIM for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # MAETransform returns (views, target)
        views, _ = batch
        imgs = views[0].to(device)  # Single view
        
        # SimMIM returns (predictions, targets, encoded)
        predictions, targets, _ = model(imgs)
        loss = criterion(predictions, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def train_epoch_byol(model, train_loader, criterion, optimizer, epoch, total_epochs):
    """Train BYOL for one epoch"""
    model.train()
    total_loss = 0
    
    # Calculate momentum value for this epoch
    momentum_val = cosine_schedule(epoch - 1, total_epochs, 0.996, 1)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        (x0, x1), _ = batch
        x0 = x0.to(device)
        x1 = x1.to(device)
        
        # Update momentum encoder
        update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
        update_momentum(model.projection_head, model.projection_head_momentum, m=momentum_val)
        
        # Forward pass
        p0 = model(x0)
        z0 = model.forward_momentum(x0)
        p1 = model(x1)
        z1 = model.forward_momentum(x1)
        
        # Compute loss
        loss = criterion(p0, z1, p1, z0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def train_epoch_dinov2(model, train_loader, criterion, optimizer, epoch, total_epochs):
    """Train DINOv2 for one epoch"""
    model.train()
    # Keep teacher in eval mode
    model.teacher_backbone.eval()
    model.teacher_head.eval()
    
    total_loss = 0
    num_batches = len(train_loader)
    total_steps = total_epochs * num_batches
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        views, _ = batch
        views = [view.to(device) for view in views]
        global_views = torch.cat(views[:2])  # First 2 views are global
        local_views = torch.cat(views[2:])   # Rest are local views
        
        # Calculate global step
        global_step = (epoch - 1) * num_batches + batch_idx
        
        # Create block mask for iBOT
        B = len(global_views)
        sequence_length = model.teacher_backbone.sequence_length
        mask = global_views.new_zeros((B, sequence_length), dtype=torch.bool)
        
        # Mask patches except class token
        H, W = model.teacher_backbone.vit.patch_embed.grid_size
        block_mask = random_block_mask(size=(B, H, W), device=mask.device)
        mask[:, 1:] = block_mask.flatten(start_dim=1)
        
        # Teacher forward (no gradients)
        with torch.no_grad():
            teacher_cls_token, teacher_features = model.forward_teacher(global_views)
            teacher_cls_out = model.teacher_head.dino_head.forward(teacher_cls_token)
            teacher_masked_out = model.teacher_head.ibot_head.forward(
                teacher_features[mask]
            )
        
        # Student forward
        student_global_cls_token, student_global_masked_features = model.forward_student(
            global_views, mask=mask
        )
        student_global_cls_out = model.student_head.dino_head.forward(
            student_global_cls_token
        )
        student_global_masked_out = model.student_head.ibot_head.forward(
            student_global_masked_features
        )
        
        student_local_cls_token, _ = model.forward_student(local_views, mask=None)
        student_local_cls_out = model.student_head.dino_head.forward(
            student_local_cls_token
        )
        
        student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out])
        
        # Calculate teacher temperature with warmup
        teacher_temp = linear_warmup_schedule(
            step=global_step,
            warmup_steps=int(30 / total_epochs * total_steps),
            start_value=0.04,
            end_value=0.07,
        )
        
        # Calculate loss
        loss, dino_loss, ibot_loss, koleo_loss = criterion(
            teacher_cls_out=teacher_cls_out,
            student_cls_out=student_cls_out,
            teacher_masked_out=teacher_masked_out,
            student_masked_out=student_global_masked_out,
            block_mask=block_mask,
            student_global_cls_token=student_global_cls_token,
            teacher_temp=teacher_temp,
            num_views=len(views),
        )
        
        # Optionally freeze last layer learning rate in early epochs
        if epoch < 2:
            for param_group in optimizer.param_groups:
                if "last_layer" in param_group:
                    param_group["lr"] = 0.0
        
        # Weight decay schedule
        weight_decay = cosine_schedule(
            step=global_step,
            max_steps=total_steps,
            start_value=0.04,
            end_value=0.4,
        )
        
        # Update weight decay for all parameter groups
        for group in optimizer.param_groups:
            if group["weight_decay"] != 0.0:
                group["weight_decay"] = weight_decay
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Momentum update teacher
        momentum = cosine_schedule(
            step=global_step,
            max_steps=total_steps,
            start_value=0.992,
            end_value=1.0,
        )
        update_momentum(model.student_backbone, model.teacher_backbone, m=momentum)
        update_momentum(model.student_head, model.teacher_head, m=momentum)
        
        total_loss += loss.item()
        pbar.set_postfix({
            'loss': loss.item(),
            'dino': dino_loss.item(),
            'ibot': ibot_loss.item(),
            'koleo': koleo_loss.item()
        })
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate_generic(model_type, model, val_loader, criterion, total_epochs=None, current_epoch=None):
    """Generic validation function"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(val_loader, desc='Validation')):
            if model_type in ['mae', 'simmim']:
                # MAETransform (used by both MAE and SimMIM)
                views, _ = batch_data
                imgs = views[0].to(device)
                
                predictions, targets, _ = model(imgs)
                loss = criterion(predictions, targets)
            elif model_type == 'dinov2':
                # DINOv2 validation
                views, _ = batch_data
                views = [view.to(device) for view in views]
                global_views = torch.cat(views[:2])
                local_views = torch.cat(views[2:])
                
                # Calculate global step for validation
                num_batches = len(val_loader)
                total_steps = total_epochs * num_batches if total_epochs else num_batches
                global_step = ((current_epoch - 1) * num_batches + batch_idx) if current_epoch else batch_idx
                
                # Create mask
                B = len(global_views)
                sequence_length = model.teacher_backbone.sequence_length
                mask = global_views.new_zeros((B, sequence_length), dtype=torch.bool)
                H, W = model.teacher_backbone.vit.patch_embed.grid_size
                block_mask = random_block_mask(size=(B, H, W), device=mask.device)
                mask[:, 1:] = block_mask.flatten(start_dim=1)
                
                # Forward pass
                teacher_cls_token, teacher_features = model.forward_teacher(global_views)
                teacher_cls_out = model.teacher_head.dino_head.forward(teacher_cls_token)
                teacher_masked_out = model.teacher_head.ibot_head.forward(teacher_features[mask])
                
                student_global_cls_token, student_global_masked_features = model.forward_student(
                    global_views, mask=mask
                )
                student_global_cls_out = model.student_head.dino_head.forward(student_global_cls_token)
                student_global_masked_out = model.student_head.ibot_head.forward(
                    student_global_masked_features
                )
                
                student_local_cls_token, _ = model.forward_student(local_views, mask=None)
                student_local_cls_out = model.student_head.dino_head.forward(student_local_cls_token)
                student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out])
                
                # Teacher temperature
                teacher_temp = linear_warmup_schedule(
                    step=global_step,
                    warmup_steps=int(30 / (total_epochs or 1) * total_steps),
                    start_value=0.04,
                    end_value=0.07,
                )
                
                # Calculate loss
                loss, _, _, _ = criterion(
                    teacher_cls_out=teacher_cls_out,
                    student_cls_out=student_cls_out,
                    teacher_masked_out=teacher_masked_out,
                    student_masked_out=student_global_masked_out,
                    block_mask=block_mask,
                    student_global_cls_token=student_global_cls_token,
                    teacher_temp=teacher_temp,
                    num_views=len(views),
                )
            else:
                (x_i, x_j), _ = batch_data
                x_i = x_i.to(device)
                x_j = x_j.to(device)
                
                if model_type == 'simclr':
                    _, z_i = model(x_i)
                    _, z_j = model(x_j)
                    loss = criterion(z_i, z_j)
                elif model_type == 'moco':
                    query = model(x_i)
                    key = model.forward_momentum(x_j)
                    loss = criterion(query, key)
                elif model_type == 'simsiam':
                    _, z1, z2, p1, p2 = model(x_i, x_j)
                    loss = criterion(z1, p2, z2, p1)
                elif model_type == 'nnclr':
                    _, z1_nn, z2_nn, p1, p2 = model(x_i, x_j)
                    loss = criterion(z1_nn, z2_nn, p1, p2)
                elif model_type == 'byol':
                    p0 = model(x_i)
                    z0 = model.forward_momentum(x_i)
                    p1 = model(x_j)
                    z1 = model.forward_momentum(x_j)
                    loss = criterion(p0, z1, p1, z0)
                
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


# ============================================================================
# Model Factory
# ============================================================================

def create_model(model_type, args):
    """Create model based on type"""
    if model_type == 'simclr':
        return SimCLRModel(base_encoder=args.encoder, projection_dim=args.projection_dim)
    elif model_type == 'moco':
        return MoCoModel(base_encoder=args.encoder, projection_dim=args.projection_dim,
                        queue_size=args.queue_size, momentum=args.moco_momentum,
                        temperature=args.temperature)
    elif model_type == 'simsiam':
        return SimSiamModel(base_encoder=args.encoder, projection_dim=args.simsiam_proj_dim,
                           pred_dim=args.simsiam_pred_dim)
    elif model_type == 'mae':
        return MAEModel(img_size=args.image_size, patch_size=args.mae_patch_size,
                       mask_ratio=args.mae_mask_ratio, decoder_embed_dim=args.mae_decoder_embed_dim,
                       decoder_depth=args.mae_decoder_depth, decoder_num_heads=args.mae_decoder_num_heads)
    elif model_type == 'nnclr':
        return NNCLRModel(base_encoder=args.encoder, projection_dim=args.projection_dim,
                         queue_size=args.queue_size, temperature=args.temperature)
    elif model_type == 'simmim':
        return SimMIMModel(img_size=args.image_size, patch_size=args.simmim_patch_size,
                          mask_ratio=args.simmim_mask_ratio)
    elif model_type == 'byol':
        return BYOLModel(base_encoder=args.encoder, projection_dim=args.byol_proj_dim,
                        hidden_dim=args.byol_hidden_dim)
    elif model_type == 'dinov2':
        return DINOv2Model(
            img_size=args.image_size,
            patch_size=args.dinov2_patch_size,
            embed_dim=args.dinov2_embed_dim,
            drop_path_rate=args.dinov2_drop_path_rate,
            ibot_separate_head=args.dinov2_separate_head
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_criterion(model_type, args):
    """Create loss criterion"""
    if model_type == 'simclr':
        return NTXentLoss(temperature=args.temperature)
    elif model_type == 'moco':
        return MoCoLoss(temperature=args.temperature, 
                       memory_bank_size=(args.queue_size, args.projection_dim))
    elif model_type == 'simsiam':
        return SimSiamLoss()
    elif model_type == 'mae':
        return MAELoss(patch_size=args.mae_patch_size)
    elif model_type == 'nnclr':
        return NNCLRLoss(temperature=args.temperature)
    elif model_type == 'simmim':
        return SimMIMLoss()
    elif model_type == 'byol':
        return BYOLLoss()
    elif model_type == 'dinov2':
        return DINOv2Loss()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch_dispatch(model_type, model, train_loader, criterion, optimizer, epoch, total_epochs):
    """Dispatch training based on model type"""
    if model_type == 'simclr':
        return train_epoch_simclr(model, train_loader, criterion, optimizer, epoch)
    elif model_type == 'moco':
        return train_epoch_moco(model, train_loader, criterion, optimizer, epoch, total_epochs)
    elif model_type == 'simsiam':
        return train_epoch_simsiam(model, train_loader, criterion, optimizer, epoch)
    elif model_type == 'mae':
        return train_epoch_mae(model, train_loader, criterion, optimizer, epoch)
    elif model_type == 'nnclr':
        return train_epoch_nnclr(model, train_loader, criterion, optimizer, epoch)
    elif model_type == 'simmim':
        return train_epoch_simmim(model, train_loader, criterion, optimizer, epoch)
    elif model_type == 'byol':
        return train_epoch_byol(model, train_loader, criterion, optimizer, epoch, total_epochs)
    elif model_type == 'dinov2':
        return train_epoch_dinov2(model, train_loader, criterion, optimizer, epoch, total_epochs)


# ============================================================================
# Checkpoint Functions
# ============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, best_val_loss,
                    checkpoint_dir, fold, model_type, is_best=False):
    """Save checkpoint with hierarchical directory structure"""
    fold_checkpoint_dir = os.path.join(checkpoint_dir, model_type, f'fold{fold}')
    os.makedirs(fold_checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'fold': fold,
        'model_type': model_type,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,              # this epoch's validation loss
        'best_val_loss': best_val_loss,    # best validation loss so far
    }
    
    checkpoint_path = os.path.join(fold_checkpoint_dir, f'{model_type}_last_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(fold_checkpoint_dir, f'{model_type}_best_checkpoint.pth')
        torch.save(checkpoint, best_path)
    
    loss_file_path = os.path.join(fold_checkpoint_dir, f'{model_type}_train_val_loss.txt')
    with open(loss_file_path, 'a') as f:
        f.write(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}\n")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load checkpoint; returns (epoch, train_loss, best_val_loss)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Older checkpoints stored the running-best loss in 'val_loss'
    best = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
    return checkpoint['epoch'], checkpoint.get('train_loss', 0.0), best


def train_fold(fold_num, args, all_fold_results):
    """Train a single fold"""
    
    print("\n" + "="*80)
    print(f"TRAINING FOLD {fold_num}/{args.num_folds} - MODEL: {args.model_type.upper()}")
    print("="*80)
    
    fold_dir = os.path.join(args.data_path, f"fold{fold_num}")
    train_dir = os.path.join(fold_dir, "train")
    val_dir = os.path.join(fold_dir, "val")
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise ValueError(f"Directories not found for fold {fold_num}")
    
    # Clear previous loss file if starting fresh
    if not args.resume:
        fold_checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_type, f'fold{fold_num}')
        os.makedirs(fold_checkpoint_dir, exist_ok=True)
        loss_file_path = os.path.join(fold_checkpoint_dir, f'{args.model_type}_train_val_loss.txt')
        if os.path.exists(loss_file_path):
            os.remove(loss_file_path)
    
    transform = get_transform(args.model_type, args.image_size)
    
    # Use LightlyDataset instead of ImageFolder
    train_dataset = LightlyDataset(input_dir=train_dir, transform=transform)
    val_dataset = LightlyDataset(input_dir=val_dir, transform=transform)
    
    print(f"Training images: {len(train_dataset)}, Validation images: {len(val_dataset)}")
    
    # MAE and SimMIM don't need drop_last since they handle varying batch sizes
    drop_last = args.model_type not in ['mae', 'simmim']
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True, drop_last=drop_last)
    
    model = create_model(args.model_type, args).to(device)
    criterion = create_criterion(args.model_type, args)
    
    # Move criterion to device for DINOv2 (contains parameters)
    if args.model_type == 'dinov2':
        criterion = criterion.to(device)
    
    # Use AdamW for MAE and DINOv2, SGD for BYOL, Adam for others
    if args.model_type == 'mae':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    elif args.model_type == 'byol':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
    elif args.model_type == 'dinov2':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.04)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume:
        fold_checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_type, f'fold{fold_num}')
        checkpoint_path = os.path.join(fold_checkpoint_dir, f'{args.model_type}_last_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            try:
                last_epoch, _, last_best = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
                start_epoch = last_epoch + 1
                best_val_loss = last_best
                print(f"Resumed from epoch {start_epoch}")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")
    
    if start_epoch > args.epochs:
        print(f"Fold {fold_num} already completed")
        return best_val_loss
    
    fold_train_losses = []
    fold_val_losses = []
    
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch_dispatch(args.model_type, model, train_loader, criterion, 
                                         optimizer, epoch, args.epochs)
        val_loss = validate_generic(args.model_type, model, val_loader, criterion, 
                                    args.epochs, epoch)
        scheduler.step()
        
        fold_train_losses.append(train_loss)
        fold_val_losses.append(val_loss)
        
        print(f"\nFold {fold_num} - Epoch {epoch}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"🎉 New best validation loss: {best_val_loss:.4f}")
        
        save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, best_val_loss,
                       args.checkpoint_dir, fold_num, args.model_type, is_best)
    
    all_fold_results[f'fold{fold_num}'] = {
        'best_val_loss': best_val_loss,
        'final_train_loss': fold_train_losses[-1],
        'final_val_loss': fold_val_losses[-1]
    }
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Self-supervised pre-training with Lightly')
    
    # Model selection
    parser.add_argument('--model_type', type=str, default='simclr',
                       choices=['simclr', 'moco', 'simsiam', 'mae', 'nnclr', 'simmim', 'byol', 'dinov2'],
                       help='SSL method to use')
    
    # Dataset arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help="Folder containing fold1..foldK, each with train/ and val/")
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--folds', type=str, default=None, help='Comma-separated fold numbers or "all"')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # Common arguments
    parser.add_argument('--encoder', type=str, default='resnet50', 
                       choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.5)
    
    # MoCo specific
    parser.add_argument('--queue_size', type=int, default=4096)
    parser.add_argument('--moco_momentum', type=float, default=0.999)
    
    # SimSiam specific
    parser.add_argument('--simsiam_proj_dim', type=int, default=2048)
    parser.add_argument('--simsiam_pred_dim', type=int, default=512)
    
    # MAE specific
    parser.add_argument('--mae_patch_size', type=int, default=32)
    parser.add_argument('--mae_mask_ratio', type=float, default=0.75)
    parser.add_argument('--mae_decoder_embed_dim', type=int, default=512)
    parser.add_argument('--mae_decoder_depth', type=int, default=1)
    parser.add_argument('--mae_decoder_num_heads', type=int, default=16)
    
    # SimMIM specific
    parser.add_argument('--simmim_patch_size', type=int, default=32)
    parser.add_argument('--simmim_mask_ratio', type=float, default=0.6)
    
    # BYOL specific
    parser.add_argument('--byol_proj_dim', type=int, default=256)
    parser.add_argument('--byol_hidden_dim', type=int, default=1024)
    
    # DINOv2 specific
    parser.add_argument('--dinov2_patch_size', type=int, default=16)
    parser.add_argument('--dinov2_embed_dim', type=int, default=384)
    parser.add_argument('--dinov2_drop_path_rate', type=float, default=0.1)
    parser.add_argument('--dinov2_separate_head', action='store_true', 
                       help='Use separate heads for DINO and iBOT')
    
    # Checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/selfsupervised')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SELF-SUPERVISED PRE-TRAINING (LIGHTLY)")
    print("="*80)
    print("✅ Using Lightly optimized implementations for all 8 models")
    print("   Models: SimCLR, MoCo, SimSiam, MAE, NNCLR, SimMIM, BYOL, DINOv2")
    print("="*80)
    
    if not os.path.exists(args.data_path):
        raise ValueError(f"Dataset path does not exist: {args.data_path}")
    
    # Parse folds to train
    if args.folds is None or args.folds.lower() == 'all':
        folds_to_train = list(range(1, args.num_folds + 1))
    else:
        folds_to_train = [int(f) for f in args.folds.split(',')]
    
    print(f"\nDataset: {args.data_path}")
    print(f"Folds to train: {folds_to_train}")
    print(f"Model: {args.model_type}")
    print(f"Encoder: {args.encoder}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print("="*80 + "\n")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    all_fold_results = {}
    best_val_losses = []
    
    for fold_num in folds_to_train:
        try:
            best_val_loss = train_fold(fold_num, args, all_fold_results)
            best_val_losses.append(best_val_loss)
        except Exception as e:
            print(f"\n❌ Error training fold {fold_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    if best_val_losses:
        for i, fold_num in enumerate(folds_to_train[:len(best_val_losses)]):
            print(f"Fold {fold_num}: Best Val Loss = {best_val_losses[i]:.4f}")
        
        print(f"\nAverage Best Val Loss: {np.mean(best_val_losses):.4f} ± {np.std(best_val_losses):.4f}")
        print(f"Best Fold: {folds_to_train[np.argmin(best_val_losses)]} (Loss: {min(best_val_losses):.4f})")
    
    print("="*80)
    
    # Save results
    results_dir = os.path.join(args.checkpoint_dir, args.model_type)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f'{args.model_type}_training_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_fold_results, f, indent=4)
    print(f"\nResults saved to: {results_path}\n")


if __name__ == '__main__':
    main()