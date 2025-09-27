import torch
import torch
import numpy as np
from typing import List, Any
from losses.loss_utils import apply_reduction
from modules.difflocks.difflocks_utils.general_util import get_window
from modules.difflocks.difflocks_utils.strand_util import compute_dirs, compute_curv


def compute_loss_l2(gt_hair_strands, pred_hair_strands):
    loss_l2 = ((pred_hair_strands - gt_hair_strands) ** 2).mean()

    return loss_l2

def compute_loss_l1(gt_hair_strands, pred_hair_strands):
    loss_l1 = torch.nn.functional.l1_loss(pred_hair_strands, gt_hair_strands).mean()

    return loss_l1

def compute_loss_dir_dot(gt_hair_strands, pred_hair_strands):
    nr_verts_per_strand=256

    pred_points = pred_hair_strands.view(-1, nr_verts_per_strand, 3)
    gt_hair_strands=gt_hair_strands.view(-1, nr_verts_per_strand, 3)

    # get also a loss for the direciton, we need to compute the direction
    pred_deltas = compute_dirs(pred_points)
    pred_deltas = pred_deltas.view(-1, 3)
    pred_deltas = torch.nn.functional.normalize(pred_deltas, dim=-1)

    gt_dir = compute_dirs(gt_hair_strands)
    gt_dir = gt_dir.view(-1, 3)
    gt_dir = torch.nn.functional.normalize(gt_dir, dim=-1)
    # loss_dir = self.cosine_embed_loss(pred_deltas, gt_dir, torch.ones(gt_dir.shape[0]).cuda())

    dot = torch.sum(pred_deltas * gt_dir, dim=-1)

    loss_dir = (1.0 - dot).mean()

    return loss_dir

def compute_loss_dir_l1(gt_hair_strands, pred_hair_strands):
    nr_verts_per_strand=256

    pred_points = pred_hair_strands.view(-1, nr_verts_per_strand, 3)
    gt_hair_strands=gt_hair_strands.view(-1, nr_verts_per_strand, 3)

    # get also a loss for the direciton, we need to compute the direction
    pred_deltas = compute_dirs(pred_points)
    pred_deltas = pred_deltas.view(-1, 3)
    pred_deltas = pred_deltas*nr_verts_per_strand #Just because the deltas are very tiny and I want them in a nicer range for the loss

    gt_dir = compute_dirs(gt_hair_strands)
    gt_dir = gt_dir.view(-1, 3)
    gt_dir = gt_dir*nr_verts_per_strand #Just because the deltas are very tiny and I want them in a nicer range for the loss
    # loss_dir = self.cosine_embed_loss(pred_deltas, gt_dir, torch.ones(gt_dir.shape[0]).cuda())

    loss_l1 = torch.nn.functional.l1_loss(pred_deltas, gt_dir).mean()
    return loss_l1

def compute_loss_curv_l1(gt_hair_strands, pred_hair_strands):
    nr_verts_per_strand=256

    pred_points = pred_hair_strands.view(-1, nr_verts_per_strand, 3)
    gt_hair_strands=gt_hair_strands.view(-1, nr_verts_per_strand, 3)

    # get also a loss for the direciton, we need to compute the direction
    pred_deltas = compute_dirs(pred_points)
    pred_curvs = compute_curv(pred_deltas)
    pred_curvs = pred_curvs.view(-1, 3)
    pred_curvs = pred_curvs*nr_verts_per_strand #Just because the deltas are very tiny and I want them in a nicer range for the loss

    gt_dir = compute_dirs(gt_hair_strands)
    gt_curvs = compute_curv(gt_dir)
    gt_curvs = gt_curvs.view(-1, 3)
    gt_curvs = gt_curvs*nr_verts_per_strand #Just because the deltas are very tiny and I want them in a nicer range for the loss
    # loss_dir = self.cosine_embed_loss(pred_deltas, gt_dir, torch.ones(gt_dir.shape[0]).cuda())

    loss_l1 = torch.nn.functional.l1_loss(pred_curvs, gt_curvs).mean()
    return loss_l1

def compute_loss_kl(mean, logstd):
    #get input data
    kl_loss = 0

    #kl loss

    kl_shape = kl( mean, logstd)
    # free bits from IAF-VAE. so that if the KL drops below a certan value, then we stop reducing the KL
    kl_shape = torch.clamp(kl_shape, min=0.25)

    kl_loss = kl_shape.mean()

    return kl_loss

def kl(mean, logstd):
    kl = (-0.5 - logstd + 0.5 * mean ** 2 + 0.5 * torch.exp(2 * logstd))
    return kl

    

    

