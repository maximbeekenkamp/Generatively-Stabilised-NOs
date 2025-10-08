import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np
import operator
from functools import reduce

from src.core.utils.lsim.distance_model import DistanceModel as LSIM_Model
from src.core.utils.params import LossParams

# Spectral metrics for turbulence validation and training
from .spectral_metrics import (
    compute_field_error_loss,
    compute_spectrum_error_loss,
    compute_field_error_validation,
    compute_spectrum_error_validation
)


# input shape: B S C W H -> output shape: B S C
def loss_lsim(lsimModel:nn.Module, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    # normalize to [0,255] for each sequence
    xMin = torch.amin(x, dim=(1,3,4), keepdim=True)
    xMax = torch.amax(x, dim=(1,3,4), keepdim=True)
    yMin = torch.amin(y, dim=(1,3,4), keepdim=True)
    yMax = torch.amax(y, dim=(1,3,4), keepdim=True)

    # Add epsilon to prevent division by zero when output is constant
    eps = 1e-8
    ref = 255 * ((x - xMin) / (xMax - xMin + eps))
    oth = 255 * ((y - yMin) / (yMax - yMin + eps))
    ##dMin = torch.minimum(xMin, yMin)
    ##dMax = torch.maximum(xMax, yMax)
    ##ref = 255 * ((x - dMin) / (dMax - dMin))
    ##oth = 255 * ((y - dMin) / (dMax - dMin))

    # compare each sequence and channel individually by moving them to batch dimension
    # and adding a dummy channel dimension and lsim parameter dimension
    sizeBatch, sizeSeq, sizeChannel = ref.shape[0], ref.shape[1], ref.shape[2]
    ref = torch.reshape(ref, (-1,1,1,ref.shape[3], ref.shape[4]))
    oth = torch.reshape(oth, (-1,1,1,oth.shape[3], oth.shape[4]))

    # clone channel dimension as lsim compares 3-channel data
    ref = ref.expand(-1,-1,3,-1,-1)
    oth = oth.expand(-1,-1,3,-1,-1)

    inDict = {"reference": ref, "other": oth}
    distance = lsimModel(inDict)

    # move results from each original channel and sequence back into a
    # new channel and sequence dimension
    distance = torch.reshape(distance, (sizeBatch, sizeSeq, sizeChannel))

    return distance





class PredictionLoss(nn.modules.loss._Loss):

    def __init__(self, p_l:LossParams, dimension:int, simFields:List[str], useGPU:bool):
        super(PredictionLoss, self).__init__()
        self.p_l = p_l
        self.dimension = dimension
        self.simFields = simFields
        self.useGPU = useGPU

        if self.dimension == 2:
            # load lsim (frozen network, gradients flow through)
            self.lsim = LSIM_Model(baseType="lsim", isTrain=False, useGPU=self.useGPU)
            self.lsim.load("src/core/utils/lsim/models/LSiM.pth")
            self.lsim.eval()
            # freeze lsim weights (but gradients still flow to prediction)
            for param in self.lsim.parameters():
                param.requires_grad = False

        # Component weights for flexible loss scheduling
        # Default: 100% field error (replaces old MSE-based defaults)
        self.component_weights = {
            'recFieldError': getattr(p_l, 'recFieldError', 1.0),
            'predFieldError': getattr(p_l, 'predFieldError', 1.0),
            'recLSIM': getattr(p_l, 'recLSIM', 0.0),
            'predLSIM': getattr(p_l, 'predLSIM', 0.0),
            'spectrumError': getattr(p_l, 'spectrumError', 0.0),
        }

    def set_loss_weights(self, weights: dict):
        """
        Set component weights for flexible loss scheduling.

        Called by FlexibleLossScheduler to dynamically adjust loss composition.

        Args:
            weights: Dict mapping component names to weights
                     e.g., {'recFieldError': 0.7, 'spectrumError': 0.3}
        """
        self.component_weights.update(weights)


    def forward(self, prediction:torch.Tensor, groundTruth:torch.Tensor, latentSpace:torch.Tensor, vaeMeanVar:Tuple[torch.Tensor, torch.Tensor],
            weighted:bool=True, fadePredWeight:float=1, noLSIM:bool=False, ignorePredLSIMSteps:int=0) -> Tuple[torch.Tensor, dict, dict]:
        device = "cuda" if self.useGPU else "cpu"

        # Handle diffusion model training: prediction is (noise, predictedNoise) tuple
        if isinstance(prediction, tuple) and len(prediction) == 2:
            noise, predictedNoise = prediction
            # Diffusion loss: MSE between actual noise and predicted noise
            loss = F.mse_loss(predictedNoise, noise)

            # Return complete lossParts dict matching new component format
            zero_loss = torch.zeros(1).to(device)
            lossParts = {
                'lossFull': loss,
                'lossRecFieldError': zero_loss,
                'lossPredFieldError': loss,  # Treat diffusion as prediction task
                'lossRecLSIM': zero_loss,
                'lossPredLSIM': zero_loss,
                'lossSpectrumError': zero_loss,
            }
            lossSeq = {'fieldError': zero_loss, 'LSIM': zero_loss, 'spectrumError': zero_loss}
            return loss, lossParts, lossSeq

        # Check sequence length for prediction losses
        pred_weight = self.component_weights.get('predFieldError', 0) + self.component_weights.get('predLSIM', 0)
        assert(not (pred_weight > 0 and prediction.shape[1] <= 1)), "Sequence length too small for prediction errors!"

        # Unweighted mode: Compute all metrics for validation logging
        if not weighted:
            numFields = self.dimension + len(self.simFields)
            zero_loss = torch.zeros(1).to(device)

            # Compute field error (replaces MSE as primary metric)
            # Use per-frame relative error for better turbulence stability
            rec_field_err_val = compute_field_error_validation(
                prediction[:, 0:1], groundTruth[:, 0:1]
            )
            lossRecFieldError = torch.tensor([rec_field_err_val]).to(device)

            if prediction.shape[1] > 1:
                pred_field_err_val = compute_field_error_validation(
                    prediction[:, 1:], groundTruth[:, 1:]
                )
                lossPredFieldError = torch.tensor([pred_field_err_val]).to(device)
            else:
                lossPredFieldError = zero_loss

            # Compute spectrum error for spectral bias validation
            spec_err_val = compute_spectrum_error_validation(
                prediction, groundTruth
            )
            lossSpectrumError = torch.tensor([spec_err_val]).to(device)

            # Compute LSIM (only on fields, only for 2D)
            if not noLSIM and self.dimension == 2:
                seqLSIM = loss_lsim(self.lsim, prediction[:,:,0:numFields],
                                    groundTruth[:,:,0:numFields]).mean((0,2))
                lossRecLSIM = seqLSIM[0:1]  # Keep as tensor
                lossPredLSIM = torch.mean(seqLSIM[1:]).unsqueeze(0) if len(seqLSIM) > 1 else zero_loss
            else:
                seqLSIM = torch.zeros(prediction.shape[1]).to(device)
                lossRecLSIM = zero_loss
                lossPredLSIM = zero_loss

            # Regularization terms (not computed in unweighted mode)
            lossRegMeanStd = zero_loss
            lossRegDiv = zero_loss
            lossRegVaeKLDiv = zero_loss
            lossRegLatStep = zero_loss

            # Legacy MSE for backward compatibility (still useful for monitoring)
            if self.dimension == 2:
                seqMSE = F.mse_loss(prediction, groundTruth, reduction="none").mean((0,2,3,4))
            elif self.dimension == 3:
                seqMSE = F.mse_loss(prediction, groundTruth, reduction="none").mean((0,2,3,4,5))
            else:
                seqMSE = zero_loss

            # Sequence-level metrics for detailed logging
            seqFieldError = torch.tensor([rec_field_err_val, pred_field_err_val if prediction.shape[1] > 1 else 0.0]).to(device)
            seqSpectrumError = torch.tensor([spec_err_val, spec_err_val]).to(device)

        # Weighted mode: Compute losses with component weights for training
        else:
            numFields = self.dimension + len(self.simFields)
            zero_loss = torch.zeros(1).to(device)

            # Initialize all loss components
            lossRecFieldError = zero_loss
            lossPredFieldError = zero_loss
            lossRecLSIM = zero_loss
            lossPredLSIM = zero_loss
            lossSpectrumError = zero_loss
            lossRegMeanStd = zero_loss
            lossRegDiv = zero_loss
            lossRegVaeKLDiv = zero_loss
            lossRegLatStep = zero_loss

            # ========================================================================
            # FIELD ERROR: Primary training loss (replaces MSE)
            # ========================================================================
            if self.component_weights.get('recFieldError', 0) > 0:
                rec_field_err = compute_field_error_loss(
                    prediction[:, 0:1], groundTruth[:, 0:1]
                )
                lossRecFieldError = self.component_weights['recFieldError'] * rec_field_err

            if self.component_weights.get('predFieldError', 0) > 0 and fadePredWeight > 0:
                pred_field_err = compute_field_error_loss(
                    prediction[:, 1:], groundTruth[:, 1:]
                )
                lossPredFieldError = (self.component_weights['predFieldError'] *
                                     fadePredWeight * pred_field_err)

            # ========================================================================
            # SPECTRUM ERROR: For spectral bias mitigation
            # ========================================================================
            if self.component_weights.get('spectrumError', 0) > 0:
                spec_err = compute_spectrum_error_loss(prediction, groundTruth)
                lossSpectrumError = self.component_weights['spectrumError'] * spec_err

            # ========================================================================
            # LSIM: Perceptual loss (2D only, frozen network)
            # ========================================================================
            seqLSIM = None
            if self.dimension == 2:
                # Compute LSIM based on which components are active
                rec_lsim_active = self.component_weights.get('recLSIM', 0) > 0
                pred_lsim_active = self.component_weights.get('predLSIM', 0) > 0

                if pred_lsim_active:
                    if rec_lsim_active:
                        # Compute for all timesteps
                        seqLSIM = loss_lsim(self.lsim, prediction[:,:,0:numFields],
                                           groundTruth[:,:,0:numFields]).mean((0,2))
                    else:
                        # Skip early timesteps if ignorePredLSIMSteps > 0
                        predLSIMStart = 1 + ignorePredLSIMSteps
                        seqLSIM = loss_lsim(self.lsim, prediction[:,predLSIMStart:,0:numFields],
                                           groundTruth[:,predLSIMStart:,0:numFields]).mean((0,2))
                        seqLSIM = torch.concat([torch.zeros(predLSIMStart, device=device), seqLSIM])

                elif rec_lsim_active:
                    # Only reconstruction timestep
                    seqLSIM = loss_lsim(self.lsim, prediction[:,0:1,0:numFields],
                                       groundTruth[:,0:1,0:numFields]).mean((0,2))

                # Apply weights
                if rec_lsim_active and seqLSIM is not None:
                    lossRecLSIM = self.component_weights['recLSIM'] * seqLSIM[0]

                if pred_lsim_active and fadePredWeight > 0 and seqLSIM is not None:
                    lossPredLSIM = (self.component_weights['predLSIM'] *
                                   fadePredWeight * torch.mean(seqLSIM[1:]))


            # mean and std regularization
            if self.p_l.regMeanStd > 0:
                if self.dimension == 2:
                    meanDiff = torch.abs(groundTruth.mean((3,4)) - prediction.mean((3,4)))
                    stdDiff  = torch.abs(groundTruth.std((3,4)) - prediction.std((3,4)))
                elif self.dimension == 3:
                    meanDiff = torch.abs(groundTruth.mean((3,4,5)) - prediction.mean((3,4,5)))
                    stdDiff  = torch.abs(groundTruth.std((3,4,5)) - prediction.std((3,4,5)))
                meanStd = meanDiff.mean() + stdDiff.mean()
                lossRegMeanStd = self.p_l.regMeanStd * meanStd

            # diveregence regularization
            if self.p_l.regMeanStd > 0:
                if self.dimension == 2:
                    vx_dx, _ = torch.gradient(prediction[:,:,0:1], dim=(3,4))
                    _, vy_dy = torch.gradient(prediction[:,:,1:2], dim=(3,4))
                    div = vx_dx + vy_dy
                elif self.dimension == 3:
                    vx_dx, _, _ = torch.gradient(prediction[:,:,0:1], dim=(3,4,5))
                    _, vy_dy, _ = torch.gradient(prediction[:,:,1:2], dim=(3,4,5))
                    _, _, vz_dz = torch.gradient(prediction[:,:,2:3], dim=(3,4,5))
                    div = vx_dx + vy_dy + vz_dz
                lossRegDiv = self.p_l.regDiv * torch.abs(div).mean()

            # KL divergence regularization for VAE
            if self.p_l.regVae > 0 and not vaeMeanVar[0] is None and not vaeMeanVar[1] is None:
                vaeMean = vaeMeanVar[0]
                vaeLogVar = vaeMeanVar[1]
                lossRegVaeKLDiv = -0.5 * torch.mean(1 + vaeLogVar - vaeMean.pow(2) - vaeLogVar.exp())
                lossRegVaeKLDiv = self.p_l.regVae * lossRegVaeKLDiv

            # latent space step regularization
            if self.p_l.regLatStep > 0 and latentSpace.shape[1] > 1:
                latFirst = latentSpace[:,0:latentSpace.shape[1]-1]
                latSecond = latentSpace[:,1:latentSpace.shape[1]]
                lossRegLatStep = self.p_l.regLatStep * torch.mean(torch.abs(latFirst - latSecond))

            # Legacy MSE for backward compatibility (not weighted, just for monitoring)
            if self.dimension == 2:
                seqMSE = F.mse_loss(prediction, groundTruth, reduction="none").mean((0,2,3,4))
            elif self.dimension == 3:
                seqMSE = F.mse_loss(prediction, groundTruth, reduction="none").mean((0,2,3,4,5))
            else:
                seqMSE = zero_loss

            # Compute unweighted field/spectrum errors for sequence logging
            seqFieldError = zero_loss
            seqSpectrumError = zero_loss

        # ========================================================================
        # TOTAL LOSS: Sum all weighted components
        # ========================================================================
        loss = (lossRecFieldError + lossPredFieldError +
                lossRecLSIM + lossPredLSIM +
                lossSpectrumError +
                lossRegMeanStd + lossRegDiv + lossRegVaeKLDiv + lossRegLatStep)

        lossParts = {
            "lossFull": loss,
            "lossRecFieldError": lossRecFieldError,
            "lossPredFieldError": lossPredFieldError,
            "lossRecLSIM": lossRecLSIM,
            "lossPredLSIM": lossPredLSIM,
            "lossSpectrumError": lossSpectrumError,
            # Regularization terms (for monitoring)
            "lossRegMeanStd": lossRegMeanStd,
            "lossRegDiv": lossRegDiv,
            "lossRegVaeKLDiv": lossRegVaeKLDiv,
            "lossRegLatStep": lossRegLatStep,
        }

        lossSeq = {
            "fieldError": seqFieldError,
            "LSIM": seqLSIM if seqLSIM is not None else zero_loss,
            "spectrumError": seqSpectrumError,
        }

        return loss, lossParts, lossSeq


class LpLoss(object):
    """
    Relative Lp loss for TNO training
    ||u - v||_p / ||v||_p
    
    From TNO implementation for turbulence forecasting
    """
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are positive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        """
        Compute relative error ||x - y||_p / ||y||_p
        
        Args:
            x: prediction tensor
            y: target tensor
            
        Returns:
            Relative Lp error
        """
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
