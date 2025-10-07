"""
Enhanced TNO Trainer Class with Advanced Phase Management
Phase 1.3: Two-phase training support (teacher forcing → fine-tuning)

Features:
- Gradual phase transitions
- Phase-aware loss weighting
- Advanced TNO monitoring
- Dynamic learning rate adjustment
"""

import logging
import time
import math
import torch
import numpy as np
from typing import Dict, Optional
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast  # GradScaler now inherited from base Trainer

from .trainer import Trainer
from src.core.models.model import PredictionModel
from src.core.training.loss import PredictionLoss
from src.core.training.loss_history import LossHistory
from src.core.utils.params import DataParams, TrainingParams


class TNOTrainer(Trainer):
    """
    Enhanced trainer for TNO models with two-phase training support.
    
    Manages automatic transition from teacher forcing to fine-tuning phase
    with gradual transitions and phase-aware optimization.
    """
    
    def __init__(
        self,
        model: PredictionModel,
        trainLoader: DataLoader,
        optimizer: Optimizer,
        lrScheduler: _LRScheduler,
        criterion: PredictionLoss,
        trainHistory: LossHistory,
        writer: SummaryWriter,
        p_d: DataParams,
        p_t: TrainingParams,
        teacher_forcing_epochs: int = 500,
        transition_epochs: int = 50,  # Gradual transition period
        phase_lr_factor: float = 0.1  # LR reduction for fine-tuning
    ):
        """
        Initialize enhanced TNO trainer with phase management.
        
        Args:
            teacher_forcing_epochs: Number of epochs for teacher forcing phase
            transition_epochs: Number of epochs for gradual transition
            phase_lr_factor: Learning rate factor for fine-tuning phase
        """
        super().__init__(
            model, trainLoader, optimizer, lrScheduler, 
            criterion, trainHistory, writer, p_d, p_t
        )
        
        self.teacher_forcing_epochs = teacher_forcing_epochs
        self.transition_epochs = transition_epochs
        self.phase_lr_factor = phase_lr_factor
        
        # Phase management state
        self.phase_switched = False
        self.current_phase = "teacher_forcing"
        self.transition_start_epoch = None
        self.base_lr = None
        
        # Enhanced monitoring
        self.tno_metrics = {
            'phase_transitions': [],
            'loss_by_phase': {'teacher_forcing': [], 'transition': [], 'fine_tuning': []},
            'lr_history': []
        }
        
        # Store initial learning rate
        if hasattr(self.optimizer, 'param_groups'):
            self.base_lr = self.optimizer.param_groups[0]['lr']

        # NOTE: Mixed precision is now inherited from base Trainer class

        print(f"[TNOTrainer] Initialized with:")
        print(f"  - Teacher forcing epochs: {teacher_forcing_epochs}")
        print(f"  - Transition epochs: {transition_epochs}")
        print(f"  - Phase LR factor: {phase_lr_factor}")
        
    def get_phase_status(self, epoch: int) -> Dict:
        """Get current training phase information"""
        if epoch < self.teacher_forcing_epochs:
            phase = "teacher_forcing"
            progress = epoch / self.teacher_forcing_epochs
        elif epoch < self.teacher_forcing_epochs + self.transition_epochs:
            phase = "transition"
            transition_epoch = epoch - self.teacher_forcing_epochs
            progress = transition_epoch / self.transition_epochs
        else:
            phase = "fine_tuning"
            progress = 1.0
            
        return {
            'phase': phase,
            'progress': progress,
            'epoch': epoch,
            'teacher_forcing_epochs': self.teacher_forcing_epochs,
            'transition_epochs': self.transition_epochs
        }
    
    def compute_transition_weight(self, epoch: int) -> float:
        """Compute gradual transition weight for loss blending"""
        if epoch < self.teacher_forcing_epochs:
            return 1.0  # Full teacher forcing
        elif epoch < self.teacher_forcing_epochs + self.transition_epochs:
            # Linear transition from 1.0 to 0.0
            transition_progress = (epoch - self.teacher_forcing_epochs) / self.transition_epochs
            return 1.0 - transition_progress
        else:
            return 0.0  # Full fine-tuning
    
    def adjust_learning_rate(self, epoch: int):
        """Adjust learning rate based on training phase"""
        phase_info = self.get_phase_status(epoch)
        
        if phase_info['phase'] == 'fine_tuning' and not self.phase_switched:
            # Reduce learning rate for fine-tuning phase
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * self.phase_lr_factor
            
            self.phase_switched = True
            logging.info(f"[TNOTrainer] Reduced learning rate to {self.base_lr * self.phase_lr_factor:.2e} for fine-tuning")
    
    def update_tno_phase(self, epoch: int):
        """Update TNO model phase with gradual transitions"""
        phase_info = self.get_phase_status(epoch)
        
        # Update TNO model phase
        if hasattr(self.model, 'set_tno_training_phase'):
            if phase_info['phase'] in ['teacher_forcing', 'transition']:
                self.model.set_tno_training_phase('teacher_forcing')
            else:
                self.model.set_tno_training_phase('fine_tuning')
        
        # Update TNO epoch
        if hasattr(self.model, 'update_tno_epoch'):
            self.model.update_tno_epoch(epoch)
        
        # Log phase transitions
        if phase_info['phase'] != self.current_phase:
            transition_info = {
                'epoch': epoch,
                'from_phase': self.current_phase,
                'to_phase': phase_info['phase']
            }
            self.tno_metrics['phase_transitions'].append(transition_info)
            
            logging.info(f"[TNOTrainer] Phase transition at epoch {epoch}: "
                        f"{self.current_phase} → {phase_info['phase']}")
            
            self.current_phase = phase_info['phase']
    
    def trainingStep(self, epoch: int):
        """Enhanced training step with TNO phase management"""
        assert (len(self.trainLoader) > 0), "Not enough samples for one batch!"
        timerStart = time.perf_counter()
        
        # Handle curriculum learning (from parent class)
        if hasattr(self, 'currentSeqLen') and hasattr(self, 'seqIncreaseSteps'):
            if self.currentSeqLen < self.seqenceLength and epoch in self.seqIncreaseSteps:
                if not self.p_t.fadeInSeqLenLin:
                    self.currentSeqLen = min(2 * self.currentSeqLen, self.seqenceLength)
                else:
                    self.currentSeqLen = min(1 + self.currentSeqLen, self.seqenceLength)

        # Enhanced TNO phase management
        self.update_tno_phase(epoch)
        self.adjust_learning_rate(epoch)
        
        # Get phase information for loss weighting
        phase_info = self.get_phase_status(epoch)
        transition_weight = self.compute_transition_weight(epoch)
        
        self.model.train()
        
        epoch_loss = 0.0
        epoch_tno_loss = 0.0
        num_batches = 0
        
        for s, sample in enumerate(self.trainLoader, 0):
            self.optimizer.zero_grad()

            device = "cuda" if self.model.useGPU else "cpu"
            data = sample["data"].to(device)
            simParameters = sample["simParameters"].to(device) if type(sample["simParameters"]) is not dict else None
            
            if "obsMask" in sample:
                obsMask = sample["obsMask"].to(device)
                obsMask = torch.unsqueeze(torch.unsqueeze(obsMask, 1), 2)
            else:
                obsMask = None

            # Phase-aware loss computation setup
            fadePredStart = self.p_t.fadeInPredLoss[0]
            fadePredEnd = self.p_t.fadeInPredLoss[1]
            fade = (epoch - fadePredStart) / (fadePredEnd - fadePredStart)
            fade = max(min(fade, 1), 0)
            fadeWeight = fade if fade > 0 else 1

            # MIXED PRECISION: Wrap forward pass in autocast
            with autocast('cuda', enabled=self.use_amp):
                # Enhanced TNO forward pass
                if self.model.p_md.arch in ["tno", "tno+Prev", "tno+2Prev", "tno+3Prev"]:
                    prediction = self.model.forwardTNO(data)
                else:
                    prediction = self.model.forward(data, simParameters, stepsLong=-1)

                # TNO-specific loss handling with phase awareness
                d = data.clone()
                p = prediction.clone()

                # Handle TNO field extraction (from parent implementation)
                if self.model.p_md.arch in ["tno", "tno+Prev", "tno+2Prev", "tno+3Prev"]:
                    numFields = self.p_d.dimension + len(self.p_d.simFields)
                    p = p[:, :, 0:numFields]
                    d = d[:, :, 0:numFields]
                    ignorePredLSIMSteps = 0
                else:
                    # Handle other architectures
                    ignorePredLSIMSteps = 0

                # Compute loss with phase-aware weighting
                lossResult = self.criterion(p, d, torch.empty(0), (None, None),
                                          weighted=True, fadePredWeight=fadeWeight * transition_weight,
                                          noLSIM=False, ignorePredLSIMSteps=ignorePredLSIMSteps)

                loss, lossParts, _ = lossResult

            # Track TNO-specific metrics
            if 'lossTNO' in lossParts:
                epoch_tno_loss += lossParts['lossTNO'].item()

            epoch_loss += loss.item()
            num_batches += 1

            # MIXED PRECISION: Use parent's centralized backward/optimizer step
            self._optimize_step(loss)

            # Enhanced logging for TNO training
            if s % 50 == 0:
                # Get TNO status
                tno_status = None
                if hasattr(self.model, 'get_tno_status'):
                    tno_status = self.model.get_tno_status()
                
                log_msg = (f"[TNOTrainer] Epoch {epoch:3d}, Batch {s:3d}, "
                          f"Phase: {phase_info['phase']}, "
                          f"Loss: {loss.item():.6f}")
                
                if tno_status:
                    log_msg += f", L: {tno_status['L']}, K: {tno_status['K']}"
                
                if 'lossTNO' in lossParts and lossParts['lossTNO'].item() > 0:
                    log_msg += f", TNO Loss: {lossParts['lossTNO'].item():.6f}"
                
                logging.info(log_msg)

        # Update LR scheduler (if not manually managing phases)
        if self.lrScheduler and phase_info['phase'] != 'fine_tuning':
            self.lrScheduler.step()

        # Record metrics
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        avg_tno_loss = epoch_tno_loss / max(num_batches, 1)
        
        self.tno_metrics['loss_by_phase'][phase_info['phase']].append(avg_epoch_loss)
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.tno_metrics['lr_history'].append(current_lr)
        
        # Tensorboard logging
        if self.writer:
            self.writer.add_scalar('TNO/Phase_Progress', phase_info['progress'], epoch)
            self.writer.add_scalar('TNO/Transition_Weight', transition_weight, epoch)
            self.writer.add_scalar('TNO/Learning_Rate', current_lr, epoch)
            if avg_tno_loss > 0:
                self.writer.add_scalar('TNO/LpLoss2', avg_tno_loss, epoch)

        timerEnd = time.perf_counter()
        
        logging.info(f"[TNOTrainer] Epoch {epoch} completed in {timerEnd - timerStart:.2f}s, "
                    f"Phase: {phase_info['phase']}, Avg Loss: {avg_epoch_loss:.6f}")
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive TNO training summary"""
        return {
            'current_phase': self.current_phase,
            'phase_transitions': self.tno_metrics['phase_transitions'],
            'teacher_forcing_epochs': self.teacher_forcing_epochs,
            'transition_epochs': self.transition_epochs,
            'total_phase_transitions': len(self.tno_metrics['phase_transitions']),
            'loss_history': self.tno_metrics['loss_by_phase']
        }