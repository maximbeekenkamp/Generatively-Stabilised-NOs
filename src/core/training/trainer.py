import logging
import time, math

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from src.core.models.model import PredictionModel
from src.core.training.loss import PredictionLoss
from src.core.training.loss_history import LossHistory
from src.core.utils.params import DataParams, TrainingParams


class Trainer(object):
    model: PredictionModel
    trainLoader: DataLoader
    optimizer: Optimizer
    lrScheduler: _LRScheduler
    criterion: PredictionLoss
    trainHistory: LossHistory
    writer: SummaryWriter
    p_t: TrainingParams

    def __init__(self, model:PredictionModel, trainLoader:DataLoader, optimizer:Optimizer, lrScheduler:_LRScheduler,
            criterion:PredictionLoss, trainHistory:LossHistory, writer:SummaryWriter, p_d:DataParams, p_t:TrainingParams,
            checkpoint_path:str=None, checkpoint_frequency:int=None):
        self.model = model
        self.trainLoader = trainLoader
        self.optimizer = optimizer
        self.lrScheduler = lrScheduler
        self.criterion = criterion
        self.trainHistory = trainHistory
        self.writer = writer
        self.p_d = p_d
        self.p_t = p_t
        self.checkpoint_path = checkpoint_path
        self.checkpoint_frequency = checkpoint_frequency if checkpoint_frequency is not None else 50

        self.seqenceLength = self.p_d.sequenceLength[0]

        if self.p_t.fadeInSeqLen[0] > 0 and not self.p_t.fadeInSeqLenLin:
            self.currentSeqLen = 2
            numSeqInc = math.ceil(math.log2(self.seqenceLength / 2.0))
            seqIncStep = math.floor((self.p_t.fadeInSeqLen[1] - self.p_t.fadeInSeqLen[0]) / (numSeqInc-1))
            self.seqIncreaseSteps = list(range(self.p_t.fadeInSeqLen[0], self.p_t.fadeInSeqLen[1]+1, seqIncStep))

            assert (numSeqInc == len(self.seqIncreaseSteps)), "Sequence length computation problem"
            print("Sequence length schedule: %d increases by factor 2 at epochs %s" % (numSeqInc, str(self.seqIncreaseSteps)))
            print("")
            logging.info("Sequence length schedule: %d increases by factor 2 at epochs %s" % (numSeqInc, str(self.seqIncreaseSteps)))
            logging.info("")

        elif self.p_t.fadeInSeqLen[0] > 0 and self.p_t.fadeInSeqLenLin:
            self.currentSeqLen = 2
            numSeqInc = max(0, self.seqenceLength - 2)
            seqIncStep = math.floor((self.p_t.fadeInSeqLen[1] - self.p_t.fadeInSeqLen[0]) / (numSeqInc-1))
            self.seqIncreaseSteps = list(range(self.p_t.fadeInSeqLen[0], self.p_t.fadeInSeqLen[1]+1, seqIncStep))

            seqIncStr = str(self.seqIncreaseSteps) if len(self.seqIncreaseSteps) < 10 else str(self.seqIncreaseSteps[:4]) + " ... " + str(self.seqIncreaseSteps[-4:])
            print("Sequence length schedule: %d increases by value 1 at epochs %s" % (numSeqInc, seqIncStr))
            print("")
            logging.info("Sequence length schedule: %d increases by value 1 at epochs %s" % (numSeqInc, seqIncStr))
            logging.info("")

        else:
            self.currentSeqLen = self.seqenceLength
            self.seqIncreaseSteps = []


    # run one epoch of training
    def trainingStep(self, epoch:int):
        assert (len(self.trainLoader) > 0), "Not enough samples for one batch!"
        timerStart = time.perf_counter()
        timerEnd = 0

        if self.currentSeqLen < self.seqenceLength and epoch in self.seqIncreaseSteps:
            if not self.p_t.fadeInSeqLenLin:
                self.currentSeqLen = min(2 * self.currentSeqLen, self.seqenceLength)
            else:
                self.currentSeqLen = min(1 + self.currentSeqLen, self.seqenceLength)

        # TNO Phase 1.1: Update epoch for automatic phase transitions
        if hasattr(self.model.modelDecoder, 'update_epoch'):
            self.model.modelDecoder.update_epoch(epoch)

        self.model.train()
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

            fadePredStart = self.p_t.fadeInPredLoss[0]
            fadePredEnd = self.p_t.fadeInPredLoss[1]
            fade = (epoch - fadePredStart) / (fadePredEnd - fadePredStart)
            fade = max(min(fade, 1), 0)

            # train prediction samples with AE until fading starts and latent network becomes active
            fadeWeight = fade if fade > 0 else 1
            useLatent = fade > 0

            # Get model output - handle both standard and diffusion model returns
            model_output = self.model(data, simParameters, useLatent=useLatent)

            # Diffusion models (ACDM, Refiner, etc.) return (noise, predictedNoise) during training
            # Standard models return (prediction, latentSpace, vaeMeanVar)
            if len(model_output) == 2:
                # Diffusion model: keep as tuple for loss calculation
                prediction = model_output
                latentSpace = None
                vaeMeanVar = (None, None)
            else:
                # Standard model: unpack 3-tuple
                prediction, latentSpace, vaeMeanVar = model_output

            if self.currentSeqLen < self.seqenceLength:
                if vaeMeanVar[0] is not None and vaeMeanVar[1] is not None:
                    vaeMeanVar = (vaeMeanVar[0][:,0:self.currentSeqLen], vaeMeanVar[1][:,0:self.currentSeqLen])

                # Handle tuple predictions (diffusion models)
                if isinstance(prediction, tuple):
                    p = tuple(pred[:,0:self.currentSeqLen] if pred is not None else None for pred in prediction)
                else:
                    p = prediction[:,0:self.currentSeqLen]

                d = data[:,0:self.currentSeqLen]
                l = latentSpace[:,0:self.currentSeqLen] if latentSpace is not None else None
            else:
                p = prediction
                d = data
                l = latentSpace

            #if obsMask is not None:
            #    p = p * obsMask
            #    d = d * obsMask

            ignorePredLSIMSteps = 0
            # ignore loss on scalar simulation parameters that are replaced in unet rollout during training
            if self.model.p_md.arch in ["unet", "unet+Prev", "unet+2Prev", "unet+3Prev",
                                    "dil_resnet", "dil_resnet+Prev", "dil_resnet+2Prev", "dil_resnet+3Prev",
                                    "resnet", "resnet+Prev", "resnet+2Prev", "resnet+3Prev",
                                    "fno", "fno+Prev", "fno+2Prev", "fno+3Prev",
                                    "dfp", "dfp+Prev", "dfp+2Prev", "dfp+3Prev",]:
                numFields = self.p_d.dimension + len(self.p_d.simFields)
                p = p[:,:,0:numFields]
                d = d[:,:,0:numFields]
                if "+Prev" in self.model.p_md.arch:
                    ignorePredLSIMSteps = 1
                elif "+2Prev" in self.model.p_md.arch:
                    ignorePredLSIMSteps = 2
                elif "+3Prev" in self.model.p_md.arch:
                    ignorePredLSIMSteps = 3
            elif self.model.p_md.arch in ["tno", "tno+Prev", "tno+2Prev", "tno+3Prev"]:
                # TNO handles field extraction differently
                # TNO predicts physical fields, not simulation parameters
                numFields = self.p_d.dimension + len(self.p_d.simFields)
                
                # Extract only physical fields for loss calculation
                p = p[:, :, 0:numFields]  # Predictions: remove sim params
                d = d[:, :, 0:numFields]  # Ground truth: remove sim params
                
                # For Phase 0 (L=1, K=1): No previous steps to ignore
                # For Phase 1 (L>1): Still compare all predictions
                ignorePredLSIMSteps = 0
                
                # Log TNO-specific information periodically
                if s % 100 == 0:
                    if hasattr(self.model.modelDecoder, 'get_info'):
                        info = self.model.modelDecoder.get_info()
                        logging.info(f"TNO Status - Phase: {info['training_phase']}, "
                                   f"L: {info['L']}, K: {info['K']}, "
                                   f"Epoch: {info['current_epoch']}")
                
                # Coordinate TNO's L with curriculum learning if enabled
                if hasattr(self.model, 'modelDecoder') and hasattr(self.model.modelDecoder, 'L'):
                    if self.p_t.fadeInSeqLen[0] > 0 and hasattr(self, 'currentSeqLen'):
                        # Adapt TNO's history length based on curriculum
                        effective_L = min(self.model.modelDecoder.L, self.currentSeqLen - 1)
                        if effective_L != self.model.modelDecoder.L and s == 0:  # Log once per epoch
                            logging.info(f"Adapting TNO L from {self.model.modelDecoder.L} "
                                       f"to {effective_L} for curriculum learning")

            loss, lossParts, lossSeq = self.criterion(p, d, l, vaeMeanVar, fadePredWeight=fadeWeight, ignorePredLSIMSteps=ignorePredLSIMSteps)

            loss.backward()

            self.optimizer.step()

            timerEnd = time.perf_counter()
            self.trainHistory.updateBatch(lossParts, lossSeq, s, (timerEnd-timerStart)/60.0)

        self.lrScheduler.step()

        timerEnd = time.perf_counter()
        self.trainHistory.updateEpoch((timerEnd-timerStart)/60.0)

        if epoch % 50 == 49:
            self.model.eval()
            with torch.no_grad():
                if obsMask is not None:
                    maskedPred = prediction * obsMask
                    maskedData = data * obsMask
                else:
                    maskedPred = prediction
                    maskedData = data

                self.trainHistory.writePredictionExample(maskedPred, maskedData)
                self.trainHistory.writeSequenceLoss(lossSeq)

        self.trainHistory.prepareAndClearForNextEpoch()

        # Periodic checkpoint saving
        if self.checkpoint_path and self.checkpoint_frequency and (epoch + 1) % self.checkpoint_frequency == 0:
            import os
            checkpoint_dir = os.path.dirname(self.checkpoint_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lrScheduler.state_dict(),
            }
            torch.save(checkpoint_data, self.checkpoint_path)
            logging.info(f"Checkpoint saved at epoch {epoch+1}: {self.checkpoint_path}")




class Tester(object):
    model: PredictionModel
    testLoader: DataLoader
    criterion: PredictionLoss
    testHistory: LossHistory
    p_t: TrainingParams

    def __init__(self, model:PredictionModel, testLoader:DataLoader, criterion:PredictionLoss,
                    testHistory:LossHistory, p_t:TrainingParams):
        self.model = model
        self.testLoader = testLoader
        self.criterion = criterion
        self.testHistory = testHistory
        self.p_t = p_t


    # run one epoch of testing
    def testStep(self, epoch:int):
        if epoch % self.testHistory.epochStep != self.testHistory.epochStep - 1:
            return

        assert (len(self.testLoader) > 0), "Not enough samples for one batch!"
        timerStart = time.perf_counter()
        timerEnd = 0

        self.model.eval()
        with torch.no_grad():
            for s, sample in enumerate(self.testLoader, 0):
                device = "cuda" if self.model.useGPU else "cpu"
                data = sample["data"].to(device)
                simParameters = sample["simParameters"].to(device) if type(sample["simParameters"]) is not dict else None
                if "obsMask" in sample is not None:
                    obsMask = sample["obsMask"].to(device)
                    obsMask = torch.unsqueeze(torch.unsqueeze(obsMask, 1), 2)
                else:
                    obsMask = None

                prediction, latentSpace, vaeMeanVar = self.model(data, simParameters, useLatent=True)

                p = prediction
                d = data
                l = latentSpace

                if obsMask is not None:
                    p = p * obsMask
                    d = d * obsMask

                _, lossParts, lossSeq = self.criterion(p, d, l, vaeMeanVar, weighted=False)

                timerEnd = time.perf_counter()
                self.testHistory.updateBatch(lossParts, lossSeq, s, (timerEnd-timerStart)/60.0)

            timerEnd = time.perf_counter()
            self.testHistory.updateEpoch((timerEnd-timerStart)/60.0)

            #if epoch % 50 == 49:
            if obsMask is not None:
                maskedPred = prediction * obsMask
                maskedData = data * obsMask
            else:
                maskedPred = prediction
                maskedData = data

            self.testHistory.writePredictionExample(maskedPred, maskedData)
            self.testHistory.writeSequenceLoss(lossSeq)

            self.testHistory.prepareAndClearForNextEpoch()


    def generatePredictions(self, output_path:str=None):
        """
        Generate predictions from the model and optionally save to .npz file.

        This method runs the model on the test loader and collects all predictions
        in the autoreg .npz format: [num_sequences, timesteps, channels, H, W]

        Args:
            output_path: Path to save .npz file. If None, just returns predictions.

        Returns:
            predictions: numpy array of shape [num_sequences, timesteps, channels, H, W]
        """
        import numpy as np
        import os

        assert (len(self.testLoader) > 0), "Not enough samples for prediction generation!"

        logging.info(f"Generating predictions with {len(self.testLoader)} batches...")

        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for s, sample in enumerate(self.testLoader, 0):
                device = "cuda" if self.model.useGPU else "cpu"
                data = sample["data"].to(device)
                simParameters = sample["simParameters"].to(device) if type(sample["simParameters"]) is not dict else None

                # Generate prediction
                prediction, _, _ = self.model(data, simParameters, useLatent=True)

                # Move to CPU and convert to numpy
                # prediction shape: [B, T, C, H, W]
                pred_np = prediction.cpu().numpy()
                all_predictions.append(pred_np)

                if (s + 1) % 10 == 0:
                    logging.info(f"  Processed {s+1}/{len(self.testLoader)} batches")

        # Concatenate all batches along batch dimension
        # Result: [total_sequences, timesteps, channels, H, W]
        predictions = np.concatenate(all_predictions, axis=0)

        logging.info(f"Generated predictions with shape: {predictions.shape}")

        # Save to .npz if path provided
        if output_path is not None:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Save in autoreg format
            np.savez_compressed(output_path, arr_0=predictions)
            logging.info(f"Saved predictions to: {output_path}")

        return predictions

