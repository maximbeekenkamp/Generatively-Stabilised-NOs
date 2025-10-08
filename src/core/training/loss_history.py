import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from typing import List, Tuple, Optional

# supress warnings and logging caused by tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from torch.utils.tensorboard import SummaryWriter

# Rich progress bar support
try:
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class LossHistory(object):
    mode: str
    modeLong: str
    writer: SummaryWriter
    dataLoaderLength: int
    epoch: int
    epochStep: int
    printInterval: int
    logInterval: int
    simFields: List[str]
    use_rich_progress: bool
    total_epochs: Optional[int]

    accuracy: dict
    batchLoss: dict

    # Rich progress bar components
    progress: Optional['Progress']
    task_id: Optional[int]


    def __init__(self, mode:str, modeLong:str, writer:SummaryWriter, dataLoaderLength:int,
                    epoch:int, epochStep:int, printInterval:int=0, logInterval:int=1, simFields:List[str]=[],
                    use_rich_progress:bool=False, total_epochs:Optional[int]=None, model_name:Optional[str]=None,
                    loss_params:Optional[object]=None):

        self.mode = mode
        self.modeLong = modeLong
        self.writer = writer
        self.dataLoaderLength = dataLoaderLength
        self.epoch = epoch
        self.epochStep = epochStep
        self.printInterval = printInterval
        self.logInterval = logInterval
        self.simFields = simFields
        self.use_rich_progress = use_rich_progress and RICH_AVAILABLE
        self.total_epochs = total_epochs
        self.model_name = model_name
        self.loss_params = loss_params

        # Initialize Rich progress bar if requested
        self.progress = None
        self.task_id = None
        if self.use_rich_progress:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("{task.fields[loss_text]}"),
                TimeElapsedColumn(),
                transient=False  # Keep progress bar visible after completion
            )
            self.progress.start()

            # If total_epochs > 5, create a single progress bar for all training
            if self.total_epochs is not None and self.total_epochs > 5:
                total_batches = self.total_epochs * self.dataLoaderLength
                description = f"[{self.model_name}] {self.modeLong}" if self.model_name else f"{self.modeLong}"
                self.task_id = self.progress.add_task(
                    description,
                    total=total_batches,
                    loss_text="Loss: N/A"
                )

        self.accuracy = {}

        self.clear()

    def clear(self):
        self.batchLoss = {
            "lossFull": [],
            # Primary training losses (NEW - field error replaces MSE)
            "lossRecFieldError": [], "lossPredFieldError": [], "lossSpectrumError": [],
            # Perceptual loss
            "lossRecLSIM": [], "lossPredLSIM": [],
            # Regularization terms
            "lossRegMeanStd": [], "lossRegDiv": [], "lossRegVaeKLDiv": [], "lossRegLatStep": [],
        }
        self.batchComparison = {
            # Field error, LSIM, and spectrum error comparisons
            "compRec1.PredfieldError": [], "compRec1.PredLSIM": [], "compRec1.PredspectrumError": [],
            "comp1.LastPredfieldError": [], "comp1.LastPredLSIM": [], "comp1.LastPredspectrumError": [],
        }


    def updateBatch(self, lossParts:dict, lossSeq:dict, sample:int, timeMin:float):
        step = self.dataLoaderLength * self.epoch + sample

        for name in self.batchLoss:
            part = lossParts[name].detach().cpu().item()
            self.batchLoss[name] += [part]

        loss = lossParts["lossFull"].detach().cpu().item()

        # Rich progress bar update
        if self.use_rich_progress and self.progress is not None:
            if self.total_epochs is not None and self.total_epochs > 5:
                # Single progress bar for all training
                self.progress.update(
                    self.task_id,
                    advance=1,
                    loss_text=f"Epoch {self.epoch+1}/{self.total_epochs} | Loss: {loss:.4f}"
                )
            else:
                # Per-epoch progress bars
                if sample == 0:
                    description = f"[{self.model_name}] Epoch {self.epoch+1}" if self.model_name else f"Epoch {self.epoch+1}"
                    self.task_id = self.progress.add_task(
                        description,
                        total=self.dataLoaderLength,
                        loss_text=f"Loss: {loss:.4f}"
                    )
                else:
                    self.progress.update(
                        self.task_id,
                        advance=1,
                        loss_text=f"Loss: {loss:.4f}"
                    )
        else:
            # Original print behavior
            toPrint = self.printInterval > 0 and sample % self.printInterval == self.printInterval - 1
            toLog = self.logInterval > 0 and sample % self.logInterval == self.logInterval - 1

            if toPrint:
                print('[%2d, %4d] (%2.2f min): %1.4f' % (
                    self.epoch+1, sample+1, timeMin, loss))
            if toLog:
                logging.info('[%2d, %4d] (%2.2f min): %1.4f' % (
                    self.epoch+1, sample+1, timeMin, loss))

        # comparisons
        for name, lossTens in lossSeq.items():
            for prefix in ["compRec1.Pred", "comp1.LastPred"]:
                if not lossTens is None:
                    seq = lossTens.detach().cpu()
                    if prefix == "compRec1.Pred":
                        comp = seq[0].item()
                        if seq.shape[0] > 1:
                            comp = comp - seq[1].item()
                    elif prefix == "comp1.LastPred":
                        if seq.shape[0] > 2:
                            comp = seq[1].item() - seq[seq.shape[0]-1].item()
                        else:
                            comp = 0
                else:
                    comp = 0
                self.batchComparison["%s%s" % (prefix, name)] += [comp]


    def updateEpoch(self, timeMin:float):
        loss = 0
        partStr = ""
        for name, lossList in self.batchLoss.items():
            part = np.array(lossList)
            part = np.mean(part)
            self.writer.add_scalar("%s/epoch_%s" % (self.mode, name), part, self.epoch)

            if name == "lossFull":
                loss = part
            else:
                # Filter out losses that are not in use (weight == 0)
                should_include = True
                if self.loss_params is not None:
                    # Map loss name to loss_params attribute (NEW - field error components)
                    if name == "lossRecFieldError" and getattr(self.loss_params, 'recFieldError', 0) <= 0:
                        should_include = False
                    elif name == "lossPredFieldError" and getattr(self.loss_params, 'predFieldError', 0) <= 0:
                        should_include = False
                    elif name == "lossSpectrumError" and getattr(self.loss_params, 'spectrumError', 0) <= 0:
                        should_include = False
                    elif name == "lossRecLSIM" and getattr(self.loss_params, 'recLSIM', 0) <= 0:
                        should_include = False
                    elif name == "lossPredLSIM" and getattr(self.loss_params, 'predLSIM', 0) <= 0:
                        should_include = False
                    # Regularization terms - always show if non-zero
                    elif name.startswith("lossReg") and part <= 1e-6:
                        should_include = False

                if should_include:
                    partStr += "%s %1.3f " % (name.replace("loss", ""), part)

            # accuracy metrics
            accName = name.replace("lossRec", "r")
            accName = accName.replace("lossPred", "p")
            accName = accName.replace("lossFull", "Loss")
            self.accuracy["l_" + accName] = part
            accName = "b_" + accName
            if accName not in self.accuracy:
                self.accuracy[accName] = float("inf")
            if self.accuracy[accName] > part:
                self.accuracy[accName] = part

        # Complete progress bar for this epoch if using Rich (only for per-epoch mode)
        if self.use_rich_progress and self.progress is not None and self.task_id is not None:
            if not (self.total_epochs is not None and self.total_epochs > 5):
                # Only complete the task for per-epoch progress bars
                self.progress.update(self.task_id, completed=self.dataLoaderLength, loss_text=f"Loss: {loss:.4f}")

        # Only print every 5th epoch (or first epoch)
        if (self.epoch + 1) % 5 == 0 or self.epoch == 0:
            print("%s Epoch %d (%2.2f min): %1.4f    %s" % (self.modeLong, self.epoch+1, timeMin, loss, partStr))
            print("")
            logging.info("%s Epoch %d (%2.2f min): %1.4f    %s" % (self.modeLong, self.epoch+1, timeMin, loss, partStr))
            logging.info("")

        # comparisons
        for name, lossList in self.batchComparison.items():
            part = np.array(lossList)
            part = np.mean(part)
            self.writer.add_scalar("%s/epoch_%s" % (self.mode, name), part, self.epoch)

    def prepareAndClearForNextEpoch(self):
        self.clear()
        self.epoch += self.epochStep
        # Reset task_id for next epoch (only in per-epoch mode)
        if not (self.total_epochs is not None and self.total_epochs > 5):
            self.task_id = None

    def cleanup(self):
        """Cleanup Rich progress bar if it exists"""
        if self.use_rich_progress and self.progress is not None:
            self.progress.stop()


    def updateAccuracy(self, params:List, otherHistories:List["LossHistory"], finalPrint:bool):
        par = {}
        for p in params:
            if p:
                par.update(p.asDict())

        if finalPrint:
            print("")
            print("Final model performance:")
            logging.info("")
            logging.info("Final model performance:")

        histories = [self] + otherHistories
        metrics = {}
        for hist in histories:
            for stat,value in hist.accuracy.items():
                metrics["m/%s_%s" % (hist.mode, stat)] = value
                if finalPrint:
                    print("%s  %s: %1.3f" % (hist.mode, stat, value))
                    logging.info("%s %s: %1.3f" % (hist.mode, stat, value))

        self.writer.add_hparams(par, metrics)


    def writePredictionExample(self, prediction:torch.Tensor, groundTruth:torch.Tensor):
        numExamples = min(prediction.shape[0], 1)
        p = torch.transpose(prediction[0:numExamples], 3, 4)
        g = torch.transpose(groundTruth[0:numExamples], 3, 4)
        if groundTruth.ndim == 5:
            velX = torch.concat([g[:,:,0:1], p[:,:,0:1]], dim=3)
            velY = torch.concat([g[:,:,1:2], p[:,:,1:2]], dim=3)
            names = ["velX", "velY"]
            data = [velX, velY]
            for i in range(2, len(self.simFields)+2):
                field = torch.concat([g[:,:,i:i+1], p[:,:,i:i+1]], dim=3)
                names += [self.simFields[i-2]]
                data += [field]
        elif groundTruth.ndim == 6:
            velX = torch.concat([g[:,:,0:1], p[:,:,0:1]], dim=3).mean(5)
            velY = torch.concat([g[:,:,1:2], p[:,:,1:2]], dim=3).mean(5)
            velZ = torch.concat([g[:,:,2:3], p[:,:,2:3]], dim=3).mean(5)
            names = ["velX", "velY", "velZ"]
            data = [velX, velY, velZ]
            for i in range(3, len(self.simFields)+3):
                field = torch.concat([g[:,:,i:i+1], p[:,:,i:i+1]], dim=3).mean(5)
                names += [self.simFields[i-3]]
                data += [field]

        for i in range(len(data)):
            d = data[i]
            dMin = torch.amin(d, dim=(1,3,4), keepdim=True)
            dMax = torch.amax(d, dim=(1,3,4), keepdim=True)
            d = (d - dMin) / (dMax - dMin)

            step = int(d.shape[1] / 3.0)
            img = None
            if step < 1:
                img = d[0,[0,-1]]
            else:
                img = d[0,[0,step,2*step,-1]]
            d = d.expand(-1,-1,3,-1,-1)
            self.writer.add_images("%s_PredictionImg/%s" % (self.mode, names[i]), img, self.epoch)
            self.writer.add_video("%s_PredictionVid/%s" % (self.mode, names[i]), d, self.epoch, fps=5)


    def writeSequenceLoss(self, lossSeq:dict):
        # Check for field error metric
        if "fieldError" not in lossSeq:
            return  # No sequence data to plot

        field_error = lossSeq["fieldError"].cpu().numpy()

        fig, ax = plt.subplots(1, figsize=(5,2), tight_layout=True)
        ax.set_ylabel("Error\n(example batch)")
        ax.set_xlabel('Sequence Step')
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

        # Plot field error (primary metric)
        ax.plot(np.arange(field_error.shape[0]), field_error, linewidth=1.5, color="r", label="Field Error")

        # Plot LSIM if available
        if lossSeq.get("LSIM") is not None:
            lsim = lossSeq["LSIM"].cpu().numpy()
            ax.plot(np.arange(lsim.shape[0]), lsim, color="b", label="LSIM")

        # Plot spectrum error if available
        if "spectrumError" in lossSeq and lossSeq["spectrumError"] is not None:
            spectrum_error = lossSeq["spectrumError"].cpu().numpy()
            ax.plot(np.arange(spectrum_error.shape[0]), spectrum_error, linewidth=1.5, color="g", label="Spectrum Error")

        ax.legend()
        self.writer.add_figure("%s_PredictionImg/errorSequence" % (self.mode), fig, self.epoch)


    def get_validation_metrics(self) -> dict:
        """
        Extract validation metrics for loss scheduler.

        Returns dict with metrics like {'field_error': 0.1, 'spectrum_error': 0.05, 'lsim': 0.02}
        Used by FlexibleLossScheduler to monitor validation performance.
        """
        metrics = {}

        # Extract field error (primary metric)
        if "l_rFieldError" in self.accuracy:
            metrics["field_error"] = self.accuracy["l_rFieldError"]
        elif "l_pFieldError" in self.accuracy:
            metrics["field_error"] = self.accuracy["l_pFieldError"]

        # Extract spectrum error (for spectral bias monitoring)
        if "l_SpectrumError" in self.accuracy:
            metrics["spectrum_error"] = self.accuracy["l_SpectrumError"]

        # Extract LSIM (perceptual quality)
        if "l_rLSIM" in self.accuracy:
            metrics["lsim"] = self.accuracy["l_rLSIM"]
        elif "l_pLSIM" in self.accuracy:
            metrics["lsim"] = self.accuracy["l_pLSIM"]

        return metrics


