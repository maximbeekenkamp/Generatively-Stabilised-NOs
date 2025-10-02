

class DataParams(object):
    def __init__(self, batch=4, augmentations=[], sequenceLength=[1,1], randSeqOffset=False,
                dataSize=[128,64], dimension=2, simFields=[], simParams=[], normalizeMode=""):
        self.batch          = batch             # batch size
        self.augmentations  = augmentations     # used data augmentations
        self.sequenceLength = sequenceLength    # number of simulation frames in one sequence
        self.randSeqOffset  = randSeqOffset     # randomize sequence starting frame
        self.dataSize       = dataSize          # target data size for scale/crop/cropRandom transformation
        self.dimension      = dimension         # number of data dimension
        self.simFields      = simFields         # which simulation fields are added (vel is always used) from ["dens", "pres"]
        self.simParams      = simParams         # which simulation parameters are added from ["rey", "mach"]
        self.normalizeMode  = normalizeMode     # which mean and std values from different data sets are used in normalization transformation

    @classmethod
    def fromDict(cls, d:dict):
        p = cls()
        p.batch          = d.get("batch",            -1)
        p.augmentations  = d.get("augmentations",    [])
        p.sequenceLength = d.get("sequenceLength",   [])
        p.randSeqOffset  = d.get("randSeqOffset",    False)
        p.dataSize       = d.get("dataSize",         -1)
        p.dimension      = d.get("dimension",        -1)
        p.simFields      = d.get("simFields",        [])
        p.simParams      = d.get("simParams",        [])
        p.normalizeMode  = d.get("normalizeMode",    "")
        return p

    def asDict(self) -> dict:
        return {
            "batch"          : self.batch,
            "augmentations"  : self.augmentations,
            "sequenceLength" : self.sequenceLength,
            "randSeqOffset"  : self.randSeqOffset,
            "dataSize"       : self.dataSize,
            "dimension"      : self.dimension,
            "simFields"      : self.simFields,
            "simParams"      : self.simParams,
            "normalizeMode"  : self.normalizeMode,
        }



class TrainingParams(object):
    def __init__(self, epochs=20, lr=0.0001, expLrGamma=1.0, weightDecay=0.0, fadeInPredLoss=[-1,0], fadeInSeqLen=[-1,0], fadeInSeqLenLin=False,
                 # Two-stage training parameters for generative operators
                 stage1_epochs=None, stage2_epochs=None, stage1_lr=None, stage2_lr=None,
                 training_stage=None, freeze_prior_after_stage1=True,
                 # Generative corrector parameters
                 correction_strength=1.0, correction_schedule=None,
                 diffusion_num_steps=32, diffusion_sigma_data=0.5,
                 # DCAR rollout parameters
                 enable_dcar=True, dcar_correction_frequency=1,
                 # Memory optimization
                 memory_efficient=True, gradient_checkpointing=False):

        # Standard training parameters
        self.epochs            = epochs            # number of training epochs
        self.lr                = lr                # learning rate
        self.expLrGamma        = expLrGamma        # factor for exponential learning rate decay
        self.weightDecay       = weightDecay       # weight decay factor to regularize the net by penalizing large weights
        self.fadeInPredLoss    = fadeInPredLoss    # start and end epoch of fading in the prediction loss terms
        self.fadeInSeqLen      = fadeInSeqLen      # start and end epoch of fading in the sequence length
        self.fadeInSeqLenLin   = fadeInSeqLenLin   # exponential or linear scaling of fading in the sequence length

        # Two-stage training parameters for generative operators
        self.stage1_epochs = stage1_epochs or epochs // 2  # epochs for stage 1 (prior only)
        self.stage2_epochs = stage2_epochs or epochs // 2  # epochs for stage 2 (corrector training)
        self.stage1_lr = stage1_lr or lr               # learning rate for stage 1
        self.stage2_lr = stage2_lr or lr * 0.1         # learning rate for stage 2 (typically lower)
        self.training_stage = training_stage           # current training stage (1, 2, or None for auto)
        self.freeze_prior_after_stage1 = freeze_prior_after_stage1  # freeze prior in stage 2

        # Generative corrector parameters
        self.correction_strength = correction_strength    # correction strength (0.0-2.0)
        self.correction_schedule = correction_schedule    # schedule for correction strength [(epoch, strength), ...]
        self.diffusion_num_steps = diffusion_num_steps   # number of diffusion denoising steps
        self.diffusion_sigma_data = diffusion_sigma_data # diffusion data noise standard deviation

        # DCAR rollout parameters
        self.enable_dcar = enable_dcar                   # enable DCAR rollout for evaluation
        self.dcar_correction_frequency = dcar_correction_frequency  # correct every N steps in rollout

        # Memory optimization
        self.memory_efficient = memory_efficient         # enable memory-efficient inference
        self.gradient_checkpointing = gradient_checkpointing  # enable gradient checkpointing
        
    @classmethod
    def fromDict(cls, d:dict):
        p = cls()
        # Standard parameters
        p.epochs            = d.get("epochs",           -1)
        p.lr                = d.get("lr",               -1)
        p.expLrGamma        = d.get("expLrGamma",        1)
        p.weightDecay       = d.get("weightDecay",      -1)
        p.fadeInPredLoss    = d.get("fadeInPredLoss",   [])
        p.fadeInSeqLen      = d.get("fadeInSeqLen",     [])
        p.fadeInSeqLenLin   = d.get("fadeInSeqLenLin",  False)

        # Two-stage training parameters
        p.stage1_epochs = d.get("stage1_epochs", p.epochs // 2 if p.epochs > 0 else 10)
        p.stage2_epochs = d.get("stage2_epochs", p.epochs // 2 if p.epochs > 0 else 10)
        p.stage1_lr = d.get("stage1_lr", p.lr)
        p.stage2_lr = d.get("stage2_lr", p.lr * 0.1 if p.lr > 0 else 0.00001)
        p.training_stage = d.get("training_stage", None)
        p.freeze_prior_after_stage1 = d.get("freeze_prior_after_stage1", True)

        # Generative corrector parameters
        p.correction_strength = d.get("correction_strength", 1.0)
        p.correction_schedule = d.get("correction_schedule", None)
        p.diffusion_num_steps = d.get("diffusion_num_steps", 32)
        p.diffusion_sigma_data = d.get("diffusion_sigma_data", 0.5)

        # DCAR rollout parameters
        p.enable_dcar = d.get("enable_dcar", True)
        p.dcar_correction_frequency = d.get("dcar_correction_frequency", 1)

        # Memory optimization
        p.memory_efficient = d.get("memory_efficient", True)
        p.gradient_checkpointing = d.get("gradient_checkpointing", False)

        return p

    def asDict(self) -> dict:
        result = {
            # Standard parameters
            "epochs"            : self.epochs,
            "lr"                : self.lr,
            "expLrGamma"        : self.expLrGamma,
            "weightDecay"       : self.weightDecay,
            "fadeInPredLoss"    : self.fadeInPredLoss,
            "fadeInSeqLen"      : self.fadeInSeqLen,
            "fadeInSeqLenLin"   : self.fadeInSeqLenLin,
        }

        # Add new parameters if they exist (for backward compatibility)
        if hasattr(self, 'stage1_epochs'):
            result.update({
                # Two-stage training parameters
                "stage1_epochs": self.stage1_epochs,
                "stage2_epochs": self.stage2_epochs,
                "stage1_lr": self.stage1_lr,
                "stage2_lr": self.stage2_lr,
                "training_stage": self.training_stage,
                "freeze_prior_after_stage1": self.freeze_prior_after_stage1,

                # Generative corrector parameters
                "correction_strength": self.correction_strength,
                "correction_schedule": self.correction_schedule,
                "diffusion_num_steps": self.diffusion_num_steps,
                "diffusion_sigma_data": self.diffusion_sigma_data,

                # DCAR rollout parameters
                "enable_dcar": self.enable_dcar,
                "dcar_correction_frequency": self.dcar_correction_frequency,

                # Memory optimization
                "memory_efficient": self.memory_efficient,
                "gradient_checkpointing": self.gradient_checkpointing,
            })

        return result



class LossParams(object):
    def __init__(self, recMSE=1.0, recLSIM=0, predMSE=1.0, predLSIM=0, extraMSEvelZ=0, regMeanStd=0, regDiv=0, regVae=0, regLatStep=0,
                 tno_lp_loss=0.0, tno_transition_weight=0.1,
                 # Generative operator loss parameters
                 diffusion_loss_weight=1.0, consistency_loss_weight=0.1, prior_loss_weight=1.0,
                 adversarial_loss_weight=0.0, perceptual_loss_weight=0.0):

        # Standard loss parameters
        self.recMSE       = recMSE       # mse loss reconstruction weight
        self.recLSIM      = recLSIM      # lsim loss reconstruction weight
        self.predMSE      = predMSE      # mse loss prediction weight
        self.predLSIM     = predLSIM     # lsim loss prediction weight
        self.regMeanStd   = regMeanStd   # mean and standard deviation regularization weight
        self.regDiv       = regDiv       # divergence regularization weight
        self.regVae       = regVae       # regularization weight for VAE KL divergence
        self.regLatStep   = regLatStep   # latent space step regularization weight

        # TNO-specific loss parameters - Phase 1.3
        self.tno_lp_loss = tno_lp_loss   # TNO relative LpLoss2 weight
        self.tno_transition_weight = tno_transition_weight  # Weight for gradual phase transitions

        # Generative operator loss parameters
        self.diffusion_loss_weight = diffusion_loss_weight    # weight for diffusion model loss
        self.consistency_loss_weight = consistency_loss_weight  # weight for prior-target consistency loss
        self.prior_loss_weight = prior_loss_weight            # weight for neural operator prior loss
        self.adversarial_loss_weight = adversarial_loss_weight  # weight for adversarial loss (future GAN support)
        self.perceptual_loss_weight = perceptual_loss_weight  # weight for perceptual loss (future enhancement)

    @classmethod
    def fromDict(cls, d:dict):
        p = cls()
        # Standard loss parameters
        p.recMSE       = d.get("recMSE", -1)
        p.recLSIM      = d.get("recLSIM", -1)
        p.predMSE      = d.get("predMSE", -1)
        p.predLSIM     = d.get("predLSIM", -1)
        p.regMeanStd   = d.get("regMeanStd", -1)
        p.regDiv       = d.get("regDiv", -1)
        p.regVae       = d.get("regVae", -1)
        p.regLatStep   = d.get("regLatStep", -1)

        # TNO-specific parameters - Phase 1.3
        p.tno_lp_loss  = d.get("tno_lp_loss", 0.0)
        p.tno_transition_weight = d.get("tno_transition_weight", 0.1)

        # Generative operator loss parameters
        p.diffusion_loss_weight = d.get("diffusion_loss_weight", 1.0)
        p.consistency_loss_weight = d.get("consistency_loss_weight", 0.1)
        p.prior_loss_weight = d.get("prior_loss_weight", 1.0)
        p.adversarial_loss_weight = d.get("adversarial_loss_weight", 0.0)
        p.perceptual_loss_weight = d.get("perceptual_loss_weight", 0.0)

        return p

    def asDict(self) -> dict:
        result = {
            # Standard loss parameters
            "recMSE"       : self.recMSE,
            "recLSIM"      : self.recLSIM,
            "predMSE"      : self.predMSE,
            "predLSIM"     : self.predLSIM,
            "regMeanStd"   : self.regMeanStd,
            "regDiv"       : self.regDiv,
            "regVae"       : self.regVae,
            "regLatStep"   : self.regLatStep,

            # TNO-specific parameters - Phase 1.3
            "tno_lp_loss"  : self.tno_lp_loss,
            "tno_transition_weight" : self.tno_transition_weight,
        }

        # Add generative operator parameters if they exist (for backward compatibility)
        if hasattr(self, 'diffusion_loss_weight'):
            result.update({
                "diffusion_loss_weight": self.diffusion_loss_weight,
                "consistency_loss_weight": self.consistency_loss_weight,
                "prior_loss_weight": self.prior_loss_weight,
                "adversarial_loss_weight": self.adversarial_loss_weight,
                "perceptual_loss_weight": self.perceptual_loss_weight,
            })

        return result



class ModelParamsEncoder(object):
    def __init__(self, arch="skip", pretrained=False, frozen=False, encWidth=16, latentSize=16):
        self.arch = arch              # architecture variant
        self.pretrained = pretrained  # load pretrained weight initialization
        self.frozen = frozen          # freeze weights after initialization
        self.encWidth = encWidth      # width of encoder network
        self.latentSize = latentSize  # size of latent space vector

    @classmethod
    def fromDict(cls, d:dict):
        p = cls()
        p.arch       = d.get("arch", "")
        p.pretrained = d.get("pretrained", False)
        p.frozen     = d.get("frozen", False)
        p.encWidth   = d.get("encWidth", -1)
        p.latentSize = d.get("latentSize", -1)
        return p

    def asDict(self) -> dict:
        return {
            "arch"       : self.arch,
            "pretrained" : self.pretrained,
            "frozen"     : self.frozen,
            "encWidth"   : self.encWidth,
            "latentSize" : self.latentSize,
        }



class ModelParamsDecoder(object):
    def __init__(self, arch="skip", pretrained=False, frozen=False, decWidth=48, vae=False, trainingNoise=0.0,
                 diffSteps=500, diffSchedule="linear", diffCondIntegration="noisy", fnoModes=(16,16), refinerStd=0.0):
        self.arch = arch                 # architecture variant
        self.pretrained = pretrained     # load pretrained weight initialization
        self.frozen = frozen             # freeze weights after initialization
        self.decWidth = decWidth         # width of decoder network
        self.vae = vae                   # use a variational AE setup
        self.trainingNoise = trainingNoise # amount of training noise added to inputs
        self.diffSteps = diffSteps       # diffusion model diffusion time steps
        self.diffSchedule = diffSchedule # diffusion model variance schedule
        self.diffCondIntegration = diffCondIntegration # integrationg of conditioning during diffusion training
        self.fnoModes = fnoModes         # number of fourier modes for FNO setup
        self.refinerStd = refinerStd     # noise standard dev. in pde refiner setup

    @classmethod
    def fromDict(cls, d:dict):
        p = cls()
        p.arch         = d.get("arch", "")
        p.pretrained   = d.get("pretrained", False)
        p.frozen       = d.get("frozen", False)
        p.decWidth     = d.get("decWidth", -1)
        p.vae          = d.get("vae", False)
        p.trainingNoise= d.get("trainingNoise", 0.0)
        p.diffSteps    = d.get("diffSteps", 500)
        p.diffSchedule = d.get("diffSchedule", "linear")
        p.diffCondIntegration  = d.get("diffCondIntegration", "noisy")
        p.fnoModes     = d.get("fnoModes", ())
        p.refinerStd   = d.get("refinerStd", 0.0)
        return p

    def asDict(self) -> dict:
        return {
            "arch"         : self.arch,
            "pretrained"   : self.pretrained,
            "frozen"       : self.frozen,
            "decWidth"     : self.decWidth,
            "vae"          : self.vae,
            "trainingNoise": self.trainingNoise,
            "diffSteps"    : self.diffSteps,
            "diffSchedule" : self.diffSchedule,
            "diffCondIntegration" : self.diffCondIntegration,
            "fnoModes"     : self.fnoModes,
            "refinerStd"   : self.refinerStd,
        }



class ModelParamsLatent(object):
    def __init__(self, arch="fc", pretrained=False, frozen=False, width=512, layers=6, heads=4, dropout=0.0,
               transTrainUnroll=False, transTargetFull=False, maxInputLen=-1):
        self.arch = arch                         # architecture variant
        self.pretrained = pretrained             # load pretrained weight initialization
        self.frozen = frozen                     # freeze weights after initialization
        self.width = width                       # latent network width
        self.layers = layers                     # number of latent network layers
        self.heads = heads                       # number of attention heads in transformer
        self.dropout = dropout                   # dropout rate in latent network
        self.transTrainUnroll = transTrainUnroll # unrolled training for transformer latent models, FALSE for one step predictions TRUE for full rollouts
        self.transTargetFull = transTargetFull   # full target data for transformer and transformer decoder latent models, FALSE for only the previous step as a target TRUE for every previous step as a target
        self.maxInputLen = maxInputLen           # how many steps of the input sequence are processed at once for models that predict full sequences (-1 for no limit)


    @classmethod
    def fromDict(cls, d:dict):
        p = cls()
        p.arch             = d.get("arch", "")
        p.pretrained       = d.get("pretrained", False)
        p.frozen           = d.get("frozen", False)
        p.width            = d.get("width", "")
        p.layers           = d.get("layers", "")
        p.heads            = d.get("heads", "")
        p.dropout          = d.get("dropout", "")
        p.transTrainUnroll = d.get("transTrainUnroll", False)
        p.transTargetFull  = d.get("transTargetFull", False)
        p.maxInputLen      = d.get("maxInputLen", -1)
        return p

    def asDict(self) -> dict:
        return {
            "arch"             : self.arch,
            "pretrained"       : self.pretrained,
            "frozen"           : self.frozen,
            "width"            : self.width,
            "layers"           : self.layers,
            "heads"            : self.heads,
            "dropout"          : self.dropout,
            "transTrainUnroll" : self.transTrainUnroll,
            "transTargetFull"  : self.transTargetFull,
            "maxInputLen"      : self.maxInputLen,
        }