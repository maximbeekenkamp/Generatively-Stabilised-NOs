import logging
import os

import numpy as np
import torch
import torch.nn as nn
from neuralop.models import FNO

from src.core.models.model_dfpnet import DfpNet
from src.core.models.model_diffusion import DiffusionModel
from src.core.models.model_diffusion_blocks import Unet
from src.core.models.model_encoder import DecoderModelSkip, EncoderModelSkip
from src.core.models.model_latent_transformer import (
    LatentModelTransformer, LatentModelTransformerDec,
    LatentModelTransformerEnc, LatentModelTransformerMGN,
    LatentModelTransformerMGNParamEmb)
from src.core.models.model_refiner import PDERefiner
from src.core.models.model_resnet import DilatedResNet
from src.core.models.model_tno import TNOModel
from src.core.models.deeponet.deeponet_config import DeepONetConfig
from src.core.models.deeponet.mlp_networks import DeepONet as StandardDeepONet
from src.core.models.deeponet.deeponet_adapter import DeepONetFormatAdapter as DeepONetWrapper
from src.core.models.deepokan.deepokan_config import DeepOKANConfig
from src.core.models.deepokan.deepokan_base import DeepOKAN
from src.core.models.deepokan.deepokan_adapter import DeepOKANFormatAdapter as DeepOKANWrapper
from src.core.utils.params import (DataParams, LossParams, ModelParamsDecoder,
                             ModelParamsEncoder, ModelParamsLatent,
                             TrainingParams)
from src.core.utils.model_utils import get_prev_steps_from_arch


class PredictionModel(nn.Module):
    p_d: DataParams
    p_t: TrainingParams
    p_l: LossParams
    p_me: ModelParamsEncoder
    p_md: ModelParamsDecoder
    p_ml: ModelParamsLatent
    useGPU: bool

    def __init__(self, p_d:DataParams, p_t:TrainingParams, p_l:LossParams, p_me:ModelParamsEncoder, p_md:ModelParamsDecoder,
                p_ml:ModelParamsLatent, pretrainPath:str="", useGPU:bool=True):
        super(PredictionModel, self).__init__()

        self.p_d = p_d
        self.p_t = p_t
        self.p_l = p_l
        self.p_me = p_me
        self.p_md = p_md
        self.p_ml = p_ml
        self.useGPU = useGPU

        if (self.p_me and self.p_me.pretrained) or (self.p_md and self.p_md.pretrained) or (self.p_ml and self.p_ml.pretrained):
            if pretrainPath:
                loadedPretrainedWeightDict = torch.load(pretrainPath, weights_only=False)

        # ENCODER
        if self.p_me:
            if self.p_me.arch == "skip":
                self.modelEncoder = EncoderModelSkip(p_d, p_me, p_ml, p_d.dimension)
            else:
                raise ValueError("Unknown encoder architecture!")

            # load pretrained weights
            if pretrainPath and self.p_me.pretrained:
                self.modelEncoder.load_state_dict(loadedPretrainedWeightDict["stateDictEncoder"])

            # freeze weights
            if self.p_me.frozen:
                for param in self.modelEncoder.parameters():
                    param.requires_grad = False
        else:
            self.modelEncoder = None


        # DECODER
        if self.p_md:
            if self.p_md.arch == "skip":
                self.modelDecoder = DecoderModelSkip(p_d, p_me, p_md, p_ml, p_d.dimension)

            elif self.p_md.arch in ["unet", "unet+Prev", "unet+2Prev", "unet+3Prev",
                                    "dil_resnet", "dil_resnet+Prev", "dil_resnet+2Prev", "dil_resnet+3Prev",
                                    "resnet", "resnet+Prev", "resnet+2Prev", "resnet+3Prev",
                                    "fno", "fno+Prev", "fno+2Prev", "fno+3Prev",
                                    "dfp", "dfp+Prev", "dfp+2Prev", "dfp+3Prev",]:
                prevSteps = get_prev_steps_from_arch(self.p_md)

                inChannels = prevSteps * (self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams))
                outChannels = self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)

                if "unet" in self.p_md.arch:
                    self.modelDecoder = Unet(dim=self.p_d.dataSize[0], out_dim=outChannels, channels=inChannels,
                                        dim_mults=(1,1,1), use_convnext=True, convnext_mult=1, with_time_emb=False)

                elif "resnet" in self.p_md.arch:
                    self.modelDecoder = DilatedResNet(inFeatures=inChannels, outFeatures=outChannels, blocks=4, features=self.p_md.decWidth, dilate="dil_" in self.p_md.arch)

                elif "fno" in self.p_md.arch:
                    self.modelDecoder = FNO(n_modes=(self.p_md.fnoModes[0],self.p_md.fnoModes[1]), hidden_channels=self.p_md.decWidth, in_channels=inChannels, out_channels=outChannels, n_layers=4)

                elif "dfp" in self.p_md.arch:
                    self.modelDecoder = DfpNet(inChannels=inChannels, outChannels=outChannels, blockChannels=self.p_md.decWidth)

                else:
                    raise ValueError("Unknown decoder architecture")

            elif self.p_md.arch in ["tno", "tno+Prev", "tno+2Prev", "tno+3Prev"]:
                # TNO (Transformer Neural Operator) architecture
                # Determine L parameter from architecture name
                tno_L = get_prev_steps_from_arch(self.p_md)
                
                # TNO specific parameters
                tno_K = 1 if tno_L == 1 else 4  # Phase 0: K=1, Phase 1: K=4 temporal bundling
                tno_width = self.p_md.decWidth or 360  # Default width for turbulence
                dataset_type = self._infer_dataset_type()
                
                self.modelDecoder = TNOModel(
                    width=tno_width,
                    L=tno_L,
                    K=tno_K,
                    dataset_type=dataset_type
                )
                
                print(f"Initialized TNO: L={tno_L}, K={tno_K}, width={tno_width}, dataset={dataset_type}")

            elif self.p_md.arch in ["deeponet", "deeponet+Prev", "deeponet+2Prev", "deeponet+3Prev"]:
                # DeepONet (Deep Operator Network) architecture
                # Determine prev_steps from architecture name
                deeponet_prev_steps = get_prev_steps_from_arch(self.p_md)

                # Create DeepONet configuration from parameters
                deeponet_config = DeepONetConfig.from_params(self.p_md, self.p_d)

                # Spatial dimensions
                H, W = self.p_d.dataSize[-2], self.p_d.dataSize[-1]

                # Create base DeepONet model
                base_deeponet = StandardDeepONet(deeponet_config, self.p_md, self.p_d)

                # Wrap with format adapter for Gen Stabilised compatibility
                self.modelDecoder = DeepONetWrapper(
                    deeponet_model=base_deeponet,
                    spatial_dims=(H, W),
                    coordinate_dim=2  # 2D spatial coordinates
                )

                print(f"Initialized DeepONet: latent_dim={deeponet_config.latent_dim}, n_sensors={deeponet_config.n_sensors}, prev_steps={deeponet_prev_steps}")

            elif self.p_md.arch in ["deepokan", "deepokan+Prev", "deepokan+2Prev", "deepokan+3Prev"]:
                # DeepOKAN (Deep Operator Network with KAN layers) architecture
                # Determine prev_steps from architecture name
                deepokan_prev_steps = get_prev_steps_from_arch(self.p_md)

                # Create DeepOKAN configuration from parameters
                deepokan_config = DeepOKANConfig.from_params(self.p_md, self.p_d)

                # Spatial dimensions
                H, W = self.p_d.dataSize[-2], self.p_d.dataSize[-1]

                # Create base DeepOKAN model
                base_deepokan = DeepOKAN(deepokan_config)

                # Wrap with format adapter for Gen Stabilised compatibility
                self.modelDecoder = DeepOKANWrapper(
                    deepokan_model=base_deepokan,
                    spatial_dims=(H, W),
                    num_channels=self.p_d.dimension + len(self.p_d.simFields)
                )

                print(f"Initialized DeepOKAN: HD={deepokan_config.HD}, sensor_dim={deepokan_config.sensor_dim}, grid_count={deepokan_config.grid_count}")

            elif self.p_md.arch in ["genop", "genop+Prev", "genop+2Prev", "genop+3Prev",
                                    "genop-fno-diffusion", "genop-fno-diffusion+Prev", "genop-fno-diffusion+2Prev", "genop-fno-diffusion+3Prev",
                                    "genop-tno-diffusion", "genop-tno-diffusion+Prev", "genop-tno-diffusion+2Prev", "genop-tno-diffusion+3Prev",
                                    "genop-unet-diffusion", "genop-unet-diffusion+Prev", "genop-unet-diffusion+2Prev", "genop-unet-diffusion+3Prev",
                                    "genop-deeponet-diffusion", "genop-deeponet-diffusion+Prev", "genop-deeponet-diffusion+2Prev", "genop-deeponet-diffusion+3Prev",
                                    "genop-deepokan-diffusion", "genop-deepokan-diffusion+Prev", "genop-deepokan-diffusion+2Prev", "genop-deepokan-diffusion+3Prev",
                                    "nodm", "nodm+Prev", "nodm+2Prev", "nodm+3Prev"]:

                # Import generative operator components
                from .genop_init import initialize_generative_operators
                from .model_registry import ModelRegistry, parse_generative_operator_architecture
                from .generative_operator_model import create_generative_operator_model

                # Parse architecture to determine prior and corrector types
                # Note: Auto-initialization in genop_init.py already registers all models
                try:
                    if self.p_md.arch.startswith("genop"):
                        prior_type, corrector_type = parse_generative_operator_architecture(self.p_md.arch)
                    elif self.p_md.arch.startswith("nodm"):
                        # Default NO+DM configuration
                        prior_type, corrector_type = "fno", "diffusion"
                    else:
                        # Default generative operator
                        prior_type, corrector_type = "fno", "diffusion"

                except ValueError as e:
                    raise ValueError(f"Invalid generative operator architecture '{self.p_md.arch}': {e}")

                # Validate combination is supported
                if not ModelRegistry.validate_combination(prior_type, corrector_type):
                    available_priors = list(ModelRegistry.list_priors().keys())
                    available_correctors = list(ModelRegistry.list_correctors().keys())
                    raise ValueError(f"Unsupported combination: {prior_type} + {corrector_type}. "
                                   f"Available priors: {available_priors}, correctors: {available_correctors}")

                # Create generative operator model
                try:
                    self.modelDecoder = create_generative_operator_model(
                        prior_name=prior_type,
                        corrector_name=corrector_type,
                        p_md=self.p_md,
                        p_d=self.p_d,
                        pretrain_checkpoint=pretrainPath,
                        enable_dcar=True,
                        memory_efficient=True
                    )

                    # Set initial training mode based on architecture
                    if hasattr(self.p_md, 'training_stage'):
                        if self.p_md.training_stage == 1:
                            self.modelDecoder.set_training_mode('prior_only')
                        elif self.p_md.training_stage == 2:
                            self.modelDecoder.set_training_mode('corrector_training')
                            self.modelDecoder.freeze_prior()
                        else:
                            self.modelDecoder.set_training_mode('full_inference')
                    else:
                        # Default to full inference mode
                        self.modelDecoder.set_training_mode('full_inference')

                    print(f"Initialized Generative Operator: {prior_type} + {corrector_type}")

                except Exception as e:
                    raise RuntimeError(f"Failed to create generative operator model: {e}")

            elif self.p_md.arch in ["decode-ddpm", "decode-ddim", "direct-ddpm+First", "direct-ddim+First",
                                    "direct-ddpm", "direct-ddim", "direct-ddpm+Prev", "direct-ddim+Prev",
                                    "direct-ddpm+2Prev", "direct-ddim+2Prev", "direct-ddpm+3Prev", "direct-ddim+3Prev", 
                                    "dfp-ddpm", "dfp-ddpm+Prev", "dfp-ddpm+2Prev", "dfp-ddpm+3Prev",
                                    "direct-ddpm+Enc", "direct-ddim+Enc", "hybrid-ddpm+Lat", "hybrid-ddim+Lat"]:
                if self.p_md.arch in ["decode-ddpm", "decode-ddim"]:
                    condChannels = self.p_me.latentSize + len(self.p_d.simParams)
                elif self.p_md.arch in ["direct-ddpm", "direct-ddim", "dfp-ddpm"]:
                    condChannels = self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)
                elif self.p_md.arch in ["direct-ddpm+First", "direct-ddim+First", "direct-ddpm+Prev", "direct-ddim+Prev", "dfp-ddpm+Prev"]:
                    condChannels = 2 * (self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams))
                elif self.p_md.arch in ["direct-ddpm+2Prev", "direct-ddim+2Prev", "dfp-ddpm+2Prev"]:
                    condChannels = 3 * (self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams))
                elif self.p_md.arch in ["direct-ddpm+3Prev", "direct-ddim+3Prev", "dfp-ddpm+3Prev"]:
                    condChannels = 4 * (self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams))
                elif self.p_md.arch in ["direct-ddpm+Enc", "direct-ddim+Enc", "hybrid-ddpm+Lat", "hybrid-ddim+Lat"]:
                    condChannels = (self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)) + (self.p_me.latentSize + len(self.p_d.simParams))
                self.modelDecoder = DiffusionModel(p_d, p_md, p_d.dimension, condChannels=condChannels)

            elif self.p_md.arch == "refiner":
                condChannels = self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)
                self.modelDecoder = PDERefiner(p_d, p_md, condChannels=condChannels)


            elif self.p_md.arch in ["skip+finetune-ddpm", "skip+finetune-ddim"]:
                self.modelDecoder = nn.ModuleList([
                    DecoderModelSkip(p_d, p_me, p_md, p_ml, p_d.dimension),
                    DiffusionModel(p_d, p_md, p_d.dimension, condChannels=self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams))
                ])

            elif self.p_md.arch in ["skip+hybrid-ddpm", "skip+hybrid-ddim"]:
                self.modelDecoder = nn.ModuleList([
                    DecoderModelSkip(p_d, p_me, p_md, p_ml, p_d.dimension),
                    DiffusionModel(p_d, p_md, p_d.dimension, condChannels=2 * (self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)))
                ])

            else:
                raise ValueError("Unknown decoder architecture!")

            # load pretraining weights
            if pretrainPath and self.p_md.pretrained:
                # Skip loading for GenerativeOperatorModel - it loads its prior during creation
                if hasattr(self.modelDecoder, '__class__') and 'GenerativeOperatorModel' in str(type(self.modelDecoder)):
                    # GenerativeOperatorModel already loaded pretrained prior via create_generative_operator_model
                    pass
                elif self.p_md.arch in ["skip+finetune-ddpm", "skip+finetune-ddim", "skip+hybrid-ddpm", "skip+hybrid-ddim"]:
                    self.modelDecoder[0].load_state_dict(loadedPretrainedWeightDict["stateDictDecoder"])
                else:
                    # Handle both old and new checkpoint formats
                    if "stateDictDecoder" in loadedPretrainedWeightDict:
                        # Old format: separate stateDictDecoder key
                        state_dict = loadedPretrainedWeightDict["stateDictDecoder"]
                    elif "model_state_dict" in loadedPretrainedWeightDict:
                        # New format: model_state_dict from Trainer.trainingStep
                        state_dict = loadedPretrainedWeightDict["model_state_dict"]
                    else:
                        raise KeyError("Checkpoint missing both 'stateDictDecoder' and 'model_state_dict' keys")

                    self.modelDecoder.load_state_dict(state_dict, strict=False)

            # freeze weights
            if self.p_md.frozen:
                if self.p_md.arch in ["skip+finetune-ddpm", "skip+finetune-ddim", "skip+hybrid-ddpm", "skip+hybrid-ddim"]:
                    for param in self.modelDecoder[0].parameters():
                        param.requires_grad = False
                else:
                    for param in self.modelDecoder.parameters():
                        param.requires_grad = False
        else:
            self.modelDecoder = None

        # LATENT MODEL
        if self.p_ml:
            if self.p_ml.arch == "transformerEnc":
                self.modelLatent = LatentModelTransformerEnc(p_d, p_me, p_ml, False)
            elif self.p_ml.arch == "transformerDec":
                self.modelLatent = LatentModelTransformerDec(p_d, p_me, p_ml)
            elif self.p_ml.arch == "transformer":
                self.modelLatent = LatentModelTransformer(p_d, p_me, p_ml)
            elif self.p_ml.arch == "transformerMGN":
                self.modelLatent = LatentModelTransformerMGN(p_d, p_me, p_ml)
                self.modelLatentParamEmb = LatentModelTransformerMGNParamEmb(p_d, p_me)
            else:
                raise ValueError("Unknown latent architecture!")

            # load pretraining weights
            if pretrainPath and self.p_ml.pretrained:
                self.modelLatent.load_state_dict(loadedPretrainedWeightDict["stateDictLatent"])

            # freeze weights
            if self.p_ml.frozen:
                for param in self.modelLatent.parameters():
                    param.requires_grad = False
        else:
            self.modelLatent = None

        self.to("cuda" if self.useGPU else "cpu")

    def _infer_dataset_type(self):
        """
        Infer dataset type from simulation fields and parameters
        Used to configure TNO for specific turbulence datasets
        """
        # Check simulation parameters to identify dataset
        if hasattr(self.p_d, 'simParams') and self.p_d.simParams:
            # Check for Mach parameter (transonic dataset)
            if "mach" in self.p_d.simParams:
                return "tra"  # Transonic cylinder dataset (Mach with or without Reynolds)
            # Check for Reynolds only (incompressible dataset)
            elif "rey" in self.p_d.simParams:
                return "inc"  # Incompressible wake (Reynolds only)

        # Check simulation fields as fallback
        if hasattr(self.p_d, 'simFields') and self.p_d.simFields:
            if "velZ" in self.p_d.simFields:
                return "iso"  # Isotropic turbulence (has 3D velocity)

        # Default to incompressible
        return "inc"

    def forward(self, data:torch.Tensor, simParameters:torch.Tensor, useLatent:bool=True, stepsLong:int=-1) -> torch.Tensor:
        device = "cuda" if self.useGPU else "cpu"
        d = data.to(device)
        simParam = simParameters.to(device) if simParameters is not None else None

        # ENCODING - LATENT MODEL - DECODING
        if not (self.p_md.arch in ["unet", "unet+Prev", "unet+2Prev", "unet+3Prev",
                "dil_resnet", "dil_resnet+Prev", "dil_resnet+2Prev", "dil_resnet+3Prev",
                "resnet", "resnet+Prev", "resnet+2Prev", "resnet+3Prev",
                "fno", "fno+Prev", "fno+2Prev", "fno+3Prev",
                "dfp", "dfp+Prev", "dfp+2Prev", "dfp+3Prev",
                "tno", "tno+Prev", "tno+2Prev", "tno+3Prev",
                "deeponet", "deeponet+Prev", "deeponet+2Prev", "deeponet+3Prev",
                "refiner",
                "direct-ddpm", "direct-ddim", "direct-ddpm+First", "direct-ddim+First",
                "direct-ddpm+Prev", "direct-ddim+Prev", "direct-ddpm+2Prev", "direct-ddim+2Prev",
                "direct-ddpm+3Prev", "direct-ddim+3Prev", "direct-ddpm+Enc", "direct-ddim+Enc",
                "dfp-ddpm", "dfp-ddpm+Prev", "dfp-ddpm+2Prev", "dfp-ddpm+3Prev",
                "genop", "genop+Prev", "genop+2Prev", "genop+3Prev",
                "genop-fno-diffusion", "genop-fno-diffusion+Prev", "genop-fno-diffusion+2Prev", "genop-fno-diffusion+3Prev",
                "genop-tno-diffusion", "genop-tno-diffusion+Prev", "genop-tno-diffusion+2Prev", "genop-tno-diffusion+3Prev",
                "genop-unet-diffusion", "genop-unet-diffusion+Prev", "genop-unet-diffusion+2Prev", "genop-unet-diffusion+3Prev",
                "genop-deeponet-diffusion", "genop-deeponet-diffusion+Prev", "genop-deeponet-diffusion+2Prev", "genop-deeponet-diffusion+3Prev",
                "nodm", "nodm+Prev", "nodm+2Prev", "nodm+3Prev"]):

            latentSpace = torch.zeros(d.shape[0], d.shape[1], self.p_me.latentSize)
            latentSpace = latentSpace.to(device)
            if not self.modelLatent or not useLatent:
                # no latent model -> fully process sequence with AE
                latentSpace = self.modelEncoder(d)
            else:
                if isinstance(self.modelLatent, LatentModelTransformerEnc):
                    latentSpace = self.forwardTransEnc(d, latentSpace, simParam)

                elif isinstance(self.modelLatent, LatentModelTransformerDec) or isinstance(self.modelLatent, LatentModelTransformer):
                    latentSpace = self.forwardTransDec(d, latentSpace, simParam)

                elif isinstance(self.modelLatent, LatentModelTransformerMGN):
                    latentSpace = self.forwardTransMGN(d, latentSpace, simParam)

                else:
                    raise ValueError("Invalid latent model!")

            if "decode" in self.p_md.arch:
                prediction = self.forwardDiffusionDecode(d, latentSpace, simParam)
                return prediction, latentSpace, (None, None)

            elif "finetune" in self.p_md.arch:
                prediction = self.forwardDiffusionFinetune(d, latentSpace, simParam)
                return prediction, latentSpace, (None, None)

            elif "hybrid" in self.p_md.arch:
                prediction = self.forwardDiffusionHybrid(d, latentSpace, simParam)
                return prediction, latentSpace, (None, None)

            else:
                prediction, vaeMeanVar = self.modelDecoder(latentSpace, simParam)
                return prediction, latentSpace, vaeMeanVar


        # DIRECT PREDICTION OF NEXT FRAME WITH DIFFERENT ARCHITECTURES
        else:
            if isinstance(self.modelDecoder, Unet) or isinstance(self.modelDecoder, DilatedResNet) or isinstance(self.modelDecoder, FNO):
                if stepsLong > 0 and (not self.training):
                    prediction = self.forwardDirectLongGPUEfficient(d, steps=stepsLong)
                else:
                    prediction = self.forwardDirect(d)
                return prediction, None, (None, None)

            elif isinstance(self.modelDecoder, TNOModel):
                # TNO forward path
                if stepsLong > 0 and (not self.training):
                    # Long rollout for evaluation
                    prediction = self.forwardTNOLong(d, steps=stepsLong)
                else:
                    # Single-step or bundled prediction for training
                    prediction = self.forwardTNO(d)
                return prediction, None, (None, None)

            elif isinstance(self.modelDecoder, DeepONetWrapper):
                # DeepONet forward path
                if stepsLong > 0 and (not self.training):
                    prediction = self.forwardDirectLongGPUEfficient(d, steps=stepsLong)
                else:
                    prediction = self.forwardDirect(d)
                return prediction, None, (None, None)

            elif hasattr(self.modelDecoder, '__class__') and 'GenerativeOperatorModel' in str(type(self.modelDecoder)):
                # Generative operator forward path (NO+DM and variants)
                if stepsLong > 0 and (not self.training):
                    # Long DCAR rollout for evaluation
                    prediction = self.forwardGenerativeOperatorLong(d, steps=stepsLong)
                else:
                    # Standard prediction for training
                    prediction = self.forwardGenerativeOperator(d)
                return prediction, None, (None, None)

            else:
                if stepsLong > 0 and (not self.training):
                    prediction = self.forwardDiffusionDirectLongGPUEfficient(d, simParam, steps=stepsLong)
                else:
                    prediction = self.forwardDiffusionDirect(d, simParam)

                return prediction, None, (None, None)




    # Transformer encoder latent model
    def forwardTransEnc(self, d:torch.Tensor, latentSpace:torch.Tensor, simParam:torch.Tensor) -> torch.Tensor:
        sizeSeq = d.shape[1]

        # transformer encoder latent model to predict single next step
        if self.training and not self.p_ml.transTrainUnroll:
            encLatentSpace = self.modelEncoder(d)
            transLatentSpace = self.modelLatent(encLatentSpace, simParam)
            latentSpace = torch.concat([encLatentSpace[:,:1], transLatentSpace[:,:-1]], dim=1)

        # transformer encoder latent model to predict all steps from first one
        else:
            latentSpace[:,0:1] = self.modelEncoder(d[:,0:1])
            for i in range(1,sizeSeq):
                start = max(0, i-self.p_ml.maxInputLen) if self.p_ml.maxInputLen > 0 else 0
                transLatentSpace = self.modelLatent(latentSpace[:,start:i], simParam[:,start:i] if simParam is not None else None)
                latentSpace[:,i] = transLatentSpace[:,-1]

        return latentSpace


    # Transformer decoder latent model
    def forwardTransDec(self, d:torch.Tensor, latentSpace:torch.Tensor, simParam:torch.Tensor) -> torch.Tensor:
        sizeSeq = d.shape[1]

        # transformer latent model to predict single next step
        if self.training and not self.p_ml.transTrainUnroll:
            encLatentSpace = self.modelEncoder(d)
            transLatentSpace = self.modelLatent(encLatentSpace[:,:-1], encLatentSpace[:,1:], simParam[:,:-1] if simParam is not None else None, simParam[:,1:] if simParam is not None else None)
            latentSpace = torch.concat([encLatentSpace[:,:1], transLatentSpace], dim=1)

        # transformer latent model to predict all steps from first one
        else:
            latentSpace[:,0:1] = self.modelEncoder(d[:,0:1])
            for i in range(1,sizeSeq):
                if self.p_ml.transTargetFull:
                    start = max(0, i-self.p_ml.maxInputLen) if self.p_ml.maxInputLen > 0 else 0
                    transLatentSpace = self.modelLatent(latentSpace[:,start:i], latentSpace[:,start:i], simParam[:,start:i] if simParam is not None else None, simParam[:,start:i] if simParam is not None else None)
                    latentSpace[:,i] = transLatentSpace[:,-1]
                    #latentSpace[:,i] = latentSpace[:,i-1] + transLatentSpace[:,-1]
                else:
                    transLatentSpace = self.modelLatent(latentSpace[:,:i], latentSpace[:,i-1:i], simParam[:,:i] if simParam is not None else None, simParam[:,i-1:i] if simParam is not None else None)
                    latentSpace[:,i:i+1] = transLatentSpace

        return latentSpace


    # Transformer latent model according to MeshGraphNet paper
    def forwardTransMGN(self, d:torch.Tensor, latentSpace:torch.Tensor, simParam:torch.Tensor) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], d.shape[1]

        latentSpace = torch.zeros(sizeBatch, sizeSeq+1, self.p_me.latentSize)
        latentSpace = latentSpace.to("cuda" if self.useGPU else "cpu")

        if simParam is not None:
            latentSpace[:,0] = self.modelLatentParamEmb(simParam[:,0]) # only use scalar simParam input
        latentSpace[:,1:2] = self.modelEncoder(d[:,0:1])
        for i in range(2,sizeSeq+1):
            start = max(1, i-self.p_ml.maxInputLen) if self.p_ml.maxInputLen > 0 else 1
            transInput = torch.concat([latentSpace[:,0:1], latentSpace[:,start:i]], dim=1)
            transLatentSpace = self.modelLatent(transInput, latentSpace[:,i-1:i])
            latentSpace[:,i:i+1] = latentSpace[:,i-1:i] + transLatentSpace
        latentSpace = latentSpace[:,1:] # discard param embedding

        return latentSpace


    # Direct prediction of next step via U-Net, ResNet, FNO, etc.
    def forwardDirect(self, d:torch.Tensor) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], d.shape[1]

        prevSteps = get_prev_steps_from_arch(self.p_md)

        prediction = []
        #for i in range(4):
        for i in range(prevSteps): # no prediction of first steps
            if self.training:
                trainNoise = self.p_md.trainingNoise * torch.normal(torch.zeros_like(d[:,i]), torch.ones_like(d[:,i]))
                prediction += [d[:,i] + trainNoise]
            else:
                prediction += [d[:,i]]

        for i in range(prevSteps, sizeSeq):
            uIn = torch.concat(prediction[i-prevSteps : i], dim=1)

            if isinstance(self.modelDecoder, FNO):
                result = self.modelDecoder(uIn)
            else:
                result = self.modelDecoder(uIn, None)
    
            if self.p_d.simParams:
                result[:,-len(self.p_d.simParams):] = d[:,i,-len(self.p_d.simParams):] # replace simparam prediction with true values
            prediction += [result]

        prediction = torch.stack(prediction, dim=1)
        return prediction


    # GPU EFFICIENT VARIANT OF DIRECT PREDICTION FOR EXTREMELY LONG SEQUENCES
    def forwardDirectLongGPUEfficient(self, d:torch.Tensor, steps:int) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], steps

        if "+Prev" in self.p_md.arch or "+2Prev" in self.p_md.arch or "+3Prev" in self.p_md.arch:
            raise ValueError("GPU efficient variant only supports 1 previous step!")

        prediction = [d[:,0]]

        uIn = prediction[0].to("cuda")
        self.modelDecoder = self.modelDecoder.to("cuda")
        for i in range(1, sizeSeq):

            if isinstance(self.modelDecoder, FNO):
                result = self.modelDecoder(uIn)
            else:
                result = self.modelDecoder(uIn, None)

            if self.p_d.simParams:
                result[:,-len(self.p_d.simParams):] = d[:,0,-len(self.p_d.simParams):].to("cuda") # replace simparam prediction with true values

            uIn = result

            if i % 100 == 0:
                # Move to CPU every 100 steps to save memory
                result = result.to("cpu")
                prediction += [result]

        prediction = torch.stack(prediction, dim=1)
        return prediction


    # Diffusion model to directly predict next step based on different conditionings
    def forwardDiffusionDirect(self, d:torch.Tensor, simParams:torch.Tensor) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], d.shape[1]

        # TRAINING
        if self.training:
            if "+Enc" in self.p_md.arch:
                latentSpace = self.modelEncoder(d[:,0:1])
                l = torch.concat((latentSpace, simParams[:,0:1]), dim=2) if simParams is not None else latentSpace
                conditioning = l.unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,d.shape[3],d.shape[4])

                randIndex = torch.randint(1, sizeSeq, (1,), device=d.device)
                conditioning = torch.concat((conditioning, d[:,randIndex-1:randIndex]), dim=2)
                data = d[:,randIndex]

            elif "+First" in self.p_md.arch:
                randIndex = torch.randint(1, sizeSeq, (1,), device=d.device)
                conditioning = torch.concat((d[:,0:1], d[:,randIndex-1:randIndex]), dim=2)
                data = d[:,randIndex:randIndex+1]

            else:
                prevSteps = get_prev_steps_from_arch(self.p_md)

                cond = []
                for i in range(prevSteps):
                    trainNoise = self.p_md.trainingNoise * torch.normal(torch.zeros_like(d[:,i:i+1]), torch.ones_like(d[:,i:i+1]))
                    cond += [d[:, i:i+1] + trainNoise] # collect input steps
                conditioning = torch.concat(cond, dim=2) # combine along channel dimension
                data = d[:, prevSteps:prevSteps+1]

            noise, predictedNoise = self.modelDecoder(conditioning=conditioning, data=data)
            return noise, predictedNoise


        # INFERENCE
        else:
            prediction = torch.zeros_like(d, device="cuda" if self.useGPU else "cpu")

            if "+Enc" in self.p_md.arch:
                prediction[:,0] = d[:,0] # no prediction of first step
                latentSpace = self.modelEncoder(d[:,0:1])
                l = torch.concat((latentSpace, simParams[:,0:1]), dim=2) if simParams is not None else latentSpace
                conditioning = l.unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,d.shape[3],d.shape[4])
                for i in range(1,sizeSeq):
                    cond = torch.concat((conditioning, prediction[:,i-1:i]), dim=2)
                    result = self.modelDecoder(conditioning=cond, data=d[:,i-1:i])
                    if self.p_d.simParams:
                        result[:,:,-len(self.p_d.simParams):] = d[:,i:i+1,-len(self.p_d.simParams):] # replace simparam prediction with true values
                    prediction[:,i:i+1] = result

            elif "+First" in self.p_md.arch:
                prediction[:,0] = d[:,0] # no prediction of first step
                for i in range(1,sizeSeq):
                    cond = torch.concat((d[:,0:1], prediction[:,i-1:i]), dim=2)
                    result = self.modelDecoder(conditioning=cond, data=d[:,i-1:i])
                    if self.p_d.simParams:
                        result[:,:,-len(self.p_d.simParams):] = d[:,i:i+1,-len(self.p_d.simParams):] # replace simparam prediction with true values
                    prediction[:,i:i+1] = result


            else:
                prevSteps = get_prev_steps_from_arch(self.p_md)

                #for i in range(4):
                for i in range(prevSteps): # no prediction of first steps
                    prediction[:,i] = d[:,i] 

                for i in range(prevSteps, sizeSeq):
                    cond = []
                    for j in range(prevSteps,0,-1):
                        cond += [prediction[:, i-j : i-(j-1)]] # collect input steps
                    cond = torch.concat(cond, dim=2) # combine along channel dimension

                    result = self.modelDecoder(conditioning=cond, data=d[:,i-1:i]) # auto-regressive inference
                    if self.p_d.simParams:
                        result[:,:,-len(self.p_d.simParams):] = d[:,i:i+1,-len(self.p_d.simParams):] # replace simparam prediction with true values
                    prediction[:,i:i+1] = result

            return prediction


    # GPU EFFICIENT VARIANT FOR DIFFUSION MODELS FOR EXTREMELY LONG SEQUENCES
    def forwardDiffusionDirectLongGPUEfficient(self, d:torch.Tensor, simParams:torch.Tensor, steps:int) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], steps

        # TRAINING
        if self.training:
            raise ValueError("Training not supported for GPU efficient variant!")

        # INFERENCE
        else:
            if not "+Prev" in self.p_md.arch:
                raise ValueError("GPU efficient variant only supports 2 previous step!")

            prediction = [d[:,0:1]]

            uInPrev = d[:,0:1].to("cuda")
            uIn = d[:,1:2].to("cuda")
            self.modelDecoder = self.modelDecoder.to("cuda")

            for i in range(2, sizeSeq):
                cond = torch.concat([uInPrev, uIn], dim=2) # combine along channel dimension

                result = self.modelDecoder(conditioning=cond, data=torch.zeros_like(uIn, device="cuda")) # auto-regressive inference

                if self.p_d.simParams:
                    result[:,:,-len(self.p_d.simParams):] = d[:,0:1,-len(self.p_d.simParams):].to("cuda") # replace simparam prediction with true values
                
                uInPrev = uIn
                uIn = result

                if i % 100 == 0:
                    # Move to CPU every 100 steps to save memory
                    result = result.to("cpu")
                    prediction += [result]

            prediction = torch.concat(prediction, dim=1)
            return prediction


    # Decoder diffusion model conditioned on latent space
    def forwardDiffusionDecode(self, d:torch.Tensor, latentSpace:torch.Tensor, simParams:torch.Tensor) -> torch.Tensor:
        # add simulation parameters to latent space
        l = torch.concat((latentSpace, simParams), dim=2) if simParams is not None else latentSpace

        # match dimensionality
        cond = l.unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,d.shape[3],d.shape[4])

        if self.training:
            noise, predictedNoise = self.modelDecoder(conditioning=cond, data=d) # prediction conditioned on latent space
            return noise, predictedNoise
        else:
            prediction = self.modelDecoder(conditioning=cond, data=d)
            return prediction


    # Diffusion model conditioned on normal decoder ouput to finetune it
    def forwardDiffusionFinetune(self, d:torch.Tensor, latentSpace:torch.Tensor, simParams:torch.Tensor) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], d.shape[1]

        cond, _ = self.modelDecoder[0](latentSpace, simParams)

        if self.training:
            noise, predictedNoise = self.modelDecoder[1](conditioning=cond, data=d)
            return noise, predictedNoise
        else:
            prediction = self.modelDecoder[1](conditioning=cond, data=d)
            return prediction


    # Diffusion model predicts next step based on previous step and secondary transformer network conditioning
    def forwardDiffusionHybrid(self, d:torch.Tensor, latentSpace:torch.Tensor, simParams:torch.Tensor) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], d.shape[1]
        if self.training:
            if "+Lat" in self.p_md.arch:
                randIndex = torch.randint(1, sizeSeq-1, (1,), device=d.device)

                l = torch.concat((latentSpace, simParams), dim=2) if simParams is not None else latentSpace
                conditioning = l.unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,d.shape[3],d.shape[4])

                conditioning = torch.concat((conditioning[:,randIndex:randIndex+1], d[:,randIndex-1:randIndex]), dim=2)
                data = d[:,randIndex:randIndex+1]

                noise, predictedNoise = self.modelDecoder(conditioning=conditioning, data=data)

            elif "skip+" in self.p_md.arch:
                predictionAeDec, _ = self.modelDecoder[0](latentSpace, simParams)

                randIndex = torch.randint(1, sizeSeq-1, (1,), device=d.device)

                conditioning = torch.concat((predictionAeDec[:,randIndex:randIndex+1], d[:,randIndex-1:randIndex]), dim=2)
                data = d[:,randIndex:randIndex+1]

                noise, predictedNoise = self.modelDecoder[1](conditioning=conditioning, data=data)

            return noise, predictedNoise

        else:
            prediction = torch.zeros_like(d, device="cuda" if self.useGPU else "cpu")
            prediction[:,0] = d[:,0] # no prediction of first and last step
            prediction[:,d.shape[1]-1] = d[:,d.shape[1]-1]

            if "+Lat" in self.p_md.arch:
                l = torch.concat((latentSpace, simParams), dim=2) if simParams is not None else latentSpace
                conditioning = l.unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,d.shape[3],d.shape[4])

                for i in range(1,sizeSeq-1):
                    cond = torch.concat((conditioning[:,i:i+1], prediction[:,i-1:i]), dim=2)
                    result = self.modelDecoder(conditioning=cond, data=d[:,i-1:i])
                    if self.p_d.simParams:
                        result[:,:,-len(self.p_d.simParams):] = d[:,i:i+1,-len(self.p_d.simParams):] # replace simparam prediction with true values
                    prediction[:,i:i+1] = result

            elif "skip+" in self.p_md.arch:
                predictionAeDec, _ = self.modelDecoder[0](latentSpace, simParams)

                for i in range(1,sizeSeq-1):
                    cond = torch.concat((predictionAeDec[:,i:i+1], prediction[:,i-1:i]), dim=2)
                    result = self.modelDecoder[1](conditioning=cond, data=d[:,i-1:i])
                    if self.p_d.simParams:
                        result[:,:,-len(self.p_d.simParams):] = d[:,i:i+1,-len(self.p_d.simParams):] # replace simparam prediction with true values
                    prediction[:,i:i+1] = result

            return prediction


    def printModelInfo(self):
        pTrain = filter(lambda p: p.requires_grad, self.parameters())
        paramsTrain = sum([np.prod(p.size()) for p in pTrain])
        params = sum([np.prod(p.size()) for p in self.parameters()])

        if self.modelEncoder:
            pTrainEnc = filter(lambda p: p.requires_grad, self.modelEncoder.parameters())
            paramsTrainEnc = sum([np.prod(p.size()) for p in pTrainEnc])
            paramsEnc = sum([np.prod(p.size()) for p in self.modelEncoder.parameters()])
        if self.modelDecoder:
            pTrainDec = filter(lambda p: p.requires_grad, self.modelDecoder.parameters())
            paramsTrainDec = sum([np.prod(p.size()) for p in pTrainDec])
            paramsDec = sum([np.prod(p.size()) for p in self.modelDecoder.parameters()])
        if self.modelLatent:
            pTrainLat = filter(lambda p: p.requires_grad, self.modelLatent.parameters())
            paramsTrainLat = sum([np.prod(p.size()) for p in pTrainLat])
            paramsLat = sum([np.prod(p.size()) for p in self.modelLatent.parameters()])

        # Log model information (removed redundant print statements)
        logging.info("Weights Trainable (All): %d (%d)   %s   %s   %s" %
                (paramsTrain, params,
                ("Enc: %d (%d)" % (paramsTrainEnc, paramsEnc)) if self.modelEncoder else "",
                ("Dec: %d (%d)" % (paramsTrainDec, paramsDec)) if self.modelDecoder else "",
                ("Lat: %d (%d)" % (paramsTrainLat, paramsLat)) if self.modelLatent else ""))
        logging.info(self)
        logging.info("Data parameters: %s" % str(self.p_d.asDict()))
        logging.info("Training parameters: %s" % str(self.p_t.asDict()))
        logging.info("Loss parameters: %s" % str(self.p_l.asDict()))
        if self.p_me:
            logging.info("Model Encoder parameters: %s" % str(self.p_me.asDict()))
        if self.p_md:
            logging.info("Model Decoder parameters: %s" % str(self.p_md.asDict()))
        if self.p_ml:
            logging.info("Model Latent parameters: %s" % str(self.p_ml.asDict()))
        logging.info("")



    @classmethod
    def load(cls, path:str, useGPU:bool=True):
        if useGPU:
            print('Loading model from %s' % path)
            loaded = torch.load(path)
        else:
            print('CPU - Loading model from %s' % path)
            loaded = torch.load(path, map_location=torch.device('cpu'))

        p_me = ModelParamsEncoder().fromDict(loaded['modelParamsEncoder']) if loaded['modelParamsEncoder'] else None
        p_md = ModelParamsDecoder().fromDict(loaded['modelParamsDecoder']) if loaded['modelParamsDecoder'] else None
        p_ml = ModelParamsLatent().fromDict(loaded['modelParamsLatent'])   if loaded['modelParamsLatent'] else None
        p_d = DataParams().fromDict(loaded['dataParams'])                  if loaded['dataParams'] else None
        p_t = TrainingParams().fromDict(loaded['trainingParams'])          if loaded['trainingParams'] else None
        p_l = LossParams().fromDict(loaded['lossParams'])                  if loaded['lossParams'] else None

        stateDictEncoder = loaded['stateDictEncoder']
        stateDictDecoder = loaded['stateDictDecoder']
        stateDictLatent = loaded['stateDictLatent']

        model = cls(p_d, p_t, p_l, p_me, p_md, p_ml, "", useGPU)

        if stateDictEncoder:
            model.modelEncoder.load_state_dict(stateDictEncoder)
        if stateDictDecoder:
            model.modelDecoder.load_state_dict(stateDictDecoder)
        if stateDictLatent:
            model.modelLatent.load_state_dict(stateDictLatent)
        model.eval()

        return model

    def forwardTNO(self, d: torch.Tensor) -> torch.Tensor:
        """
        Enhanced TNO forward pass for training - Phase 1.2
        
        Features:
        - Improved temporal bundling with dynamic K adaptation
        - Better multi-channel prediction handling
        - Phase-aware processing
        - Memory-efficient implementation
        
        Args:
            d: Input tensor [B, T, C, H, W] where T is sequence length
        
        Returns:
            prediction: Output tensor [B, T, C, H, W] with predictions
        """
        sizeBatch, sizeSeq = d.shape[0], d.shape[1]
        device = d.device
        dtype = d.dtype
        
        # Get TNO parameters
        K = self.modelDecoder.K  # Temporal bundling size
        L = self.modelDecoder.L  # History length
        
        # Initialize prediction with first timestep
        prediction = [d[:, 0]]
        
        # Enhanced temporal bundling loop
        i = 1
        while i < sizeSeq:
            # Dynamic K adaptation based on remaining steps
            remaining_steps = sizeSeq - i
            effective_K = min(K, remaining_steps)
            
            # Prepare TNO input with proper history
            input_start = max(0, i - L)
            input_end = i
            
            # Extract input sequence for TNO
            if input_end > input_start:
                tno_input = d[:, input_start:input_end + 1]  # [B, L+1, C, H, W]
            else:
                # Edge case: not enough history, pad with first timestep
                tno_input = d[:, 0:1].expand(-1, L + 1, -1, -1, -1)
            
            # TNO forward pass - handles multi-channel internally now
            tno_output = self.modelDecoder(tno_input)  # [B, K, target_channels, H, W]
            
            # Extract only needed predictions
            tno_predictions = tno_output[:, :effective_K]  # [B, effective_K, target_channels, H, W]
            
            # Handle different field reconstruction strategies
            num_input_fields = d.shape[2]
            target_channels = tno_predictions.shape[2]
            
            # Create full field predictions
            for step_idx in range(effective_K):
                # Get single step prediction
                step_pred = tno_predictions[:, step_idx]  # [B, target_channels, H, W]
                
                # Create full field tensor
                full_pred = torch.zeros(
                    sizeBatch, num_input_fields, d.shape[3], d.shape[4],
                    device=device, dtype=dtype
                )
                
                # Strategy 1: Direct mapping for single-channel predictions
                if target_channels == 1:
                    full_pred[:, 0:1] = step_pred
                    # Preserve other fields (simulation parameters, etc.)
                    if num_input_fields > 1 and i + step_idx < sizeSeq:
                        full_pred[:, 1:] = d[:, i + step_idx, 1:]
                        
                # Strategy 2: Multi-channel mapping for complex predictions
                elif target_channels > 1:
                    # Map multiple channels from TNO to corresponding input fields
                    channels_to_copy = min(target_channels, num_input_fields)
                    full_pred[:, :channels_to_copy] = step_pred[:, :channels_to_copy]
                    
                    # Preserve remaining fields (typically simulation parameters)
                    if num_input_fields > target_channels and i + step_idx < sizeSeq:
                        full_pred[:, target_channels:] = d[:, i + step_idx, target_channels:]
                
                prediction.append(full_pred)
            
            # Move to next position
            i += effective_K
        
        # Stack all predictions
        prediction = torch.stack(prediction, dim=1)  # [B, T, C, H, W]
        
        return prediction
    
    def forwardTNOLong(self, d: torch.Tensor, steps: int = 20, batch_size: int = None, use_temporal_bundling: bool = True) -> torch.Tensor:
        """
        Enhanced TNO long rollout for evaluation - Phase 1.2
        
        Features:
        - Memory-efficient sliding window approach
        - Optional temporal bundling for faster rollouts
        - Batch processing for very long sequences
        - Better error handling and stability
        
        Args:
            d: Input tensor [B, T, C, H, W]
            steps: Number of steps to predict
            batch_size: Optional batch size for chunked processing
            use_temporal_bundling: Whether to use K>1 bundling for efficiency
        
        Returns:
            prediction: Extended prediction [B, T+steps, C, H, W]
        """
        sizeBatch, _, num_fields, H, W = d.shape
        device = d.device
        dtype = d.dtype
        
        # Get TNO parameters
        L = self.modelDecoder.L
        K = self.modelDecoder.K if use_temporal_bundling else 1
        
        # Memory-efficient sliding window approach
        window_size = max(L + 1, 10)  # Keep minimal history
        prediction = d.clone()
        
        # Process in chunks if batch size is specified
        effective_batch_size = batch_size if batch_size and batch_size < steps else steps
        
        predicted_steps = 0
        while predicted_steps < steps:
            # Determine how many steps to predict in this iteration
            remaining_steps = steps - predicted_steps
            steps_this_batch = min(K, remaining_steps, effective_batch_size)
            
            # Extract input sequence (sliding window)
            if prediction.shape[1] > window_size:
                # Use only recent history to save memory
                input_seq = prediction[:, -window_size:]  # [B, window_size, C, H, W]
            else:
                input_seq = prediction
            
            # Get last L+1 steps for TNO input
            tno_input = input_seq[:, -(L+1):]  # [B, L+1, C, H, W]
            
            try:
                # TNO forward pass
                tno_output = self.modelDecoder(tno_input)  # [B, K, target_channels, H, W]
                
                # Extract predictions for this batch
                batch_predictions = tno_output[:, :steps_this_batch]  # [B, steps_this_batch, target_channels, H, W]
                
                # Convert to full field format
                new_steps = []
                for step_idx in range(steps_this_batch):
                    step_pred = batch_predictions[:, step_idx]  # [B, target_channels, H, W]
                    
                    # Create full field prediction
                    full_pred = torch.zeros(sizeBatch, num_fields, H, W, device=device, dtype=dtype)
                    
                    # Handle different channel mapping strategies
                    target_channels = step_pred.shape[1]
                    if target_channels == 1:
                        # Single channel prediction
                        full_pred[:, 0:1] = step_pred
                        # Preserve simulation parameters
                        if num_fields > 1:
                            full_pred[:, 1:] = prediction[:, -1, 1:]
                    elif target_channels > 1:
                        # Multi-channel prediction
                        channels_to_copy = min(target_channels, num_fields)
                        full_pred[:, :channels_to_copy] = step_pred[:, :channels_to_copy]
                        # Preserve remaining fields
                        if num_fields > target_channels:
                            full_pred[:, target_channels:] = prediction[:, -1, target_channels:]
                    
                    new_steps.append(full_pred)
                
                # Stack new predictions and append
                new_predictions = torch.stack(new_steps, dim=1)  # [B, steps_this_batch, C, H, W]
                prediction = torch.cat([prediction, new_predictions], dim=1)
                
                predicted_steps += steps_this_batch
                
                # Memory cleanup for very long rollouts
                if prediction.shape[1] > window_size * 2:
                    # Keep only recent history to prevent memory issues
                    prediction = prediction[:, -window_size:]
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Fallback to smaller batches
                    logging.warning(f"Memory error at step {predicted_steps}, reducing batch size from {K} to {K//2}")
                    K = max(1, K // 2)
                    steps_this_batch = min(K, remaining_steps, 1)
                    continue
                else:
                    raise e
                    
        return prediction

    def set_tno_training_phase(self, phase: str):
        """
        Set TNO training phase if using TNO decoder - Phase 1.2
        
        Args:
            phase: "teacher_forcing" or "fine_tuning"
        """
        if hasattr(self.modelDecoder, 'set_training_phase'):
            self.modelDecoder.set_training_phase(phase)
            logging.info(f"TNO training phase set to: {phase}")
    
    def update_tno_epoch(self, epoch: int):
        """
        Update TNO epoch for automatic phase transition - Phase 1.2
        
        Args:
            epoch: Current training epoch
        """
        if hasattr(self.modelDecoder, 'update_epoch'):
            self.modelDecoder.update_epoch(epoch)
    
    def get_tno_status(self):
        """
        Get current TNO configuration and status - Phase 1.2
        
        Returns:
            Dictionary with TNO status or None if not using TNO
        """
        if hasattr(self.modelDecoder, 'get_status'):
            status = self.modelDecoder.get_status()
            return {
                'model_type': 'TNO',
                'L': status['L'],
                'K': status['K'],
                'training_phase': status['training_phase'],
                'current_epoch': status['current_epoch'],
                'dataset_type': status['dataset_type'],
                'teacher_forcing_epochs': status.get('teacher_forcing_epochs', 'N/A'),
                'width': status['width'],
                'target_size': status.get('target_size', 'N/A')
            }
        return None
    
    def get_tno_forward_info(self):
        """
        Get TNO-specific forward pass information - Phase 1.2
        
        Returns:
            Information about TNO forward capabilities
        """
        if hasattr(self.modelDecoder, 'get_status'):
            status = self.modelDecoder.get_status()
            return {
                'temporal_bundling': status['K'] > 1,
                'memory_length': status['L'],
                'supports_multi_channel': len(status['field_config']['target_fields']) > 1,
                'phase_aware': True,
                'long_rollout_optimized': True
            }
        return None

    def forwardGenerativeOperator(self, d: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through generative operator model for training.

        Args:
            d: Input tensor [B, T, C, H, W]

        Returns:
            prediction: Predicted tensor [B, T, C, H, W]
        """
        # Direct forward through the generative operator model
        prediction = self.modelDecoder(d)
        return prediction

    def forwardGenerativeOperatorLong(self, d: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Long DCAR rollout for generative operator evaluation.

        Args:
            d: Input tensor [B, T, C, H, W] where T is the initial condition length
            steps: Number of additional steps to predict

        Returns:
            prediction: Full trajectory [B, T + steps, C, H, W]
        """
        # Use DCAR rollout from the generative operator model
        if hasattr(self.modelDecoder, 'dcar_rollout'):
            prediction = self.modelDecoder.dcar_rollout(
                initial_states=d,
                num_steps=steps
            )
        else:
            # Fallback to standard iterative rollout if DCAR not available
            prediction = self._fallback_generative_rollout(d, steps)

        return prediction

    def _fallback_generative_rollout(self, d: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Fallback iterative rollout for generative operators.

        Args:
            d: Initial condition [B, T, C, H, W]
            steps: Number of steps to predict

        Returns:
            trajectory: Full prediction [B, T + steps, C, H, W]
        """
        device = d.device
        B, T, C, H, W = d.shape

        # Initialize trajectory with input
        trajectory = d.clone()

        # Determine window size for input
        window_size = 1
        if hasattr(self.modelDecoder, 'prior_model'):
            if hasattr(self.modelDecoder.prior_model, 'prev_steps'):
                window_size = self.modelDecoder.prior_model.prev_steps
            elif hasattr(self.modelDecoder.prior_model, 'L'):
                window_size = self.modelDecoder.prior_model.L

        for step in range(steps):
            # Extract input window
            if trajectory.shape[1] >= window_size:
                input_window = trajectory[:, -window_size:]
            else:
                # Pad with last frame if needed
                last_frame = trajectory[:, -1:].repeat(1, window_size - trajectory.shape[1], 1, 1, 1)
                input_window = torch.cat([last_frame, trajectory], dim=1)

            # Predict next frame
            with torch.no_grad():
                prediction = self.modelDecoder(input_window)
                next_frame = prediction[:, -1:]  # Take last predicted frame

            # Append to trajectory
            trajectory = torch.cat([trajectory, next_frame], dim=1)

        return trajectory

    def get_generative_operator_info(self):
        """
        Get generative operator specific information.

        Returns:
            Information about the generative operator model
        """
        if hasattr(self.modelDecoder, 'get_model_info'):
            return self.modelDecoder.get_model_info()
        return None

    def save(self, basePath:str, epoch:int=-1, noPrint:bool=False):
        if not noPrint:
            print('Saving model to %s' % basePath)

        saveDict = {
            'stateDictEncoder'   : self.modelEncoder.state_dict() if self.modelEncoder else None,
            'stateDictDecoder'   : self.modelDecoder.state_dict() if self.modelDecoder else None,
            'stateDictLatent'    : self.modelLatent.state_dict() if self.modelLatent else None,
            'modelParamsEncoder' : self.p_me.asDict() if self.p_me else None,
            'modelParamsDecoder' : self.p_md.asDict() if self.p_md else None,
            'modelParamsLatent'  : self.p_ml.asDict() if self.p_ml else None,
            'dataParams'         : self.p_d.asDict() if self.p_d else None,
            'trainingParams'     : self.p_t.asDict() if self.p_t else None,
            'lossParams'         : self.p_l.asDict() if self.p_l else None,
            }

        if epoch > 0:
            path = os.path.join(basePath, "Model_E%03d.pth" % epoch)
        else:
            path = os.path.join(basePath, "Model.pth")
        torch.save(saveDict, path)

