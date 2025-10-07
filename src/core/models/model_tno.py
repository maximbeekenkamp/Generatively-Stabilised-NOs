"""
TNO (Transformer Neural Operator) Model for Gen Stabilised
Adapted from the original TNO implementation for turbulence prediction

Original TNO: https://github.com/amazon-science/transformer-neural-operator
Paper: "Transformer Neural Operator for Turbulence Forecasting"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import operator
from functools import reduce


class U_net_2D(nn.Module):
    """
    2D U-Net architecture for branch and t-branch networks in TNO
    """
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net_2D, self).__init__()
        self.input_channels = input_channels

        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv2_1 = self.conv_block(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv3_1 = self.conv_block(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate)

        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels*2, output_channels)
        self.deconv0 = self.deconv(input_channels*2, output_channels)

        self.output_layer = self.output(input_channels*2, output_channels, kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_deconv2 = self.deconv2(out_conv3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)
        return out
    
    def _initialize_weights(self, module):
        """Xavier initialization"""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            init.xavier_normal_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
    
    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        layers = nn.Sequential(
            nn.Conv2d(in_planes, output_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True)
        )
        layers.apply(self._initialize_weights)
        return layers

    def conv_block(self, in_channels, out_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_rate),
            nn.SiLU(inplace=True)
        )

    def deconv(self, input_channels, output_channels):
        layers = nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True)
        )
        layers.apply(self._initialize_weights)
        return layers

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        layer = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2)
        layer.apply(self._initialize_weights)
        return layer


class TimeNN(nn.Module):
    """
    Time Neural Network (Trunk network) for processing coordinate grids
    """
    def __init__(self, in_width, hidden_width=128, output_dim=128, num_layers=10):
        super(TimeNN, self).__init__()
        self.hidden_dim = hidden_width
        self.num_layers = num_layers
        
        # Create layers dynamically
        layers = []
        for i in range(num_layers):
            in_dim = in_width if i == 0 else hidden_width
            layers.append(self._create_linear(in_dim, hidden_width))
        
        self.layers = nn.ModuleList(layers)
        self.output_layer = self._create_linear(hidden_width, output_dim)
        self.activation = F.tanh

    def _create_linear(self, in_dim, out_dim):
        """Create linear layer with Xavier initialization"""
        layer = nn.Linear(in_dim, out_dim)
        init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            init.zeros_(layer.bias)
        return layer

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x


class TNO(nn.Module):
    """
    Core TNO (Transformer Neural Operator) architecture
    
    TNO combines three components:
    1. Branch network: Processes auxiliary functions (boundary conditions, parameters)
    2. Trunk network: Processes coordinate grids
    3. T-Branch network: Processes solution history
    
    The outputs are combined via Hadamard product and decoded to predictions.
    """
    def __init__(self, width, L, K, target_size=(80, 160)):
        super(TNO, self).__init__()

        self.width = width
        self.L = L  # Input sequence length
        self.K = K  # Output prediction length
        self.target_size = target_size

        # Adaptive pooling for resolution invariance
        self.adaptive_pool_T = nn.AdaptiveAvgPool2d(self.target_size)
        self.adaptive_pool_B = nn.AdaptiveAvgPool2d(self.target_size)

        # Lifting layers
        self.fc_T_1 = nn.Linear(self.L, self.width)  # T-Branch lifting
        self.fc_B_1 = nn.Linear(self.L, self.width)  # Branch lifting

        # U-Net processors
        self.unet_T = U_net_2D(self.width, self.width, 3, 0)
        self.unet_B = U_net_2D(self.width, self.width, 3, 0)

        # Trunk (Time Neural Network) - processes (t, x, y) coordinates
        self.time_nn = TimeNN(in_width=3, hidden_width=self.width, output_dim=self.width, num_layers=12)

        # Decoder network
        self.fc_2 = nn.Linear(self.width, self.width)
        self.fc_3 = nn.Linear(self.width, self.width)
        self.fc_4 = nn.Linear(self.width, self.K)

    def forward(self, branch_input, trunk_input, tbranch_input):
        """
        Forward pass through TNO - Corrected Architecture

        Args:
            branch_input: Auxiliary field [B, 1, H, W] - single timepoint
            trunk_input: Coordinate grids [B, H, W, 3] - (t, x, y) coordinates
            tbranch_input: Solution history [B, L*C_sol, H, W] - flattened temporal history

        Returns:
            predictions: [B, H, W, K] prediction for next K timesteps
        """
        original_size = branch_input.shape[2:4]  # [H, W]
        B = branch_input.shape[0]

        # === Trunk Network - process (t,x,y) coordinates ===
        # Input: [B, H, W, 3] → Output: [B, H, W, width]
        H, W = trunk_input.shape[1], trunk_input.shape[2]
        trunk_flat = trunk_input.reshape(-1, 3)  # [B*H*W, 3]
        trunk_processed = self.time_nn(trunk_flat)  # [B*H*W, width]
        trunk_processed = trunk_processed.reshape(B, H, W, self.width)  # [B, H, W, width]

        # === Branch Network - process auxiliary field ===
        # Input: [B, 1, H, W] → Output: [B, H, W, width]
        # Lift 1 channel → width channels using 1x1 conv
        branch_lift = nn.Conv2d(branch_input.shape[1], self.width, kernel_size=1, bias=False).to(branch_input.device)
        branch_lifted = branch_lift(branch_input)  # [B, width, H, W]

        branch_processed = self.adaptive_pool_B(branch_lifted)  # Downsample: [B, width, 80, 160]
        branch_processed = F.tanh(self.unet_B(branch_processed))  # U-Net: [B, width, 80, 160]

        # Upsample back to original size
        upsample_B = nn.Upsample(size=original_size, mode='bilinear', align_corners=True)
        branch_processed = upsample_B(branch_processed)  # [B, width, H, W]
        branch_processed = branch_processed.permute(0, 2, 3, 1)  # [B, H, W, width]

        # === T-Branch Network - process solution history ===
        # Input: [B, L*C_sol, H, W] → Output: [B, H, W, width]
        # Lift L*C_sol channels → width channels using 1x1 conv
        tbranch_lift = nn.Conv2d(tbranch_input.shape[1], self.width, kernel_size=1, bias=False).to(tbranch_input.device)
        tbranch_lifted = tbranch_lift(tbranch_input)  # [B, width, H, W]

        tbranch_processed = self.adaptive_pool_T(tbranch_lifted)  # Downsample: [B, width, 80, 160]
        tbranch_processed = self.unet_T(tbranch_processed)  # U-Net: [B, width, 80, 160]

        # Upsample back to original size
        upsample_T = nn.Upsample(size=original_size, mode='bilinear', align_corners=True)
        tbranch_processed = upsample_T(tbranch_processed)  # [B, width, H, W]
        tbranch_processed = tbranch_processed.permute(0, 2, 3, 1)  # [B, H, W, width]

        # === Hadamard Product Combination ===
        # All inputs now: [B, H, W, width]
        combined_output = tbranch_processed * trunk_processed * branch_processed  # [B, H, W, width]

        # === Decoder ===
        combined_output = F.relu(self.fc_2(combined_output))  # [B, H, W, width]
        combined_output = F.relu(self.fc_3(combined_output))  # [B, H, W, width]
        final_output = self.fc_4(combined_output)  # [B, H, W, K]

        return final_output  # [B, H, W, K]


class TNOModel(nn.Module):
    """
    TNO Model wrapper for Gen Stabilised integration
    
    This class adapts TNO for the Gen Stabilised framework by:
    1. Converting from [B, T, C, H, W] format to TNO's expected inputs
    2. Handling different dataset types (Inc, Tra, Iso)
    3. Providing consistent interface with other models
    """
    
    def __init__(self, width=360, L=1, K=1, dataset_type="inc", target_size=(80, 160), teacher_forcing_epochs=500):
        super(TNOModel, self).__init__()
        
        self.width = width
        self.L = L
        self.K = K
        self.dataset_type = dataset_type.lower()
        self.target_size = target_size
        
        # Training phase management - Phase 1.1 Enhancement
        self.teacher_forcing_epochs = teacher_forcing_epochs
        self.current_epoch = 0
        self.training_phase = "teacher_forcing" if L > 1 else "fine_tuning"
        
        # Dataset-specific field mappings - Corrected for actual data structure
        self.field_mappings = {
            "inc": {
                "branch_fields": ["reynolds"],  # Reynolds parameter
                "tbranch_fields": ["velocity_x", "velocity_y"],  # Velocity history
                "target_fields": ["velocity_x", "velocity_y"],  # Predict velocity
                "total_channels": 3,  # vx, vy, reynolds
                "solution_channels": 2,  # vx, vy
                "param_channels": 1   # reynolds
            },
            "tra": {
                "branch_fields": ["mach"],  # Mach parameter only
                "tbranch_fields": ["velocity_x", "velocity_y", "density", "pressure"],  # All solution fields
                "target_fields": ["velocity_x", "velocity_y", "density", "pressure"],  # Predict all fields
                "total_channels": 5,  # vx, vy, density, pressure, mach
                "solution_channels": 4,  # vx, vy, density, pressure
                "param_channels": 1   # mach (no reynolds in TRA data)
            },
            "iso": {
                "branch_fields": ["energy"],  # Turbulent kinetic energy
                "tbranch_fields": ["velocity_x", "velocity_y", "velocity_z"],  # 3D velocity
                "target_fields": ["velocity_x", "velocity_y", "velocity_z"],
                "total_channels": 4,  # vx, vy, vz, z_slice
                "solution_channels": 3,  # vx, vy, vz
                "param_channels": 1   # z_slice
            }
        }
        
        # Validate dataset type
        if self.dataset_type not in self.field_mappings:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
        self.field_config = self.field_mappings[self.dataset_type]
        
        # Core TNO
        self.tno = TNO(width, L, K, target_size)

    def _extract_branch_input(self, data):
        """
        Extract auxiliary function at SINGLE timepoint (t=0) - TNO Corrected

        Branch processes the auxiliary input function v(t₀,·) at a single time.
        NO temporal dimension - returns [B, 1, H, W] for channel-first Conv2d processing.

        Args:
            data: Input tensor [B, T, C, H, W]

        Returns:
            branch_input: Tensor [B, 1, H, W] containing auxiliary field at t=0
        """
        B, T, C, H, W = data.shape

        if self.dataset_type == "inc":
            # Reynolds parameter from last channel at t=0
            if C >= self.field_config["total_channels"]:
                branch_input = data[:, 0, -1:, :, :]  # [B, 1, H, W]
            else:
                # Create synthetic obstacle mask if Reynolds not available
                branch_input = torch.zeros(B, 1, H, W, device=data.device)
                center_h, center_w = H // 2, W // 2
                obstacle_size = min(H, W) // 8
                branch_input[:, :,
                           center_h - obstacle_size//2:center_h + obstacle_size//2,
                           center_w - obstacle_size//2:center_w + obstacle_size//2] = 1.0

        elif self.dataset_type == "tra":
            # Mach parameter from last channel at t=0 (NO Reynolds averaging)
            if C >= self.field_config["total_channels"]:
                branch_input = data[:, 0, -1:, :, :]  # [B, 1, H, W] - Mach only
            else:
                branch_input = torch.ones(B, 1, H, W, device=data.device) * 0.5

        elif self.dataset_type == "iso":
            # Turbulent kinetic energy from velocity components at t=0
            if C >= 3:
                velocity = data[:, 0, 0:3, :, :]  # [B, 3, H, W]
                tke = 0.5 * (velocity ** 2).sum(dim=1, keepdim=True)  # [B, 1, H, W]
                branch_input = tke
            else:
                branch_input = torch.randn(B, 1, H, W, device=data.device) * 0.1
        else:
            # Fallback: use first parameter channel at t=0
            param_start_idx = self.field_config["solution_channels"]
            if C > param_start_idx:
                branch_input = data[:, 0, param_start_idx:param_start_idx+1, :, :]  # [B, 1, H, W]
            else:
                branch_input = torch.zeros(B, 1, H, W, device=data.device)

        return branch_input  # [B, 1, H, W]

    def _generate_coordinate_grids(self, data, output_time_idx=None):
        """
        Generate (t,x,y) coordinate grids for trunk network - TNO Corrected

        Trunk processes coordinate inputs (t,x,y) for the output prediction time.
        Returns [B, H, W, 3] for processing through the trunk MLP.

        Args:
            data: [B, T, C, H, W] tensor
            output_time_idx: Optional time index for prediction (default: L for next step)

        Returns:
            trunk_input: [B, H, W, 3] coordinate tensor with (t, x, y) at each spatial location
        """
        B, T, C, H, W = data.shape

        # Generate normalized spatial coordinate grids [0, 1]
        x = torch.linspace(0, 1, W, device=data.device)
        y = torch.linspace(0, 1, H, device=data.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # [H, W]

        # Time coordinate (normalized) for prediction step
        # Default: predict at t = L (next step after history)
        if output_time_idx is None:
            t_norm = float(self.L) / max(T, self.L + 1)
        else:
            t_norm = float(output_time_idx) / T

        t_coord = torch.full_like(grid_x, t_norm)  # [H, W]

        # Stack as [H, W, 3] for (t, x, y)
        coords = torch.stack([t_coord, grid_x, grid_y], dim=-1)  # [H, W, 3]

        # Expand for batch dimension: [B, H, W, 3]
        trunk_input = coords.unsqueeze(0).expand(B, -1, -1, -1)

        return trunk_input  # [B, H, W, 3]

    def _extract_tbranch_input(self, data):
        """
        Extract L timesteps of ALL solution fields - TNO Corrected

        T-Branch processes the solution history U_hist over L timesteps.
        Returns [B, L*C_sol, H, W] by flattening temporal and channel dimensions.
        NO AVERAGING - preserves all coupled solution variables independently.

        Args:
            data: Input tensor [B, T, C, H, W]

        Returns:
            tbranch_input: Tensor [B, L*C_sol, H, W] with flattened solution history
        """
        B, T, C, H, W = data.shape

        if self.dataset_type == "inc":
            # Channels 0-1: vx, vy (all velocity components, no magnitude)
            if C >= 2:
                solution_fields = data[:, :self.L, 0:2, :, :]  # [B, L, 2, H, W]
            else:
                solution_fields = data[:, :self.L, 0:1, :, :]  # [B, L, 1, H, W]
            # Flatten L and C_sol: [B, L*2, H, W] or [B, L*1, H, W]
            L_actual, C_sol = solution_fields.shape[1], solution_fields.shape[2]
            tbranch_input = solution_fields.reshape(B, L_actual * C_sol, H, W)

        elif self.dataset_type == "tra":
            # Channels 0-3: vx, vy, density, pressure (ALL coupled solution fields)
            if C >= 4:
                solution_fields = data[:, :self.L, 0:4, :, :]  # [B, L, 4, H, W]
            else:
                # Fallback if data incomplete
                available_channels = min(C, 4)
                solution_fields = data[:, :self.L, 0:available_channels, :, :]
            # Flatten L and C_sol: [B, L*4, H, W]
            L_actual, C_sol = solution_fields.shape[1], solution_fields.shape[2]
            tbranch_input = solution_fields.reshape(B, L_actual * C_sol, H, W)

        elif self.dataset_type == "iso":
            # Channels 0-2: vx, vy, vz (all 3D velocity components)
            if C >= 3:
                solution_fields = data[:, :self.L, 0:3, :, :]  # [B, L, 3, H, W]
            else:
                solution_fields = data[:, :self.L, 0:min(C, 3), :, :]
            # Flatten L and C_sol: [B, L*3, H, W]
            L_actual, C_sol = solution_fields.shape[1], solution_fields.shape[2]
            tbranch_input = solution_fields.reshape(B, L_actual * C_sol, H, W)
        else:
            # Fallback: use all solution channels
            solution_channels = self.field_config["solution_channels"]
            solution_fields = data[:, :self.L, 0:solution_channels, :, :]
            L_actual, C_sol = solution_fields.shape[1], solution_fields.shape[2]
            tbranch_input = solution_fields.reshape(B, L_actual * C_sol, H, W)

        return tbranch_input  # [B, L*C_sol, H, W]

    def forward(self, data):
        """
        Forward pass through TNO model - Corrected Architecture

        Args:
            data: [B, T, C, H, W] tensor in Gen Stabilised format

        Returns:
            output: [B, K, target_channels, H, W] predictions
        """
        B, T, C, H, W = data.shape

        # Ensure we have enough timesteps for the current L
        if T < self.L:
            raise ValueError(f"Input sequence length {T} < required history length {self.L}")

        # Extract inputs for TNO's three networks (corrected shapes)
        branch_input = self._extract_branch_input(data)         # [B, 1, H, W]
        trunk_input = self._generate_coordinate_grids(data)     # [B, H, W, 3]
        tbranch_input = self._extract_tbranch_input(data)       # [B, L*C_sol, H, W]

        # Forward through TNO core
        tno_output = self.tno(branch_input, trunk_input, tbranch_input)  # [B, H, W, K]

        # Convert to Gen Stabilised format
        output = tno_output.permute(0, 3, 1, 2)  # [B, K, H, W]

        # Determine number of target channels based on dataset
        num_target_fields = len(self.field_config["target_fields"])

        if num_target_fields > 1:
            # Multi-channel prediction: expand TNO output to all target fields
            # Note: TNO predicts K timesteps, we expand each to include all solution channels
            output = output.unsqueeze(2).expand(-1, -1, num_target_fields, -1, -1)  # [B, K, target_channels, H, W]
        else:
            # Single-channel prediction
            output = output.unsqueeze(2)  # [B, K, 1, H, W]

        return output

    def set_training_phase(self, phase):
        """
        Set training phase (teacher_forcing or fine_tuning)

        IMPORTANT: L (input history size) stays CONSTANT across both phases!
        Only the SOURCE of inputs changes:
        - Teacher forcing: training loop provides ground truth history
        - Fine-tuning: training loop provides model's own predictions

        This matches the reference TNO implementations where L and K are fixed constants.

        Args:
            phase: "teacher_forcing" or "fine_tuning"
        """
        if phase not in ["teacher_forcing", "fine_tuning"]:
            raise ValueError(f"Invalid phase: {phase}. Must be 'teacher_forcing' or 'fine_tuning'")

        self.training_phase = phase

        # L stays constant - no change needed!
        # The phase transition is handled by how the training loop feeds data,
        # not by changing the model's input size
        
    def update_epoch(self, epoch):
        """
        Update current epoch and automatically transition phases if needed - Phase 1.1 Enhancement
        
        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch
        
        # Automatic phase transition
        if epoch >= self.teacher_forcing_epochs and self.training_phase == "teacher_forcing":
            print(f"[TNO] Epoch {epoch}: Transitioning from teacher forcing to fine-tuning")
            self.set_training_phase("fine_tuning")
            
    def get_status(self):
        """
        Get current TNO configuration and status - Phase 1.1 Enhancement
        
        Returns:
            Dictionary with current TNO parameters and status
        """
        return {
            'width': self.width,
            'L': self.L,
            'K': self.K,
            'training_phase': self.training_phase,
            'current_epoch': self.current_epoch,
            'dataset_type': self.dataset_type,
            'teacher_forcing_epochs': self.teacher_forcing_epochs,
            'target_size': self.target_size,
            'field_config': self.field_config
        }
    
    def count_params(self):
        """Count total parameters"""
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c
        
    def get_info(self):
        """Get model information for trainer logging (backward compatibility)"""
        # Use new get_status method but format for backward compatibility
        status = self.get_status()
        
        # Determine training phase description based on architecture configuration
        phase_map = {
            1: "Phase 0 (Basic Integration)",
            2: "Phase 1 (Teacher Forcing)", 
            3: "Phase 2 (Extended History)",
            4: "Phase 3 (Full History)"
        }
        
        training_phase_desc = phase_map.get(self.L, f"Custom (L={self.L})")
        
        return {
            'training_phase': f"{training_phase_desc} - {status['training_phase']}",
            'L': status['L'],
            'K': status['K'],
            'width': status['width'],
            'dataset_type': status['dataset_type'],
            'current_epoch': status['current_epoch'],
            'target_size': status['target_size']
        }


# Alias for compatibility
Net2d = TNOModel