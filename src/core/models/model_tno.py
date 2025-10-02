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

        # Trunk (Time Neural Network)
        self.time_nn = TimeNN(in_width=self.L, hidden_width=self.width, output_dim=self.width, num_layers=12)

        # Decoder network
        self.fc_2 = nn.Linear(self.width, self.width)
        self.fc_3 = nn.Linear(self.width, self.width)
        self.fc_4 = nn.Linear(self.width, self.K)

    def forward(self, branch_input, trunk_input, tbranch_input):
        """
        Forward pass through TNO
        
        Args:
            branch_input: Auxiliary functions [B, H, W, L] 
            trunk_input: Coordinate grids [B, H, W, L]
            tbranch_input: Solution history [B, H, W, L]
            
        Returns:
            predictions: [B, H, W, K]
        """
        original_size = trunk_input.shape[1:3]
        
        # Trunk Network - process coordinate grids
        num, grid_x, grid_y = trunk_input.shape[0], trunk_input.shape[1], trunk_input.shape[2]
        trunk_flat = trunk_input.reshape(num, -1, self.L)
        trunk_processed = self.time_nn(trunk_flat)
        trunk_processed = trunk_processed.reshape(num, grid_x, grid_y, self.width)
        
        # Branch Network - process auxiliary functions
        branch_processed = self.fc_B_1(branch_input)
        branch_processed = branch_processed.permute(0, 3, 1, 2)  # [B, C, H, W]
        branch_processed = self.adaptive_pool_B(branch_processed)  
        branch_processed = F.tanh(self.unet_B(branch_processed))
        
        # Upsample back to original size
        upsample_B = nn.Upsample(size=original_size, mode='bilinear', align_corners=True)
        branch_processed = upsample_B(branch_processed)
        branch_processed = branch_processed.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        # T-Branch Network - process solution history
        tbranch_processed = self.fc_T_1(tbranch_input)
        tbranch_processed = tbranch_processed.permute(0, 3, 1, 2)  # [B, C, H, W]
        tbranch_processed = self.adaptive_pool_T(tbranch_processed)  
        tbranch_processed = self.unet_T(tbranch_processed)
        
        # Upsample back to original size
        upsample_T = nn.Upsample(size=original_size, mode='bilinear', align_corners=True)
        tbranch_processed = upsample_T(tbranch_processed)
        tbranch_processed = tbranch_processed.permute(0, 2, 3, 1)  # [B, H, W, C]

        # Combine via Hadamard product
        combined_output = tbranch_processed * trunk_processed * branch_processed 

        # Decoder
        combined_output = self.fc_2(combined_output)
        combined_output = self.fc_3(combined_output)
        final_output = self.fc_4(combined_output)
        
        return final_output


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
        
        # Dataset-specific field mappings - Phase 1.1 Enhanced
        self.field_mappings = {
            "inc": {
                "branch_fields": ["reynolds"],  # Reynolds parameter
                "tbranch_fields": ["velocity_x", "velocity_y"],  # Velocity history
                "target_fields": ["velocity_x", "velocity_y"],  # Predict velocity
                "total_channels": 4,  # vx, vy, pressure, reynolds
                "physics_channels": 3,  # vx, vy, pressure
                "param_channels": 1   # reynolds
            },
            "tra": {
                "branch_fields": ["mach", "reynolds"],  # Mach + Reynolds parameters
                "tbranch_fields": ["pressure"],  # Pressure history
                "target_fields": ["pressure"],  # Predict pressure
                "total_channels": 6,  # vx, vy, density, pressure, reynolds, mach
                "physics_channels": 4,  # vx, vy, density, pressure
                "param_channels": 2   # reynolds, mach
            },
            "iso": {
                "branch_fields": ["energy", "z_index"],  # Turbulent energy + Z-slice
                "tbranch_fields": ["velocity_x", "velocity_y", "velocity_z"],  # 3D velocity
                "target_fields": ["velocity_x", "velocity_y", "velocity_z"],
                "total_channels": 5,  # vx, vy, vz, pressure, z_slice
                "physics_channels": 4,  # vx, vy, vz, pressure
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
        Extract auxiliary input functions based on dataset type - Phase 1.1 Enhanced
        
        Args:
            data: Input tensor [B, T, C, H, W]
            
        Returns:
            branch_input: Tensor [B, H, W, L] containing auxiliary functions
        """
        B, T, C, H, W = data.shape
        
        if self.dataset_type == "inc":
            # For Inc: Extract Reynolds parameter (spatially expanded)
            if C >= self.field_config["total_channels"]:
                reynolds_field = data[:, :self.L, -1:, :, :]  # [B, L, 1, H, W] - last channel
                branch_input = reynolds_field.squeeze(2).permute(0, 2, 3, 1)  # [B, H, W, L]
            else:
                # Create synthetic obstacle mask if Reynolds not available
                branch_input = torch.zeros(B, H, W, self.L, device=data.device)
                center_h, center_w = H // 2, W // 2
                obstacle_size = min(H, W) // 8
                branch_input[:, 
                           center_h - obstacle_size//2:center_h + obstacle_size//2,
                           center_w - obstacle_size//2:center_w + obstacle_size//2, 
                           :] = 1.0
                        
        elif self.dataset_type == "tra":
            # For Tra: Extract Mach and Reynolds parameters
            if C >= self.field_config["total_channels"]:
                param_fields = data[:, :self.L, -2:, :, :]  # [B, L, 2, H, W] - last 2 channels
                # Average parameters across channels
                branch_input = param_fields.mean(dim=2).permute(0, 2, 3, 1)  # [B, H, W, L]
            else:
                branch_input = torch.ones(B, H, W, self.L, device=data.device) * 0.5
                
        elif self.dataset_type == "iso":
            # For Iso: Compute turbulent kinetic energy from velocity components
            if C >= 3:
                velocity_fields = data[:, :self.L, 0:3, :, :]  # [B, L, 3, H, W] - first 3 channels (vx,vy,vz)
                tke = 0.5 * (velocity_fields ** 2).sum(dim=2)  # [B, L, H, W] - Turbulent Kinetic Energy
                branch_input = tke.permute(0, 2, 3, 1)  # [B, H, W, L]
            else:
                branch_input = torch.randn(B, H, W, self.L, device=data.device) * 0.1
        else:
            # Fallback to original behavior
            param_start_idx = self.field_config["physics_channels"]
            params = data[:, :, param_start_idx:, :, :]  # [B, T, param_channels, H, W]
            params = params[:, 0, :, :, :]  # [B, param_channels, H, W]
            params = params.unsqueeze(1).expand(-1, self.L, -1, -1, -1)  # [B, L, param_channels, H, W]
            branch_input = params.mean(dim=2)  # [B, L, H, W]
            branch_input = branch_input.permute(0, 2, 3, 1)
        
        return branch_input

    def _generate_coordinate_grids(self, data):
        """
        Generate coordinate grids for trunk network
        
        Args:
            data: [B, T, C, H, W] tensor
            
        Returns:
            trunk_input: [B, H, W, L] coordinate grids
        """
        B, T, C, H, W = data.shape
        
        # Generate normalized coordinate grids
        x = torch.linspace(0, 1, W, device=data.device)
        y = torch.linspace(0, 1, H, device=data.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        
        # Create coordinate grid [H, W, 2]
        coords = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        
        # Expand for batch and sequence dimensions [B, H, W, L]
        # For simplicity, we'll use spatial coordinates repeated L times
        coords = coords.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 2, 1]
        coords = coords.expand(B, -1, -1, -1, self.L)  # [B, H, W, 2, L]
        
        # Take mean over coordinate dimensions to get [B, H, W, L]
        trunk_input = coords.mean(dim=3)  # [B, H, W, L]
        
        return trunk_input

    def _extract_tbranch_input(self, data):
        """
        Extract solution history based on dataset type - Phase 1.1 Enhanced
        
        Args:
            data: Input tensor [B, T, C, H, W]
            
        Returns:
            tbranch_input: Tensor [B, H, W, L] containing solution history
        """
        B, T, C, H, W = data.shape
        
        if self.dataset_type == "inc":
            # Extract velocity components and compute magnitude
            if C >= 3:
                velocity = data[:, :self.L, 0:2, :, :]  # [B, L, 2, H, W] - vx, vy (channels 0,1)
                vel_mag = torch.sqrt((velocity ** 2).sum(dim=2) + 1e-8)  # [B, L, H, W] - avoid div by 0
                tbranch_input = vel_mag.permute(0, 2, 3, 1)  # [B, H, W, L]
            else:
                tbranch_input = data[:, :self.L, 0, :, :].permute(0, 2, 3, 1)
                
        elif self.dataset_type == "tra":
            # Extract pressure field for transonic flow prediction
            if C >= 4:
                pressure = data[:, :self.L, 3, :, :]  # [B, L, H, W] - pressure (channel 3)
                tbranch_input = pressure.permute(0, 2, 3, 1)  # [B, H, W, L]
            else:
                pressure = data[:, :self.L, 0, :, :]  # [B, L, H, W] - fallback to first channel
                tbranch_input = pressure.permute(0, 2, 3, 1)
            
        elif self.dataset_type == "iso":
            # Extract all velocity components for 3D turbulence
            if C >= 4:
                velocity = data[:, :self.L, 0:3, :, :]  # [B, L, 3, H, W] - vx, vy, vz
                vel_mag = torch.sqrt((velocity ** 2).sum(dim=2) + 1e-8)  # [B, L, H, W]
                tbranch_input = vel_mag.permute(0, 2, 3, 1)  # [B, H, W, L]
            else:
                tbranch_input = data[:, :self.L, 0, :, :].permute(0, 2, 3, 1)
        else:
            # Fallback to original behavior
            physics = data[:, :self.L, :self.field_config["physics_channels"], :, :]  # [B, L, physics_channels, H, W]
            tbranch_input = physics.mean(dim=2)  # [B, L, H, W]
            tbranch_input = tbranch_input.permute(0, 2, 3, 1)
        
        return tbranch_input

    def forward(self, data):
        """
        Forward pass through TNO model with multi-channel support - Phase 1.1 Enhanced
        
        Args:
            data: [B, T, C, H, W] tensor in Gen Stabilised format
            
        Returns:
            output: [B, K, target_channels, H, W] predictions 
        """
        B, T, C, H, W = data.shape
        
        # Ensure we have enough timesteps for the current L
        if T < self.L:
            raise ValueError(f"Input sequence length {T} < required history length {self.L}")
        
        # Extract inputs for TNO's three networks
        branch_input = self._extract_branch_input(data)      # [B, H, W, L]
        trunk_input = self._generate_coordinate_grids(data)  # [B, H, W, L]  
        tbranch_input = self._extract_tbranch_input(data)    # [B, H, W, L]
        
        # Forward through TNO
        tno_output = self.tno(branch_input, trunk_input, tbranch_input)  # [B, H, W, K]
        
        # Convert to Gen Stabilised format and handle multi-channel prediction
        output = tno_output.permute(0, 3, 1, 2)  # [B, K, H, W]
        
        # Determine number of target channels based on dataset
        num_target_fields = len(self.field_config["target_fields"])
        
        if num_target_fields > 1:
            # Multi-channel prediction: expand or split TNO output
            output = output.unsqueeze(2).expand(-1, -1, num_target_fields, -1, -1)  # [B, K, target_channels, H, W]
        else:
            # Single-channel prediction (original behavior)
            output = output.unsqueeze(2)  # [B, K, 1, H, W]
        
        return output

    def set_training_phase(self, phase):
        """
        Set training phase (teacher_forcing or fine_tuning) - Phase 1.1 Enhancement
        
        Args:
            phase: "teacher_forcing" or "fine_tuning"
        """
        if phase not in ["teacher_forcing", "fine_tuning"]:
            raise ValueError(f"Invalid phase: {phase}. Must be 'teacher_forcing' or 'fine_tuning'")
            
        self.training_phase = phase
        
        # Adjust L based on phase
        if phase == "teacher_forcing":
            self.L = 2  # Use 2 previous timesteps
        else:
            self.L = 1  # Use only 1 previous timestep
            
        # Update TNO core with new L
        self.tno.L = self.L
        
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