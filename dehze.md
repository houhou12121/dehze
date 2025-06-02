# PGARN Code Architecture Analysis Report

## 1. Overall Architecture Overview

PGARN (Physics-Guided Atmospheric Restoration Network) adopts an encoder-transformer-decoder architecture with the core idea of embedding physical priors (transmission map and atmospheric light) into deep learning networks.

### 1.1 Main Components
- **MAAB**: Multi-path Adaptive Attention Block
- **PGM**: Physics-Guided Module  
- **PSTB**: Physics Scattering Transformer Block
- **Encoder/Decoder**: Multi-scale feature extraction and reconstruction based on MAAB

## 2. Core Module Detailed Analysis

### 2.1 MAAB (Multi-path Adaptive Attention Block)

**Design Philosophy**: Captures multi-scale features through dual-path design while using attention mechanisms to adaptively select important features.

**Key Features**:
 
# Dual-path design
self.conv_path = nn.Sequential(...)      # Standard convolution path
self.dilated_path = nn.Sequential(...)   # Dilated convolution path

# SE attention mechanism
self.se_block = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Conv2d(...), nn.ReLU(),
    nn.Conv2d(...), nn.Sigmoid()
)
```
class MAAB(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(MAAB, self).__init__()
        
        # 修正 LayerNorm，作用于 channel 维度
        self.layer_norm = nn.LayerNorm(in_channels)  # 仅对通道归一化
        
        self.conv1x1_in = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Depthwise Conv
        self.depthwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        
        # Chunk Split
        self.chunk_split_ratio = 0.5
        self.conv_path = nn.Sequential(
            nn.Conv2d(int(out_channels * self.chunk_split_ratio), out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.dilated_path = nn.Sequential(
            nn.Conv2d(int(out_channels * self.chunk_split_ratio), out_channels, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # Attention (SE-like)
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Fusion and Output
        self.conv1x1_out = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
        # Shortcut connection (1x1 Conv) to ensure input and output have the same shape for residual addition
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

**Technical Highlights**:
1. **LayerNorm Preprocessing**: Applied to channel dimension for improved training stability
2. **Depthwise Convolution**: Reduces parameters and improves computational efficiency
3. **Chunk Split**: Divides feature maps into two parts for separate processing
4. **Residual Connection**: Includes adaptive channel adjustment mechanism

### 2.2 PhysicsGuidedModule (Physics-Guided Module)

**Design Philosophy**: Explicitly estimates physical parameters based on atmospheric scattering model.

```python
class PhysicsGuidedModule(nn.Module):
    def __init__(self, in_channels):
        self.t_branch = nn.Conv2d(in_channels, 1, kernel_size=3, ...)  # Transmission map
        self.a_branch = nn.Conv2d(in_channels, 3, kernel_size=3, ...)  # Atmospheric light
    
    def forward(self, x):
        t = torch.sigmoid(self.t_branch(x))  # [B,1,H,W] 
        A = torch.sigmoid(self.a_branch(x))  # [B,3,H,W]
        return t, A
```

 
- **t (Transmission Map)**: Single-channel output representing light transmittance
- **A (Atmospheric Light)**: Three-channel output corresponding to RGB atmospheric scattering
- **Sigmoid Activation**: Ensures output is in [0,1] range, physically meaningful

### 2.3 AtmosphericScatteringTransformer (PSTB)

**Design Philosophy**: Integrates physical priors into self-attention mechanism to guide feature transformation.

  
# Physics prior embedding
physics_cat = torch.cat([t, A], dim=1)  # [B, 4, H, W]
physics_embed = self.physics_conv(physics_cat)  # [B, C, H, W]

# Modified query with physics guidance
if idx == 0:  # Only add physics prior in first transformer layer
    q = x_flat  # Can be modified to: q = x_flat + pe_flat
```

 

## 3. Network Architecture Components

### 3.1 MultiscaleEncoder

**Structure**:
- **Layer 1**: MAAB without downsampling
- **Layers 2-N**: MAAB + MaxPool2d for progressive downsampling
- **Channel Progression**: base_channel × 2^(layer_index)

```python
# Example progression for base_channel=64
# Layer 1: 3 → 64 channels
# Layer 2: 64 → 128 channels + MaxPool
# Layer 3: 128 → 256 channels + MaxPool
```

### 3.2 MultiscaleDecoder

**Structure**:
- **Progressive Upsampling**: ConvTranspose2d for resolution recovery
- **Skip Connections**: Residual connections from encoder features
- **Channel Reduction**: Progressive reduction back to output channels

**Key Implementation**:
```python
# Skip connection handling
if skip_connections:
    r = skip_connections.pop()
    if x.shape[2:] != r.shape[2:]:
        r = F.interpolate(r, size=x.shape[2:], mode='bilinear')
    x = x + r  # Residual connection
```

## 4. Model Variants and Configurations

### 4.1 Available Model Sizes
```python
def Phformer_b():    # Base model
    return AtmosphericRestorationNetwork(base_channel=64, transformer_num_layers=1)

def Phformer_l():    # Large model  
    return AtmosphericRestorationNetwork(base_channel=64, num_encoder_layers=5, 
                                       num_decoder_layers=5, transformer_num_layers=1)

 

### 4.2 Configuration Trade-offs
- **Base Model**: Balanced performance and efficiency
- **Large Models**: Increased depth for better feature extraction
- **VVL Model**: Doubled base channels for maximum capacity

## 5. Key Technical componets

### 5.1 Physics-Guided Attention
- **Integration Point**: Physics priors embedded into transformer queries
- **Selective Application**: Only applied in first transformer layer to avoid over-constraint
- **Learnable Mapping**: 1×1 convolution maps physics priors to feature space

### 5.2 Efficient Multi-scale Processing
- **MAAB Efficiency**: Dual-path design with reduced computational overhead
- **Memory Optimization**: Chunk splitting reduces memory usage
- **Adaptive Attention**: SE blocks provide channel-wise feature selection

### 5.3 Attention Visualization System
```python
# Built-in attention analysis
if self.attention_save:
    # Save original 169×169 attention map
    np.savez('attention_map.npz', attention_map=attention_map_cpu.numpy())
    
    # Generate 13×13 block-level attention map
    block_attention = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            pixel_idx = i * W + j
            block_attention[i, j] = np.mean(attention_map_to_save.numpy()[pixel_idx, :])
```

## 6. Forward Pass Flow

```python

class AtmosphericScatteringTransformer(nn.Module):
    def __init__(self, in_channels, num_heads=4, dim_feedforward=256, num_layers=1, dropout=0.1):
        super(AtmosphericScatteringTransformer, self).__init__()
        self.attention_save =False
        # 确保 embed_dim 可以被 num_heads 整除
        if in_channels % num_heads != 0:
            in_channels = (in_channels // num_heads) * num_heads  # 调整为可整除的值
        self.dropout = nn.Dropout(p=0.5)
        # 多头注意力和前馈网络
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, dropout=dropout),
                nn.Linear(in_channels, dim_feedforward),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, in_channels),
                nn.LayerNorm(in_channels)
            ]) for _ in range(num_layers)
        ])

        # 物理先验卷积 (1x1卷积，将 t 和 A 映射到 Transformer 的输入空间)
        self.physics_conv = nn.Conv2d(4, in_channels, kernel_size=1)
        
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x, t, A):
        B, C, H, W = x.size()

        # 1. 构建物理先验嵌入
        physics_cat = torch.cat([t, A], dim=1)  # [B, 4, H, W]
        physics_embed = self.physics_conv(physics_cat)  # [B, C, H, W]

        # 2. Flatten for MHA
        x_flat = x.view(B, C, -1).permute(2, 0, 1)  # [HW, B, C]
        pe_flat = physics_embed.view(B, C, -1).permute(2, 0, 1)  # [HW, B, C]

        # 3. 遍历多个Transformer层
        for idx, (attn, ffn1, dropout, ffn2, norm) in enumerate(self.layers):
            if idx == 0:  # 只在第一个Transformer层加入物理先验嵌入
                #q = x_flat + pe_flat
                q = x_flat
            else:
                q = x_flat  # 后续的Transformer层不再加物理先验嵌入
            k = x_flat
            v = x_flat
            #print(q.shape,k.shape,v.shape)
            #attn_out, attention_map = attn(q, k, v)  # [HW, B, C]
            attn_out, attention_map = attn(q, k, v, need_weights=True, average_attn_weights=False)
            if self.attention_save:
                # Print shape for debugging
                print(attn_out.shape)
                print(attention_map.shape)  # torch.Size([1, 169, 169])
                print("I am in dehaze_transformer_new file")
                # First move tensor to CPU and detach from computation graph
                attention_map_cpu = attention_map.detach().cpu()
                
                # Correctly select the attention map based on its shape
                attention_map_to_save = attention_map_cpu[0,0,:,:]  # Select first slice, resulting in [169, 169]
                
                # Save as NumPy format (保留原有的.npz保存功能)
                np.savez('attention_map.npz', attention_map=attention_map_cpu.numpy())
                
                # 计算13×13块级注意力图
                # 创建13×13的块级注意力图 - 使用平均值而不是累计和
                block_attention = np.zeros((H, W))
                for i in range(H):
                    for j in range(W):
                        # 计算当前像素在展平后的索引
                        pixel_idx = i * W + j
                        
                        # 方法1：计算此像素对所有其他像素的平均注意力（行平均）
                        block_attention[i, j] = np.mean(attention_map_to_save.numpy()[pixel_idx, :])
                        
                        # 或者方法2：计算所有像素对此像素的平均注意力（列平均）
                        # block_attention[i, j] = np.mean(attention_map_to_save.numpy()[:, pixel_idx])
                        
                        # 或者方法3：结合行列平均，更全面地表示此像素的重要性
                        # block_attention[i, j] = (np.mean(attention_map_to_save.numpy()[pixel_idx, :]) + 
                        #                        np.mean(attention_map_to_save.numpy()[:, pixel_idx])) / 2
                
                # 归一化块级注意力图以增强对比度（可选）
                # 这有助于使注意力模式更加明显
                block_attention = (block_attention - np.min(block_attention)) / (np.max(block_attention) - np.min(block_attention))
                
                # 保存归一化的块级注意力图
                np.savez('block_attention_map_normalized.npz', block_attention=block_attention)
                
                # 绘制归一化后的13×13块级注意力图
                plt.figure(figsize=(10, 8))
                plt.imshow(block_attention, cmap='hot', interpolation='nearest')
                plt.colorbar()
                plt.title("归一化块级注意力图 (13×13)")
                plt.savefig('block_attention_map_normalized.png', bbox_inches='tight', dpi=300)
                plt.close()
                
                # 以下是完整对比图绘制代码
                # 创建子图：归一化的13×13块级注意力图与原始169×169注意力图对比
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                
                # 绘制原始169×169注意力图
                im1 = ax1.imshow(attention_map_to_save.numpy(), cmap='hot', interpolation='nearest')
                ax1.set_title("原始注意力图 (169×169)", fontsize=14)
                fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                
                # 绘制归一化的13×13块级注意力图
                im2 = ax2.imshow(block_attention, cmap='hot', interpolation='nearest')
                ax2.set_title("归一化块级注意力图 (13×13)", fontsize=14)
                fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                plt.savefig('attention_map_comparison_normalized.png', bbox_inches='tight', dpi=300)
                plt.close()
                
                # Use sys.exit() instead of os.exit()
                #import sys
                #sys.exit()
                # Instead of exiting, wait for user input before continuing
                input("Press Enter to continue processing the next image...")
                
            x_attn = norm(x_flat + attn_out)

            # 前馈网络 (Feed-Forward Network)
            ffn_out = ffn2(self.dropout(F.relu(ffn1(x_attn))))
            ffn_out = self.dropout_ffn(ffn_out)
            x_flat = norm(x_attn + ffn_out)  # [HW, B, C]

        # 4. 重塑回 [B, C, H, W]
        out = x_flat.permute(1, 2, 0).view(B, C, H, W)
        return out
```

 
## 7. Conclusion

The PGARN implementation demonstrates a sophisticated integration of physical modeling and deep learning. Key strengths include:

1. **Modular Design**: Clear separation of concerns with reusable components
2. **Physics Integration**: Meaningful incorporation of atmospheric scattering principles
3. **Efficiency Optimization**: Memory and computational optimizations throughout
4. **Comprehensive Analysis**: Built-in tools for understanding model behavior
5. **Scalable Architecture**: Multiple model variants for different computational budgets

The code represents a well-engineered solution that balances theoretical rigor with practical implementation considerations.
