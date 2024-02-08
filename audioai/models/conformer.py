import flax.linen as nn
from flax import struct
from typing import Dict


@struct.dataclass
class ConformerConfig:
    out_dim: int
    emb_dim: int
    mlp_dim: int
    qkv_dim: int
    conv_dim: int
    num_heads: int
    num_layers: int
    kernel_size: int
    stride: int
    attention_bias: bool
    conv_bias: bool
    expansion_factor: int
    dropout_rate: float
    deterministic: bool
    
    def from_dict(self, config: Dict):
        return ConformerConfig(
            out_dim=config['out_dim'],
            mlp_dim=config['mlp_dim'],
            qkv_dim=config['qkv_dim'],
            conv_dim=config['conv_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            kernel_size=config['kernel_size'],
            stride=config['stride'],
            attention_bias=config['attention_bias'],
            conv_bias=config['conv_bias'],
            expansion_factor=config['expansion_factor'],
            dropout_rate=config['dropout_rate'],
            deterministic=config['deterministic']
        )
    
class Conv2dSubsampling(nn.Module):
    config: ConformerConfig
    
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.config.emb_dim,
            kernel_size=self.config.kernel_size,
            strides=self.config.stride,
            name='conv1'
        )
        x = nn.gelu(x)
        x = nn.Conv(
            features=self.config.emb_dim,
            kernel_size=self.config.kernel_size,
            strides=self.config.stride,
            name='conv2'
        )
        return x
    
class PointwiseConv1d(nn.Module):
    config: ConformerConfig
    
    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]
        return nn.Conv(
            features=input_dim * self.config.expansion_factor,
            kernel_size=1,
            strides=1,
        )

class DepthwiseConv1d(nn.Module):
    config: ConformerConfig
    
    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]
        return nn.Conv(
            features=input_dim,
            kernel_size=self.config.kernel_size,
            strides=self.config.stride,
            groups=input_dim,
            padding='VALID'
        )
    
class ConvModule(nn.Module):
    config: ConformerConfig
    
    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm(bias_init=nn.initializers.ones)(x)
        x = PointwiseConv1d(self.config, name="pointwise_conv_1")(x)
        x = nn.glu(x)
        x = DepthwiseConv1d(self.config, name="depthwise_conv")(x)
        x = nn.LayerNorm(bias_init=nn.initializers.ones)(x)
        x = nn.swish(x)
        x = PointwiseConv1d(self.config, name="pointwise_conv_2")(x)
        x = nn.Dropout(rate=self.config.dropout_rate)(x)
        return x
    
class Attention(nn.Module):
    config: ConformerConfig
    
    ...
    
class FeedForward(nn.Module):
    ...
        
        
class ConformerEncoder(nn.Module):
    ...
    
class Conformer(nn.Module):
    ...