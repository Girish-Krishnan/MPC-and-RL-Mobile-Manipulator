
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.attn = None

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    # @get_local('attn')
    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 1, dim_head = 64):
        super().__init__()
        image_height, image_width = image_size
        patch_height, patch_width = pair(patch_size)
        self.layer_norm = nn.LayerNorm(dim)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

    
    def forward(self, img):
        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)
        x = self.layer_norm(x)
        return x


class ViTFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(ViTFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        # Separate network for vector input
        self.vec_network = nn.Sequential(
                nn.Linear(observation_space['vec'].shape[0], 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU()
            )
        # ViT for depth
        self.simple_vit = SimpleViT(image_size=(64, 128), patch_size=8, dim=64, depth=2, heads=4, mlp_dim=256, channels=1, dim_head=64)

        # combined layers
        n_flatten = 64  
        combined_input_dim = 64 + n_flatten
        self.combined_network = nn.Sequential(
            nn.Linear(combined_input_dim, features_dim),
            nn.ReLU()
        )
    def forward(self, observations):
        # Process vector input
        vec_features = self.vec_network(observations['vec'])
        
        # Process depth input
        if len(observations['depth'].shape) == 4:
            depth_features = self.normalize_depth_map(observations['depth'][:, :, :, 0])
            depth_features = depth_features.unsqueeze(1)
        else:   
            depth_features = self.normalize_depth_map(observations['depth'][:, :, :, :, 0])
        

        depth_features = self.simple_vit(depth_features)
    
        # Concatenate the features
        combined_features = torch.cat((vec_features, depth_features), dim=1)

        combined_out = self.combined_network(combined_features)
        return combined_out
    
    def normalize_depth_map(self, depth_map):
        min_val = depth_map.min()
        max_val = depth_map.max()
        return (depth_map - min_val) / (max_val - min_val)
    