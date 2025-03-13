import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class SigVisEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_dim
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embed = nn.Conv2d(config.num_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, padding=0)
        self.num_patches = (self.image_size // self.patch_size) ** 2  # [B, D, H', W'] with H'=W'=image_size/patch_size
        self.num_positions = self.num_patches
        self.pos_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def interpolate_positional_encodiing(self, embeddings, height, width):
        num_patches = embeddings.shape[1]
        num_positions = self.pos_embedding.weight.shape[0]
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.pos_embedding(self.position_ids)
        patch_pos_embedding = self.pos_embedding.weight.unsqueeze(0)
        dimension = embeddings.shape[-1]
        h_mod = height // self.patch_size
        w_mod = width // self.patch_size
        num_position_mod = int(num_positions ** 0.5)
        # [1, num_positions, D] -> [1, sqrt(N), sqrt(N), D]
        patch_pos_embed = patch_pos_embedding.reshape(1, num_position_mod, num_position_mod, dimension)
        # [1, sqrt(N), sqrt(N), D] -> [1, D, sqrt(N), sqrt(N)]
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(h_mod, w_mod), mode="bicubic", align_corners=False)
        # [1, D, H', W'] -> [1, H'*W', D]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dimension)
        return patch_pos_embed

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        patch_embeds = self.patch_embed(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_positional_encodiing(embeddings, pixel_values.shape[2], pixel_values.shape[3])
        else:
            embeddings = embeddings + self.pos_embedding(self.position_ids)
        return embeddings

@dataclass
class SigTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_dim
        self.token_embedding = nn.Embedding(config.vocab_size, self.embed_dim)
        self.max_position_embeddings = config.max_position_embeddings
        self.pos_embedding = nn.Embedding(self.max_position_embeddings, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.max_position_embeddings).unsqueeze(0))

    def forward(self, input_ids):
        token_embeds = self.token_embedding(input_ids)
        seq_length = input_ids.shape[1]
        pos_ids = self.position_ids[:, :seq_length]
        pos_embeds = self.pos_embedding(pos_ids)
        return token_embeds + pos_embeds

@dataclass
class SiglipModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vis_embeddings = SigVisEmbeddings(config)
        self.text_embeddings = SigTextEmbeddings(config)
        self.vision_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.text_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.logit_scale = nn.Parameter(torch.ones([]))

    def forward(self, pixel_values, input_ids, interpolate_pos_encoding=False):
        vis_embeds = self.vis_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        vis_feat = self.vision_proj(vis_embeds.mean(dim=1))
        text_embeds = self.text_embeddings(input_ids)
        text_feat = self.text_proj(text_embeds.mean(dim=1))
        vis_norm = F.normalize(vis_feat, p=2, dim=-1)
        text_norm = F.normalize(text_feat, p=2, dim=-1)
        logits = self.logit_scale.exp() * torch.matmul(vis_norm, text_norm.t())
        return logits

# Dummy configuration for testing.
class DummyConfig:
    def __init__(self):
        self.hidden_dim = 64
        self.image_size = 224
        self.patch_size = 16
        self.num_channels = 3
        self.vocab_size = 1000
        self.max_position_embeddings = 32

config = DummyConfig()
model = SiglipModel(config)

dummy_pixels = torch.randn(2, config.num_channels, config.image_size, config.image_size)
dummy_input_ids = torch.randint(0, config.vocab_size, (2, 16))

logits = model(dummy_pixels, dummy_input_ids, interpolate_pos_encoding=True)
print("Logits shape:", logits.shape)
print("Logits:", logits)
