from __future__ import annotations

import torch
import torch.nn as nn
import timm


class TabularMLP(nn.Module):
    """Small MLP to encode 96 tabular features → 128-dim embedding."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetadataEncoder(nn.Module):
    """Encode body-part and descriptor groups as semantic metadata tokens."""

    def __init__(self, tabular_dim: int, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        body_dim = min(49, tabular_dim)
        descriptor_dim = max(0, tabular_dim - body_dim)
        self.body = nn.Sequential(nn.Linear(body_dim, 64), nn.GELU(), nn.Linear(64, d_model))
        self.descriptor = nn.Sequential(
            nn.Linear(max(1, descriptor_dim), 64), nn.GELU(), nn.Linear(64, d_model)
        )
        self.field_type = nn.Parameter(torch.zeros(2, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tabular: torch.Tensor) -> torch.Tensor:
        body = tabular[:, :49]
        descriptor = tabular[:, 49:]
        if descriptor.shape[1] == 0:
            descriptor = torch.zeros_like(body[:, :1])
        tokens = torch.stack((self.body(body), self.descriptor(descriptor)), dim=1)
        return self.dropout(self.norm(tokens + self.field_type.unsqueeze(0)))


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.image_norm = nn.LayerNorm(d_model)
        self.metadata_norm = nn.LayerNorm(d_model)
        self.image_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.metadata_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.image_ffn = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.metadata_ffn = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))

    def forward(self, image_tokens: torch.Tensor, metadata_tokens: torch.Tensor):
        image_query = self.image_norm(image_tokens)
        metadata_query = self.metadata_norm(metadata_tokens)
        image_update, _ = self.image_attention(image_query, metadata_query, metadata_query)
        metadata_update, _ = self.metadata_attention(metadata_query, image_query, image_query)
        image_tokens = image_tokens + image_update
        metadata_tokens = metadata_tokens + metadata_update
        return image_tokens + self.image_ffn(image_tokens), metadata_tokens + self.metadata_ffn(metadata_tokens)


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pool_attention = nn.MultiheadAttention(d_model, 1, batch_first=True)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, image_tokens: torch.Tensor, metadata_tokens: torch.Tensor):
        for layer in self.layers:
            image_tokens, metadata_tokens = layer(image_tokens, metadata_tokens)
        query = self.query.expand(image_tokens.shape[0], -1, -1)
        pooled_image, _ = self.pool_attention(query, image_tokens, image_tokens)
        return torch.cat((pooled_image.squeeze(1), metadata_tokens.mean(dim=1)), dim=1)


class HierarchicalHeads(nn.Module):
    def __init__(self, fused_dim: int, num_main: int, num_subclass: int, num_disease: int, dropout: float):
        super().__init__()
        self.mainclass = nn.Sequential(nn.Linear(fused_dim, 128), nn.ReLU(), nn.Linear(128, num_main))
        self.subclass = nn.Sequential(nn.Linear(fused_dim + num_main, 128), nn.ReLU(), nn.Linear(128, num_subclass))
        self.disease = nn.Sequential(nn.Linear(fused_dim + num_main + num_subclass, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_disease))

    def forward(self, fused: torch.Tensor):
        main_logits = self.mainclass(fused)
        subclass_logits = self.subclass(torch.cat((fused, main_logits.detach()), dim=1))
        disease_logits = self.disease(torch.cat((fused, main_logits.detach(), subclass_logits.detach()), dim=1))
        return {"mainclass": main_logits, "subclass": subclass_logits, "disease": disease_logits}


class DermaSenseModel(nn.Module):
    """
    Multimodal model:
      Image branch  : EfficientNet-B3 (pretrained) → 1536-dim
      Tabular branch: TabularMLP (96 → 128-dim)
      Fusion        : Concat (1536+128) → FC → num_classes
    """
    def __init__(
        self,
        num_classes: int,
        tabular_dim: int = 96,
        fusion_dim: int = 256,
        dropout: float = 0.3,
        pretrained: bool = True,
        num_main_classes: int = 8,
        num_subclass_classes: int = 19,
        fusion_mode: str = "concat",
        hierarchical: bool = False,
        use_metadata: bool = True,
        d_model: int = 256,
        n_attn_layers: int = 2,
        n_heads: int = 8,
    ):
        super().__init__()

        # ── Image branch ──────────────────────────────────────────────────────
        if fusion_mode not in {"concat", "cross_attention"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.hierarchical = hierarchical
        self.use_metadata = use_metadata
        self.backbone = timm.create_model("efficientnet_b3", pretrained=pretrained, features_only=True, out_indices=[-1])
        img_feat_dim = self.backbone.feature_info.channels()[-1]
        self.image_projection = nn.Conv2d(img_feat_dim, d_model, kernel_size=1)

        # ── Tabular branch ────────────────────────────────────────────────────
        tab_out_dim = 128
        self.tab_mlp = TabularMLP(tabular_dim, tab_out_dim, dropout)
        self.metadata_encoder = MetadataEncoder(tabular_dim, d_model, 0.1)
        self.cross_attention = CrossAttentionFusion(d_model, n_heads, n_attn_layers, 0.1)
        self.concat_projection = nn.Linear(img_feat_dim + tab_out_dim, fusion_dim)
        representation_dim = d_model * 2 if fusion_mode == "cross_attention" else fusion_dim

        # ── Fusion head ───────────────────────────────────────────────────────
        fused_dim = img_feat_dim + tab_out_dim       # 1536 + 128 = 1664
        self.fusion = nn.Sequential(
            nn.Linear(representation_dim, representation_dim),
            nn.LayerNorm(representation_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(representation_dim, num_classes)
        self.hierarchical_heads = HierarchicalHeads(representation_dim, num_main_classes, num_subclass_classes, num_classes, dropout)

    def forward(
        self,
        images: torch.Tensor,      # (B, 3, 300, 300)
        tabular: torch.Tensor | None = None,
    ):
        feature_map = self.backbone(images)[-1]
        if self.fusion_mode == "cross_attention" and self.use_metadata:
            image_tokens = self.image_projection(feature_map).flatten(2).transpose(1, 2)
            metadata_tokens = self.metadata_encoder(tabular)
            fused = self.cross_attention(image_tokens, metadata_tokens)
            representation = self.fusion(fused)
            outputs = self.hierarchical_heads(representation) if self.hierarchical else {"disease": self.classifier(representation)}
        else:
            img_feat = feature_map.mean(dim=(2, 3))
            if self.use_metadata and tabular is not None:
                fused = self.concat_projection(torch.cat((img_feat, self.tab_mlp(tabular)), dim=1))
            else:
                fused = self.concat_projection(torch.cat((img_feat, torch.zeros(img_feat.shape[0], 128, device=img_feat.device)), dim=1))
            representation = self.fusion(fused)
            outputs = {"disease": self.classifier(representation)}
            if self.hierarchical:
                outputs = self.hierarchical_heads(representation)
        return outputs if self.hierarchical else outputs["disease"]


def build_model(num_classes: int, tabular_dim: int = 96,
                fusion_dim: int = 256, dropout: float = 0.3,
                pretrained: bool = True, fusion_mode: str = "concat",
                hierarchical: bool = False, use_metadata: bool = True,
                num_main_classes: int = 8, num_subclass_classes: int = 19,
                d_model: int = 256, n_attn_layers: int = 2, n_heads: int = 8) -> DermaSenseModel:
    return DermaSenseModel(
        num_classes=num_classes,
        tabular_dim=tabular_dim,
        fusion_dim=fusion_dim,
        dropout=dropout,
        pretrained=pretrained,
        fusion_mode=fusion_mode,
        hierarchical=hierarchical,
        use_metadata=use_metadata,
        num_main_classes=num_main_classes,
        num_subclass_classes=num_subclass_classes,
        d_model=d_model,
        n_attn_layers=n_attn_layers,
        n_heads=n_heads,
    )