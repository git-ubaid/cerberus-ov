# from models.sam_enconder import ImageEncoder
# from segment_anything.build_sam import ImageEncoderViT
from models.sam_enconder import ImageEncoder
from typing import Optional, Tuple, Type
from functools import partial
import torch

def build_sam_vit_b(pretrained=False, progress=True, checkpoint=None):
    print("INSIDE build_sam_vit_b")
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )
def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 448
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    img_enc_vit = ImageEncoder(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
    return img_enc_vit
