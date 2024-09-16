from segment_anything.build_sam import ImageEncoderViT
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from segment_anything.modeling.common import LayerNorm2d, MLPBlock
from segment_anything.modeling.image_encoder import Block, PatchEmbed

print(f"\n=> In file 'sam_encoder.py' START")

def get_upsizing_block(scale_factor, in_channels = 256, out_channels = 256):
    layers = []
    
   
    while scale_factor > 1:
        layers.append(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2))
        layers.append(LayerNorm2d(in_channels))  # Adjust to appropriate size
        layers.append(nn.GELU())
        
        # Update channel sizes for next layer
        # in_channels = out_channels
        # out_channels = max(out_channels // 2, 32)  # Reduce output channels until 32
        
        # Halve the scale factor
        scale_factor //= 2
    
    return nn.Sequential(*layers)


class ImageEncoder(ImageEncoderViT):
    

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        self.necks = nn.ModuleList()
        self.resizing = nn.ModuleList()
        max_scale = 4
        for i in range(max_scale):
            resize_block = get_upsizing_block(2**(max_scale-i))
            self.resizing.append(resize_block)
            self.necks.append(nn.Sequential(
                nn.Conv2d(
                    embed_dim,
                    out_chans,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
                nn.Conv2d(
                    out_chans,
                    out_chans,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
            ))
        self.necks.append(nn.Sequential(
                nn.Conv2d(
                    embed_dim,
                    out_chans,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
                nn.Conv2d(
                    out_chans,
                    out_chans,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
            ))

        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)
        

        # self.final_neck = nn.Sequential(
        #     nn.Conv2d(
        #         embed_dim,
        #         out_chans,
        #         kernel_size=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(out_chans),
        #     nn.Conv2d(
        #         out_chans,
        #         out_chans,
        #         kernel_size=3,
        #         padding=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(out_chans),
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intermediate_1 = x = self.patch_embed(x)
        
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx==3:
                intermediate_2 = x
            if idx==6:
                intermediate_3 = x
            if idx==9:
                intermediate_4 = x

        intermediate_5 = x

        layers = [intermediate_1, intermediate_2, intermediate_3, intermediate_4, intermediate_5]
        print("LAYERS:")

        for idx in range(5):
            layers[idx] = self.necks[idx](layers[idx].permute(0, 3, 1, 2))
            print(layers[idx].shape)
            if idx+1 != 5:
                layers[idx] = self.resizing[idx](layers[idx])
            print(layers[idx].shape)
        return layers



class PatchEmbedCustom(PatchEmbed):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("Before Projection:", x.shape)
        x = self.proj(x)
        print("After Projection Shape:", x.shape)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        print("Permute Shape:", x.shape)
        return x
    
# SAM utilises encoder from ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py as 'ImageEncoderViT'
# Arguments passed to ImageEncoderViT class in SAM

# In build_sam.py; when called  for image_encoder called as "image_encoder=ImageEncoderViT()" by passing following params.
# ImageEncoderViT(
#             depth=encoder_depth,
#             embed_dim=encoder_embed_dim,
#             img_size=image_size,
#             mlp_ratio=4,
#             norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
#             num_heads=encoder_num_heads,
#             patch_size=vit_patch_size,
#             qkv_bias=True,
#             use_rel_pos=True,
#             global_attn_indexes=encoder_global_attn_indexes,
#             window_size=14,
#             out_chans=prompt_embed_dim,
#         )


# In image_encoder.py.__init__ def; following arguments are recieved in "class ImageEncoderViT(nn.Module)".

    # """
    # -----------------------Description----------------------------
    # Args:
    #     img_size (int): Input image size.
    #     patch_size (int): Patch size.
    #     in_chans (int): Number of input image channels.
    #     embed_dim (int): Patch embedding dimension.
    #     depth (int): Depth of ViT.
    #     num_heads (int): Number of attention heads in each ViT block.
    #     mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    #     qkv_bias (bool): If True, add a learnable bias to query, key, value.
    #     norm_layer (nn.Module): Normalization layer.
    #     act_layer (nn.Module): Activation layer.
    #     use_abs_pos (bool): If True, use absolute positional embeddings.
    #     use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
    #     rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
    #     window_size (int): Window size for window attention blocks.
    #     global_attn_indexes (list): Indexes for blocks using global attention.
    # """
    # 
    # -----------------------Values----------------------------
        
    # img_size: int = 1024,
    # patch_size: int = 16,
    # in_chans: int = 3,
    # embed_dim: int = 768,
    # depth: int = 12,
    # num_heads: int = 12,
    # mlp_ratio: float = 4.0,
    # out_chans: int = 256,
    # qkv_bias: bool = True,
    # norm_layer: Type[nn.Module] = nn.LayerNorm,
    # act_layer: Type[nn.Module] = nn.GELU,
    # use_abs_pos: bool = True,
    # use_rel_pos: bool = False, / Passed True 
    # rel_pos_zero_init: bool = True,
    # window_size: int = 0, / Passed 14
    # global_attn_indexes: Tuple[int, ...] = (),

    #     Above args are used by calling _build_SAM if 3 diff settings as follows:
    # def build_sam_vit_h(checkpoint=None):
    #     return _build_sam(
    #         encoder_embed_dim=1280,
    #         encoder_depth=32,
    #         encoder_num_heads=16,
    #         encoder_global_attn_indexes=[7, 15, 23, 31],
    #         checkpoint=checkpoint,
    #     )


    # build_sam = build_sam_vit_h


    # def build_sam_vit_l(checkpoint=None):
    #     return _build_sam(
    #         encoder_embed_dim=1024,
    #         encoder_depth=24,
    #         encoder_num_heads=16,
    #         encoder_global_attn_indexes=[5, 11, 17, 23],
    #         checkpoint=checkpoint,
    #     )


    # def build_sam_vit_b(checkpoint=None):
    #     return _build_sam(
    #         encoder_embed_dim=768,
    #         encoder_depth=12,
    #         encoder_num_heads=12,
    #         encoder_global_attn_indexes=[2, 5, 8, 11],
    #         checkpoint=checkpoint,
    #     )

    # Following is actual call for building SAM (image encoder only):
    # def _build_sam(encoder_embed_dim,encoder_depth,encoder_num_heads,encoder_global_attn_indexes,checkpoint=None,):
    # prompt_embed_dim = 256
    # image_size = 1024
    # vit_patch_size = 16
    # image_embedding_size = image_size // vit_patch_size
    # sam = Sam(
    #     image_encoder=ImageEncoderViT(
    #         depth=encoder_depth,
    #         embed_dim=encoder_embed_dim,
    #         img_size=image_size,
    #         mlp_ratio=4,
    #         norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
    #         num_heads=encoder_num_heads,
    #         patch_size=vit_patch_size,
    #         qkv_bias=True,
    #         use_rel_pos=True,
    #         global_attn_indexes=encoder_global_attn_indexes,
    #         window_size=14,
    #         out_chans=prompt_embed_dim,
    #     ),


# Following are all of the parameters of cerberus
"""
#  Following parameters are given in cerberus initially wjile executing run_infer_tile/wsi.py as 'args'
# -----------------------Description----------------------------
# --version                   Show version.
# --gpu=<id>                  GPU list. [default: 0]
# --model=<path>              Path to saved checkpoint.
# --nr_inference_workers=<n>  Number of workers during inference. [default: 0]
# --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 0]
# --batch_size=<n>            Batch size. [default: 10]
# --input_dir=<path>          Path to input data directory. Assumes the files are not nested within directory.
# --output_dir=<path>         Path to output data directory. Will create automtically if doesn't exist. [default: output/]
# --patch_input_shape=<n>     Shape of input patch to the network- Assume square shape. [default: 448]
# --patch_output_shape=<n>    Shape of network output- Assume square shape. [default: 144]

# ----------------------------Values------------------------------
=# variable: --gpu      value: 0
=# variable: --model    value: /home/i222826/cerberus-ov/resnet34_cerberus/
=# variable: --nr_inference_workers     value: 0
=# variable: --nr_post_proc_workers     value: 0
=# variable: --batch_size       value: 1
=# variable: --input_dir        value: /home/i222826/cerberus-ov/cerb_input_tile/
=# variable: --output_dir       value: /home/i222826/cerberus-ov/cerb_tile_output/
=# variable: --patch_input_shape        value: 448
=# variable: --patch_output_shape       value: 144
=# variable: --help     value: False
=# variable: --version  value: False


# Following is 'run_paramset' dictionary for holding some of the parameters of cerberus
run_paramset = {
      "dataset_kwargs": {
          'class_input_shape': 144, 
          'input_shape': 448, 
          'output_shape': 448, 
          'req_target_code': {
              'Gland-INST': 'IP-ERODED-CONTOUR-11', 
              'Gland-TYPE': 'TP', 
              'Lumen-INST': 'IP-ERODED-CONTOUR-3', 
              'Nuclei-INST': 'IP-ERODED-CONTOUR-3', 
              'Nuclei-TYPE': 'TP', 
              'Patch-Class': 'PC'
              }
          },
      "loader_kwargs": {
          'test': {'batch_size': 24, 'nr_procs': 0}, 
          'train': {'batch_size': 24, 'nr_procs': 0}, 
          'valid': {'batch_size': 24, 'nr_procs': 0}
          },
      "loss_kwargs": {
          'Gland-INST': {'loss': {'ce': 1}, 'weight': 0}, 
          'Gland-TYPE': {'loss': {'ce': 1, 'dice': 1}, 'weight': 1}, 
          'Lumen-INST': {'loss': {'ce': 1}, 'weight': 0}, 
          'Nuclei-INST': {'loss': {'ce': 1}, 'weight': 0}, 
          'Nuclei-TYPE': {'loss': {'ce': 1, 'dice': 1}, 'weight': 0}, 
          'Patch-Class': {'loss': {'ce': 1}, 'weight': 0}
          }, 
      "model_kwargs": {
          'backbone_imagenet_pretrained': False, 
          'decoder_kwargs': {
              'Gland': {'INST': 3}, 
              'Gland#TYPE': {'TYPE': 3}, 
              'Lumen': {'INST': 3}, 
              'Nuclei': {'INST': 3}, 
              'Nuclei#TYPE': {'TYPE': 7}, 
              'Patch-Class': {'OUT': 9}
              }, 
          'encoder_backbone_name': 'resnet34', 
          'fullnet_custom_pretrained': True, 
          'considered_tasks': ['Nuclei', 'Nuclei#TYPE', 'Gland', 'Gland#TYPE', 'Lumen', 'Patch-Class']
          }, 
      "optimizer_kwargs": {'betas': [0.9, 0.999], 'lr': 0.001, 'weight_decay': 0.0}, 
      "seed": 5 

}
"""
print(f"\n=> In file 'sam_encoder.py' END")