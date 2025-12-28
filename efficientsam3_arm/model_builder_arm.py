# filepath: /Users/Pablo.Vargas2/Documents/efficientsam3/efficientsam3_arm/model_builder_arm.py
"""
ARM-compatible model builder for EfficientSAM3.

This module provides a clean interface for building EfficientSAM3 models on ARM devices
(Apple Silicon with MPS or CPU) by importing only inference-critical modules and avoiding
problematic tracker dependencies.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import warnings
from iopath.common.file_io import g_pathmgr


from .model.videt import ViT
from .model.necks import Sam3DualViTDetNeck
from .model.position_encoding import PositionEmbeddingSine
from .model.text_encoder_student import TextStudentEncoder
from .model.vl_combiner import SAM3VLBackbone
from .model.sam3_image import Sam3Image
from .model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from .model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from .model.decoder import TransformerDecoder, TransformerDecoderLayer, TransformerDecoderLayerv2, TransformerEncoderCrossAttention
from .model.maskformer_segmentation import UniversalSegmentationHead, PixelDecoder
from .model.memory import (
    CXBlock,
    SimpleMaskDownSampler,
    SimpleFuser,
    SimpleMaskEncoder,
)
from .model.geometry_encoders import SequenceGeometryEncoder
from .model.sam3_tracking_predictor import Sam3TrackerPredictor
from .sam.transformer import RoPEAttention
from .model.sam1_task_predictor import SAM3InteractiveImagePredictor

# Silence warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils._python_dispatch")

class ImageStudentEncoder(nn.Module):
    def __init__(self, backbone, in_channels, embed_dim, embed_size, img_size):
        super().__init__()
        self.backbone = backbone
        self.embed_size = embed_size
        self.img_size = img_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.head(feats)
        if feats.shape[-1] != self.embed_size or feats.shape[-2] != self.embed_size:
            feats = F.interpolate(
                feats,
                size=(self.embed_size, self.embed_size),
                mode="bilinear",
                align_corners=False,
            )
        return feats
    
def _create_position_encoding(precompute_resolution=None):
    """Create position encoding for visual backbone."""
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )

def _create_vit_backbone(compile_mode=None):
    """Create ViT backbone for visual feature extraction."""
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
    )

def _create_vit_neck(position_encoding, vit_backbone, enable_inst_interactivity=False):
    """Create ViT neck for feature pyramid."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )

def _create_student_vision_backbone(
    backbone_type, model_name, compile_mode=None, enable_inst_interactivity=True
) -> Sam3DualViTDetNeck:
    """Create EfficientSAM3 visual backbone with a student backbone and neck."""
    
    # Position encoding
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    
    # EFFICIENTVIT BACKBONE (commented out for ARM compatibility)
    # if backbone_type == "efficientvit":
    #     from sam3.backbones.efficientvit.efficientvit.backbone import (
    #         efficientvit_backbone_b0,
    #         efficientvit_backbone_b1,
    #         efficientvit_backbone_b2,
    #     )
    #     if model_name == "b0":
    #         backbone = efficientvit_backbone_b0()
    #     elif model_name == "b1":
    #         backbone = efficientvit_backbone_b1()
    #     elif model_name == "b2":
    #         backbone = efficientvit_backbone_b2()
    #     else:
    #         raise ValueError(f"Unknown EfficientViT model: {model_name}")
        
    #     class EfficientViTTrunkWrapper(nn.Module):
    #         def __init__(self, model):
    #             super().__init__()
    #             self.model = model
    #             self.channel_list = [model.width_list[-1]]
            
    #         def forward(self, x):
    #             x = x[0] if isinstance(x, list) else x
    #             out = self.model(x)
    #             return out['stage_final']
        
    #     wrapped_backbone = EfficientViTTrunkWrapper(backbone)
    #     in_channels = wrapped_backbone.channel_list[0]

    if backbone_type == "repvit":
        from .backbones.repvit import (
            repvit_m0_9, repvit_m1_1, repvit_m2_3
        )
        name_map = {
            "m0.9": repvit_m0_9, "m0_9": repvit_m0_9,
            "m1.1": repvit_m1_1, "m1_1": repvit_m1_1,
            "m2.3": repvit_m2_3, "m2_3": repvit_m2_3,
        }
        if model_name not in name_map:
             raise ValueError(f"Unknown RepViT model: {model_name}")
        
        backbone = name_map[model_name](distillation=False, num_classes=0)
        
        class RepViTTrunkWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                # Infer channels
                dummy = torch.zeros(1, 3, 224, 224)
                with torch.no_grad():
                    for f in model.features:
                        dummy = f(dummy)
                self.channel_list = [dummy.shape[1]]

            def forward(self, x):
                for f in self.model.features:
                    x = f(x)
                return x

        wrapped_backbone = RepViTTrunkWrapper(backbone)
        in_channels = wrapped_backbone.channel_list[0]

    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
    
    # Wrap with ImageStudentEncoder to include the projection head
    student_encoder = ImageStudentEncoder(
        backbone=wrapped_backbone,
        in_channels=in_channels,
        embed_dim=1024, # SAM3 expects 1024 channels
        embed_size=72,
        img_size=1008,
    )
    
    # Add channel_list to student_encoder so Sam3DualViTDetNeck can read it
    student_encoder.channel_list = [1024]

    # Wrap student_encoder to return a list as expected by Sam3DualViTDetNeck
    class ListWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.channel_list = model.channel_list
            
        def forward(self, x):
            return [self.model(x)]
            
    final_trunk = ListWrapper(student_encoder)

    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        final_trunk,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    return vit_neck

def _create_student_text_encoder(bpe_path: str, backbone_type: str) -> TextStudentEncoder:
    """Create Student text encoder."""
    
    # Default config values
    cfg = {
        "context_length": 77, # MobileCLIP default
        "vocab_size": 49408,
        "dim": 512,
        "ffn_multiplier_per_layer": 4.0,
        "n_heads_per_layer": 8,
        "n_transformer_layers": 12,
        "norm_layer": "layer_norm_fp32",
        "causal_masking": False,
        "model_name": "base",
        "embed_dropout": 0.0,
        "no_scale_embedding": False,
        "no_pos_embedding": False,
    }

    if backbone_type == "MobileCLIP-S0":
        cfg.update({
            "dim": 512,
            "n_transformer_layers": 4,
            "n_heads_per_layer": 8,
            "model_name": "mct",
        })
    elif backbone_type in ["MobileCLIP-S1", "MobileCLIP2-S0", "MobileCLIP2-S2"]:
        cfg.update({
            "dim": 512,
            "n_transformer_layers": 12,
            "n_heads_per_layer": 8,
            "model_name": "base",
        })
    elif backbone_type == "MobileCLIP-B":
        cfg.update({
            "dim": 512,
            "n_transformer_layers": 12,
            "n_heads_per_layer": 8,
            "model_name": "base",
            "causal_masking": True,
        })
    elif backbone_type in ["MobileCLIP2-S3", "MobileCLIP2-S4", "MobileCLIP2-L"]:
        cfg.update({
            "dim": 768,
            "n_transformer_layers": 12,
            "n_heads_per_layer": 12,
            "model_name": "base", 
        })
    
    return TextStudentEncoder(
        cfg=cfg,
        context_length=32, # Match teacher input length
        output_dim=256, # SAM3 d_model
        bpe_path=bpe_path
    )

def _create_vl_backbone(vit_neck, text_encoder):
    """Create visual-language backbone."""
    return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)

def _create_sam3_model(
    backbone,
    transformer,
    input_geometry_encoder,
    segmentation_head,
    dot_prod_scoring,
    inst_interactive_predictor,
    eval_mode,
):
    """Create the SAM3 image model."""
    common_params = {
        "backbone": backbone,
        "transformer": transformer,
        "input_geometry_encoder": input_geometry_encoder,
        "segmentation_head": segmentation_head,
        "num_feature_levels": 1,
        "o2m_mask_predict": True,
        "dot_prod_scoring": dot_prod_scoring,
        "use_instance_query": False,
        "multimask_output": True,
        "inst_interactive_predictor": inst_interactive_predictor,
    }

    matcher = None
    if not eval_mode:
        from efficientsam3_arm.train.matcher import BinaryHungarianMatcherV2

        matcher = BinaryHungarianMatcherV2(
            focal=True,
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            alpha=0.25,
            gamma=2,
            stable=False,
        )
    common_params["matcher"] = matcher
    model = Sam3Image(**common_params)

    return model

def _create_transformer_encoder() -> TransformerEncoderFusion:
    """Create transformer encoder with its layer."""
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder

def _create_transformer_decoder() -> TransformerDecoder:
    """Create transformer decoder with its layer."""
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )
    return decoder

def _create_dot_product_scoring():
    """Create dot product scoring module."""
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)

def _create_segmentation_head(compile_mode=None):
    """Create segmentation head with pixel decoder."""
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode=compile_mode,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head

def _create_sam3_transformer(has_presence_token: bool = True) -> TransformerWrapper:
    """Create SAM3 transformer encoder and decoder."""
    encoder: TransformerEncoderFusion = _create_transformer_encoder()
    decoder: TransformerDecoder = _create_transformer_decoder()

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)

def _create_geometry_encoder():
    """Create geometry encoder with all its components."""
    # Create position encoding for geometry encoder
    geo_pos_enc = _create_position_encoding()
    # Create CX block for fuser
    cx_block = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    # Create geometry encoder layer
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    # Create geometry encoder
    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder

def _create_tracker_maskmem_backbone():
    """Create the SAM3 Tracker memory encoder."""
    # Position encoding for mask memory backbone
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )

    # Mask processing components
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152]
    )

    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )

    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)

    maskmem_backbone = SimpleMaskEncoder(
        out_dim=64,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )

    return maskmem_backbone

def _create_tracker_transformer():
    """Create the SAM3 Tracker transformer components."""
    # Self attention
    self_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=False,
        use_rope_real=False,
    )

    # Cross attention
    cross_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=False,
        use_rope_real=False,
    )

    # Encoder layer
    encoder_layer = TransformerDecoderLayerv2(
        cross_attention_first=False,
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=self_attention,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )

    # Encoder
    encoder = TransformerEncoderCrossAttention(
        remove_cross_attention_layers=[],
        batch_first=True,
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
    )

    # Transformer wrapper
    transformer = TransformerWrapper(
        encoder=encoder,
        decoder=None,
        d_model=256,
    )

    return transformer

def _create_vision_backbone(
    compile_mode=None, enable_inst_interactivity=True
) -> Sam3DualViTDetNeck:
    """Create SAM3 visual backbone with ViT and neck."""
    # Position encoding
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    # ViT backbone
    vit_backbone: ViT = _create_vit_backbone(compile_mode=compile_mode)
    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    # Visual neck
    return vit_neck

def build_tracker(
    apply_temporal_disambiguation: bool, with_backbone: bool = False, compile_mode=None
) -> Sam3TrackerPredictor:
    """
    Build the SAM3 Tracker module for video tracking.

    Returns:
        Sam3TrackerPredictor: Wrapped SAM3 Tracker module
    """

    # Create model components
    maskmem_backbone = _create_tracker_maskmem_backbone()
    transformer = _create_tracker_transformer()
    backbone = None
    if with_backbone:
        vision_backbone = _create_vision_backbone(compile_mode=compile_mode)
        backbone = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)
    # Create the Tracker module
    model = Sam3TrackerPredictor(
        image_size=1008,
        num_maskmem=7,
        backbone=backbone,
        backbone_stride=14,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        # SAM parameters
        multimask_output_in_sam=True,
        # Evaluation
        forward_backbone_per_frame_for_eval=True,
        trim_past_non_cond_mem_for_eval=False,
        # Multimask
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        # Additional settings
        always_start_from_first_ann_frame=False,
        # Mask overlap
        non_overlap_masks_for_mem_enc=False,
        non_overlap_masks_for_output=False,
        max_cond_frames_in_attn=4,
        offload_output_to_cpu_for_eval=False,
        # SAM decoder settings
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        clear_non_cond_mem_around_input=True,
        fill_hole_area=0,
        use_memory_selection=apply_temporal_disambiguation,
    )

    return model

def _load_checkpoint(model, checkpoint_path, device="cpu"):
    """Load model checkpoint from file."""
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location=device, weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    sam3_image_ckpt = {
        k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
    }
    if model.inst_interactive_predictor is not None:
        sam3_image_ckpt.update(
            {
                k.replace("tracker.", "inst_interactive_predictor.model."): v
                for k, v in ckpt.items()
                if "tracker" in k
            }
        )
    missing_keys, _ = model.load_state_dict(sam3_image_ckpt, strict=False)
    if len(missing_keys) > 0:
        print(
            f"loaded {checkpoint_path} and found "
            f"missing and/or unexpected keys:\n{missing_keys=}"
        )


def _setup_device_and_mode(model, device, eval_mode):
    """Setup model device and evaluation mode."""
    if device == "cuda":
        model = model.cuda()
    elif device == "mps":
        model = model.to("mps")
    elif device == "cpu":
        model = model.cpu()
    
    # Ensure all model parameters and buffers are on the correct device
    for name, param in model.named_parameters():
        if param.device.type != device:
            param.data = param.data.to(device)
    for name, buffer in model.named_buffers():
        if buffer.device.type != device:
            buffer.data = buffer.data.to(device)
    
    # Clear decoder coordinate caches that may contain tensors on wrong device
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'decoder'):
        decoder = model.transformer.decoder
        if hasattr(decoder, 'coord_cache'):
            decoder.coord_cache.clear()
        if hasattr(decoder, 'compilable_cord_cache'):
            decoder.compilable_cord_cache = None
            decoder.compilable_stored_size = None
    
    if eval_mode:
        model.eval()
    return model

def build_efficientsam3_image_model(
    bpe_path=None,
    device=None,
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=False,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
    backbone_type="efficientvit",
    model_name="b0",
    # Legacy argument support
    efficientvit_model=None,
    text_encoder_type=None, # e.g. "MobileCLIP-S0"
):
    """
    Build EfficientSAM3 image model with a student backbone

    Args:
        bpe_path: Path to the BPE tokenizer vocabulary
        device: Device to load the model on ('cuda' or 'cpu')
        eval_mode: Whether to set the model to evaluation mode
        checkpoint_path: Optional path to EfficientSAM3 model checkpoint
        load_from_HF: Whether to load checkpoint from HuggingFace (if available)
        enable_segmentation: Whether to enable segmentation head
        enable_inst_interactivity: Whether to enable instance interactivity (SAM 1 task)
        compile: To enable compilation, set to True
        backbone_type: Type of backbone ('efficientvit', 'repvit', 'tinyvit')
        model_name: Model variant (e.g. 'b0', 'm1.1', '5m')
        efficientvit_model: Deprecated, use backbone_type and model_name instead
        text_encoder_type: Type of text encoder (e.g. 'MobileCLIP-S0'). If None, uses standard SAM3 text encoder.

    Returns:
        An EfficientSAM3 image model
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
    if efficientvit_model is not None:
        backbone_type = "efficientvit"
        model_name = efficientvit_model

    if bpe_path is None:
        # Get absolute path to the assets directory
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(current_file)
        bpe_path = os.path.join(project_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
        
    # Create visual components with student backbone
    compile_mode = "default" if compile else None
    vision_encoder = _create_student_vision_backbone(
        backbone_type=backbone_type,
        model_name=model_name,
        compile_mode=compile_mode,
        enable_inst_interactivity=enable_inst_interactivity,
    )

    # Create text components
    if text_encoder_type:
        text_encoder = _create_student_text_encoder(bpe_path, text_encoder_type)
    else:
        raise ValueError("text_encoder_type must be specified for EfficientSAM3 student model.")
        # Weight of MobileClip are already released, so is better to nor rely on the standard text encoder.
        #text_encoder = _create_text_encoder(bpe_path)

    # Create visual-language backbone
    backbone = _create_vl_backbone(vision_encoder, text_encoder)

    # Create transformer components
    transformer = _create_sam3_transformer()

    # Create dot product scoring
    dot_prod_scoring = _create_dot_product_scoring()

    # Create segmentation head if enabled
    segmentation_head = (
        _create_segmentation_head(compile_mode=compile_mode)
        if enable_segmentation
        else None
    )

    # Create geometry encoder
    input_geometry_encoder = _create_geometry_encoder()
    if enable_inst_interactivity:
        sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
        inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
    else:
        inst_predictor = None
    # Create the SAM3 model
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        inst_predictor,
        eval_mode,
    )
    if load_from_HF and checkpoint_path is None:
        # For EfficientSAM3, you may need to specify a different HuggingFace repo
        # checkpoint_path = download_ckpt_from_hf()  # Update this for EfficientSAM3
        pass
    # Load checkpoint if provided
    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path, device)

    # Setup device and mode
    model = _setup_device_and_mode(model, device, eval_mode)

    return model

def build_sam3_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
    enable_text_encoder=True,
    enable_vision_encoder=True,
):
    """
    Build SAM3 image model

    Args:
        bpe_path: Path to the BPE tokenizer vocabulary
        device: Device to load the model on ('cuda' or 'cpu')
        eval_mode: Whether to set the model to evaluation mode
        checkpoint_path: Optional path to model checkpoint
        enable_segmentation: Whether to enable segmentation head
        enable_inst_interactivity: Whether to enable instance interactivity (SAM 1 task)
        compile_mode: To enable compilation, set to "default"
        enable_text_encoder: Whether to enable text encoder
        enable_vision_encoder: Whether to enable vision encoder

    Returns:
        A SAM3 image model
    """
    if bpe_path is None:
        bpe_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
        )
    # Create visual components
    compile_mode = "default" if compile else None
    if enable_vision_encoder:
        vision_encoder = _create_vision_backbone(
            compile_mode=compile_mode, enable_inst_interactivity=enable_inst_interactivity
        )
    else:
        vision_encoder = None

    # Create text components
    if enable_text_encoder:
        text_encoder = _create_text_encoder(bpe_path)
    else:
        text_encoder = None

    # Create visual-language backbone
    backbone = _create_vl_backbone(vision_encoder, text_encoder)

    # Create transformer components
    transformer = _create_sam3_transformer()

    # Create dot product scoring
    dot_prod_scoring = _create_dot_product_scoring()

    # Create segmentation head if enabled
    segmentation_head = (
        _create_segmentation_head(compile_mode=compile_mode)
        if enable_segmentation
        else None
    )

    # Create geometry encoder
    input_geometry_encoder = _create_geometry_encoder()
    if enable_inst_interactivity:
        sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
        inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
    else:
        inst_predictor = None
    # Create the SAM3 model
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        inst_predictor,
        eval_mode,
    )
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    # Load checkpoint if provided
    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path)

    # Setup device and mode
    model = _setup_device_and_mode(model, device, eval_mode)

    return model

def load_checkpoint_weights(model, checkpoint_path, device="cpu", verbose=False):
    """
    Load checkpoint weights into a model.
    
    This is a utility function for loading weights separately from model creation.
    
    Args:
        model: The model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load weights to
        verbose: Whether to print loading information
    
    Returns:
        tuple: (model, missing_keys, unexpected_keys)
    """
    if verbose:
        print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Handle different checkpoint formats
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Load with non-strict mode to handle partial checkpoints
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if verbose:
        if len(missing_keys) > 0:
            print(f"⚠️  Missing keys: {len(missing_keys)}")
        if len(unexpected_keys) > 0:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
            print("✅ All weights loaded successfully")
    
    return model, missing_keys, unexpected_keys


def get_model_info(model):
    """
    Get information about a model.
    
    Args:
        model: The model to inspect
    
    Returns:
        dict: Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / 1024 / 1024,  # Assuming fp32
        "is_training": model.training,
    }


# Supported configurations
SUPPORTED_BACKBONES = {
    "efficientvit": ["b0", "b1", "b2"],
    "repvit": ["m0_9", "m0.9", "m1_1", "m1.1", "m2_3", "m2.3"],
    "tinyvit": ["5m", "11m", "21m"],
}

SUPPORTED_TEXT_ENCODERS = [
    "MobileCLIP-S0",
    "MobileCLIP-S1",
    "MobileCLIP-S2",
    "MobileCLIP-B",
    "MobileCLIP2-S0",
    "MobileCLIP2-S2",
    "MobileCLIP2-S3",
    "MobileCLIP2-S4",
    "MobileCLIP2-L",
]


def validate_config(backbone_type, model_name, text_encoder_type=None):
    """
    Validate model configuration.
    
    Args:
        backbone_type: Type of vision backbone
        model_name: Specific model variant
        text_encoder_type: Type of text encoder (optional)
    
    Raises:
        ValueError: If configuration is invalid
    """
    if backbone_type not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone type: {backbone_type}. "
            f"Supported: {list(SUPPORTED_BACKBONES.keys())}"
        )
    
    if model_name not in SUPPORTED_BACKBONES[backbone_type]:
        raise ValueError(
            f"Unsupported model name '{model_name}' for backbone '{backbone_type}'. "
            f"Supported: {SUPPORTED_BACKBONES[backbone_type]}"
        )
    
    if text_encoder_type and text_encoder_type not in SUPPORTED_TEXT_ENCODERS:
        raise ValueError(
            f"Unsupported text encoder: {text_encoder_type}. "
            f"Supported: {SUPPORTED_TEXT_ENCODERS}"
        )


# Export main function
__all__ = [
    "build_efficientsam3_image_model", 
    "load_checkpoint_weights",
    "get_model_info",
    "validate_config",
    "SUPPORTED_BACKBONES",
    "SUPPORTED_TEXT_ENCODERS",
]
