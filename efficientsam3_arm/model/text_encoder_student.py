# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import os
import torch
import torch.nn as nn
from efficientsam3_arm.backbones.mobile_clip import MobileCLIPTextTransformer
from efficientsam3_arm.model.tokenizer_ve import SimpleTokenizer

class TextStudentEncoder(nn.Module):
    def __init__(self, cfg, context_length, output_dim, bpe_path=None):
        super().__init__()
        self.context_length = context_length
        
        if bpe_path is None:
            # Get the absolute path to the assets directory
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            bpe_path = os.path.join(project_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
        
        # Verify the file exists
        if not os.path.exists(bpe_path):
            raise FileNotFoundError(f"BPE vocabulary file not found at: {bpe_path}")
        
        self.tokenizer = SimpleTokenizer(bpe_path=bpe_path)
        
        # MobileCLIP Transformer (Student)
        # Includes its own embedding layer (49408 x dim)
        self.encoder = MobileCLIPTextTransformer(
            cfg=cfg,
            projection_dim=cfg["dim"]
        )
        
        # Post-Projection (Student Dim -> Output Dim)
        self.projector = nn.Linear(cfg["dim"], output_dim)

    def forward(self, text, input_boxes=None, device=None):
        # 1. Tokenize
        tokenized = self.tokenizer(text, context_length=self.context_length).to(device)
        
        # 2. Get input embeddings
        # We use the student's embedding layer
        input_embeds = self.encoder.forward_embedding(tokenized) # [Batch, Seq, Dim]
        
        # 3. MobileCLIP Transformer
        # Pass embeddings directly
        text_memory = self.encoder(input_embeds, return_all_tokens=True, input_is_embeddings=True) 
        
        # 4. Post-Project
        text_memory = self.projector(text_memory) # [Batch, Seq, OutputDim]
        
        # 5. Prepare output tuple compatible with SAM3VLBackbone
        # text_attention_mask: [Batch, Seq] (True for padding, False for valid - inverted logic)
        # But wait, VETextEncoder returns (tokenized != 0).bool().ne(1)
        # (tokenized != 0) is True for valid. .ne(1) makes it False for valid.
        # So False is valid, True is padding.
        text_attention_mask = (tokenized != 0).bool().ne(1)
        
        # Return tuple: (mask, memory, embeds)
        # memory and embeds are [Seq, Batch, Dim]
        return text_attention_mask, text_memory.transpose(0, 1), input_embeds.transpose(0, 1)
