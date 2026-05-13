import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def forward(self, inputs):
        """Layer Normalization.

        This module applies Layer Normalization, with rescaling and shift,
        only on the last dimension. See Lecture 07 (I), slide 23.

        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The input tensor. This tensor can have an arbitrary number N of
            dimensions, as long as `inputs.shape[N-1] == hidden_size`. The
            leading N - 1 dimensions `dims` can be arbitrary.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The output tensor, having the same shape as `inputs`.
        """

        # ==========================
        # TODO: Write your code here
        # ==========================
        # First you calculate the mean over the last dimension 
        mean = inputs.mean(dim=-1, keepdim=True)

        # Now calculate the variance (biased) and divide by n 
        variance = ((inputs - mean) ** 2).mean(dim=-1, keepdim=True)

        # Normalize
        x_hat = (inputs - mean )/torch.sqrt(variance + self.eps)

        # Now you apply the scale which is learnable and shift it
        outputs = x_hat * self.weight + self.bias

        return outputs


    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_size, num_heads):
        super().__init__()
        self.head_size = head_size
        self.num_heads = num_heads

        # ==========================
        # TODO: Write your code here
        # ==========================

        # Defining the embedding dimension
        embed_dim = num_heads * head_size 

        # Defining the Weights for query, key, and value
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=True)

        # Defining W_Y
        self.W_Y = nn.Linear(embed_dim, embed_dim, bias=True)

    def get_attention_weights(self, queries, keys, mask=None):
        """Compute the attention weights.

        This computes the attention weights for all the sequences and all the
        heads in the batch. For a single sequence and a single head (for
        simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        and K are the keys (matrix of size `(sequence_length, head_size)`), then
        the attention weights are computed as

            weights = softmax(Q * K^{T} / sqrt(head_size))

        Here "*" is the matrix multiplication. See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads.

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads.

        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        attention_weights (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)`)
            Tensor containing the attention weights for all the heads and all
            the sequences in the batch.
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        # Compute QK^T
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))
        # Scale by sqrt (head_size)
        attn_scores = attn_scores/math.sqrt(self.head_size)
        # Apply the padding mask before the softmax is applied
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        
        # Applying the softmax over the sequence dimension
        attention_weights = F.softmax(attn_scores, dim=-1)

        # Returning attention_weights
        return attention_weights

    def apply_attention(self, queries, keys, values, mask=None):
        """Apply the attention.

        This computes the output of the attention, for all the sequences and
        all the heads in the batch. For a single sequence and a single head
        (for simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        K are the keys (matrix of size `(sequence_length, head_size)`), and V are
        the values (matrix of size `(sequence_length, head_size)`), then the ouput
        of the attention is given by

            weights = softmax(Q * K^{T} / sqrt(head_size))
            attended_values = weights * V
            outputs = concat(attended_values)

        Here "*" is the matrix multiplication, and "concat" is the operation
        that concatenates the attended values of all the heads (see the
        `merge_heads` function). See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads.

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads.

        values (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the values for all the positions in the sequences
            and all the heads.

        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the concatenated outputs of the attention for all
            the sequences in the batch, and all positions in each sequence.
        """

        # ==========================
        # TODO: Write your code here
        # ==========================

        # Calculating the attention weights
        attn_weights = self.get_attention_weights(queries, keys, mask)

        # Computing the attended values
        values_attended = torch.matmul(attn_weights, values)

        # Concatenate the heads
        outputs = self.merge_heads(values_attended)
        
        # Returning outputs 
        return outputs 

    def split_heads(self, tensor):
        """Split the head vectors.

        This function splits the head vectors that have been concatenated (e.g.
        through the `merge_heads` function) into a separate dimension. This
        function also transposes the `sequence_length` and `num_heads` axes.
        It only reshapes and transposes the input tensor, and it does not
        apply any further transformation to the tensor. The function `split_heads`
        is the inverse of the function `merge_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Input tensor containing the concatenated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Reshaped and transposed tensor containing the separated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """

        # ==========================
        # TODO: Write your code here
        # ==========================
        # Defining batch_size and sequence_length
        batch_size, seq_length, _ = tensor.size()

        # Defining dimensions 
        dim = self.head_size

        # Reshape the dimension to create separate heads 
        tensor = tensor.view(batch_size, seq_length, self.num_heads, dim)

        # Now you need to transpose the sequence length and the number of heads
        tensor = tensor.transpose(1,2)
        
        # Return Tensor 
        return tensor

    def merge_heads(self, tensor):
        """Merge the head vectors.

        This function concatenates the head vectors in a single vector. This
        function also transposes the `sequence_length` and the newly created
        "merged" dimension. It only reshapes and transposes the input tensor,
        and it does not apply any further transformation to the tensor. The
        function `merge_heads` is the inverse of the function `split_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Input tensor containing the separated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Reshaped and transposed tensor containing the concatenated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """

        # ==========================
        # TODO: Write your code here
        # ==========================

        # Defining batch_size, number of heads, sequence length, and dimensions
        batch_size, num_heads, seq_length, dim = tensor.size()

        # Now apply the transpose back 
        tensor = tensor.transpose(1,2)

        # Merge heads together 
        tensor = tensor.contiguous().view(batch_size, seq_length, num_heads * dim)

        # Returning the tensor 
        return tensor

    def forward(self, hidden_states, mask=None):
        """Multi-headed attention.

        This applies the multi-headed attention on the input tensors `hidden_states`.
        For a single sequence (for simplicity), if X are the hidden states from
        the previous layer (a matrix of size `(sequence_length, num_heads * head_size)`
        containing the concatenated head vectors), then the output of multi-headed
        attention is given by

            Q = X * W_{Q} + b_{Q}        # Queries
            K = X * W_{K} + b_{K}        # Keys
            V = X * W_{V} + b_{V}        # Values

            Y = attention(Q, K, V)       # Attended values (concatenated for all heads)
            outputs = Y * W_{Y} + b_{Y}  # Linear projection

        Here "*" is the matrix multiplication.

        Parameters
        ----------
        hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Input tensor containing the concatenated head vectors for all the
            sequences in the batch, and all positions in each sequence. This
            is, for example, the tensor returned by the previous layer.

        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the output of multi-headed attention for all the
            sequences in the batch, and all positions in each sequence.
        """
        # ==========================
        # TODO: Write your code here
        # ==========================

        # Defining queries, keys, and values projections 
        q = self.W_Q(hidden_states)
        k = self.W_K(hidden_states)
        v = self.W_V(hidden_states) 

        # Splitting projections into heads 
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Applying attention 
        values_attended = self.apply_attention(q, k, v, mask)

        # Producing the Final Linear Projection 
        output = self.W_Y(values_attended)

        # Returning output 
        return output 

class PostNormAttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.30):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()


        self.layer_norm_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(head_size=embed_dim//num_heads, num_heads=num_heads)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x, mask=None):

        attention_outputs = self.attn(x, mask)
        attention_outputs = self.layer_norm_1(x + attention_outputs)
        outputs = self.linear(attention_outputs)

        outputs = self.layer_norm_2(outputs + attention_outputs)
        return outputs

class PreNormAttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """A decoder layer.

        This module combines a Multi-headed Attention module and an MLP to
        create a layer of the transformer, with normalization and skip-connections.
        See Lecture 06, slide 33.

        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(head_size=embed_dim//num_heads, num_heads=num_heads)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x, mask=None):
        # ==========================
        # TODO: Write your code here
        # ==========================
        # First apply the pre-norm before attention 
        x_norm = self.layer_norm_1(x)

        # Defining attention_outputs 
        outputs_attn = self.attn(x_norm, mask)
        outputs_attn = x + outputs_attn

        # Defining the prenorm before the feedforward 
        attn_norm = self.layer_norm_2(outputs_attn)

        # Defining outputs 
        outputs = self.linear(attn_norm)

        outputs = outputs_attn + outputs

        # Return outputs 
        return outputs



class Transformer(nn.Module):

    def __init__(
        self,
        vocabulary_size: int = 30522,
        sequence_length: int = 256,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 1,
        num_layers: int = 2,
        block: str = "prenorm",
        dropout=0.3,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            block - Type of attention block
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        #Adding the cls token to the sequnence
        self.sequence_length= 1 + sequence_length
        # Layers/Networks
        self.embedding = nn.Embedding(vocabulary_size, embed_dim)
        if block =="prenorm":
          self.transformer = nn.ModuleList([PreNormAttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        elif block =="postnorm":
          self.transformer = nn.ModuleList([PostNormAttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        else:
          raise ValueError(f"Invalid block type {block}")
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,self.sequence_length,embed_dim))

    def forward(self, x, mask=None):
        """Transformer

        This is a small version of  Transformer

        Parameters
        ----------
        x - (`torch.LongTensor` of shape `(batch_size, sequence length)`)
            The input tensor containing text.

        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, embed_dim)`)
            A tensor containing the output from the mlp_head.
        """
        # Preprocess input

        x = self.embedding(x)
        B, T, _ = x.shape

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]
        # Add dropout and then the transformer (remember to update the mask because of the CLS token)
        # ==========================
        # TODO: Write your code here
        # ==========================

        # Defining x with the self.dropout 
        x = self.dropout(x)
        
        # If mask is not none, defining mask_cls and mask 
        if mask is not None:
            mask_cls = torch.ones((B, 1), dtype=mask.dtype, device=mask.device)
            mask = torch.cat([mask_cls, mask], dim=1)
        

        for layer in self.transformer:
            x = layer(x, mask)

        # Take the cls token representation and send it to mlp_head
        # ==========================
        # TODO: Write your code here
        # ==========================
        representation_cls = x[:, 0]
        output = self.mlp_head(representation_cls)
        return output 
