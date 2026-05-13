import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size


        self.w_ir = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_iz = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_in = nn.Parameter(torch.empty(hidden_size, input_size))

        self.b_ir = nn.Parameter(torch.empty(hidden_size))
        self.b_iz = nn.Parameter(torch.empty(hidden_size))
        self.b_in = nn.Parameter(torch.empty(hidden_size))

        self.w_hr = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hn = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.b_hr = nn.Parameter(torch.empty(hidden_size))
        self.b_hz = nn.Parameter(torch.empty(hidden_size))
        self.b_hn = nn.Parameter(torch.empty(hidden_size))
        for param in self.parameters():
            nn.init.uniform_(param, a=-(1/hidden_size)**0.5, b=(1/hidden_size)**0.5)


    def forward(self, inputs, hidden_states):
        """GRU.

        This is a Gated Recurrent Unit
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, input_size)`)
          The input tensor containing the embedded sequences.

        hidden_states (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
          The (initial) hidden state.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
          A feature tensor encoding the input sentence.

        hidden_states (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
          The final hidden state.
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        # Obtaining the batch size and given sequence length from the input tensor
        # The inputs shape are: (batch_size, sequence_length (seq_length), input_size)
        batch_size, seq_length, _ = inputs.size()
        
        # Removing the hidden state's first dimension 
        # hidden_states will initially have shapes (1, batch_size, hidden_size)
        # The squeeze produces (batch_size, hidden_size)
        h_t = hidden_states.squeeze(0)

        # Creating list which stores hidden state outputs at every timestep
        outputs = []

        # Looping through all the timesteps in the input sequence 
        for t in range(seq_length):
            # Obtaining the input vector at the given timestep t
            # The shape has a (batch_size, input_size)
            x_t = inputs[:, t, :]
            # Computing the reset gate 
            r_t = torch.sigmoid(F.linear(x_t, self.w_ir, self.b_ir) + 
                                F.linear(h_t, self.w_hr, self.b_hr))
            
            # Computing the update gate z_t
            z_t = torch.sigmoid(F.linear(x_t, self.w_iz, self.b_iz) + 
                                F.linear(h_t, self.w_hz, self.b_hz))
            
            # Computing the candidate hidden gate n_t
            n_t = torch.tanh(F.linear(x_t, self.w_in, self.b_in) + r_t*(F.linear(h_t, self.w_hn, self.b_hn)))

            # Computing hidden_state 
            h_t = (1-z_t)*n_t+z_t*h_t

            # Save hidden state for this given timestep
            # unsqueeze(1) converts shape from (batch_size, hidden_size)
            #  to (batch_size, 1, hidden_size) so we can concatenate later
            outputs.append(h_t.unsqueeze(1))

        # Concatenating the outputs across every timestep 
        outputs = torch.cat(outputs, dim=1)

        # Adding back the first dimension to align with the given hidden state 
        hidden_states = h_t.unsqueeze(0)

        # returning outputs and hidden_state 
        return outputs, hidden_states


class Attn(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        dropout=0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)

        self.W = nn.Linear(hidden_size*2, hidden_size)

        self.V = nn.Linear(hidden_size, hidden_size)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs, hidden_states, mask = None):
        """Soft Attention mechanism.

        This is a one layer MLP network that implements Soft (i.e. Bahdanau) Attention with masking
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the embedded sequences.

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The (initial) hidden state.

        mask ( optional `torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence with attention applied.

        x_attn (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`)
            The attention vector.
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        # Defining batch_size and seq_length
        batch_size, seq_length, _ = inputs.size()

        # Obtain the last hidden state of the decoder 
        h_t = hidden_states[-1]

        # Expand the hidden states across the sequence lengths 
        h_t = h_t.unsqueeze(1).repeat(1, seq_length, 1)

        # Now you should concatenate the output of the encoder with the hidden state 
        x = torch.cat((inputs, h_t), dim=2)

        # Applying linear transformation with tanh
        x = self.tanh(self.W(x))

        # Applying second linear with multiply and sum 
        x = self.V(x)

        # Sum across the hidden dimensions to produce attention scores 
        attn_x = torch.sum(x, dim=2, keepdim=True)

        # Apply the given mask if provided 
        if mask is not None:
            attn_x = attn_x.masked_fill(mask.unsqueeze(2) == 0, -1e9)
        
        # Now you apply the softmax 
        attn_x = self.softmax(attn_x)

        # Apply the attention weights to the output of the encoder 
        outputs = inputs * attn_x
        
        # Return outputs and attention 
        return outputs, attn_x


class Encoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            vocabulary_size, embedding_size, padding_idx=0,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, inputs, hidden_states):
        """GRU Encoder.

        This is a Bidirectional Gated Recurrent Unit Encoder network
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the token sequences.

        hidden_states
            The (initial) hidden state.
            - h (`torch.FloatTensor` of shape `(num_layers*2, batch_size, hidden_size)`)

        Returns
        -------
        x (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence.

        hidden_states
            The final hidden state.
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        # Defining the embedding layer 
        x = self.embedding(inputs)

        # Defining the drop out
        x = self.dropout(x)

        # Run the bidirectional GRU
        x, hidden_states = self.rnn(x, hidden_states)

        # Note you can define the x shape as (batch_size, sequence_length, hidden_states * 2)
        # Now you should split the forward and backward outputs of x 
        forward_x = x[:,:,:self.hidden_size]
        backward_x = x[:,:,self.hidden_size:]

        # Now proceed to sum the two directions together 
        x = forward_x + backward_x

        # You can define the hidden state shape as (num_layers*2, batch_size, hidden_size)
        forward_h = hidden_states[0:self.num_layers]
        backward_h = hidden_states[self.num_layers:self.num_layers*2]

        hidden_states = forward_h + backward_h

        # Returning x and hidden_states
        return x, hidden_states

    def initial_states(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        shape = (self.num_layers*2, batch_size, self.hidden_size)
        # The initial state is a constant here, and is not a learnable parameter
        h_0 = torch.zeros(shape, dtype=torch.float, device=device)
        return h_0

class DecoderAttn(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout=0.0,
        with_attn=True,
    ):

        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        if with_attn:
            self.mlp_attn = Attn(hidden_size, dropout)
        else:
            self.mlp_attn = None

    def forward(self, inputs, hidden_states, mask=None):
        """GRU Decoder network with Soft attention

        This is a Unidirectional Gated Recurrent Unit Encoder network

        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the encoded input sequence.

        hidden_states
            The (initial) hidden state.
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)

        Returns
        -------
        x (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor decoding the orginally encoded input sentence.

        hidden_states
            The final hidden state.
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        if self.mlp_attn is not None:
            x, _ = self.mlp_attn(inputs, hidden_states, mask)
        
        else:
            x = inputs
        
        # Now run the GRU on your intended inputs 
        x, hidden_states = self.rnn(x, hidden_states)

        # Return the decoder output and hidden states 
        return x, hidden_states
                

class EncoderDecoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout = 0.0,
        encoder_only=False,
        with_attn=True,
    ):
        super().__init__()
        self.encoder_only = encoder_only
        self.encoder = Encoder(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        if not encoder_only:
          self.decoder = DecoderAttn(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            with_attn=with_attn,
          )

    def forward(self, inputs, mask=None):
        """GRU Encoder-Decoder network with Soft attention.

        This is a Gated Recurrent Unit network for Sentiment Analysis. This
        module returns a decoded feature for classification.

        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
         Returns
        -------
        x (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor representing the input sentence for sentiment analysis

        hidden_states
            The final hidden state. This is a tuple containing
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
        """
        hidden_states = self.encoder.initial_states(inputs.shape[0])
        x, hidden_states = self.encoder(inputs, hidden_states)
        if self.encoder_only:
            return x[:, 0], hidden_states
        x, hidden_states = self.decoder(x, hidden_states, mask)
        return x[:, 0], hidden_states
