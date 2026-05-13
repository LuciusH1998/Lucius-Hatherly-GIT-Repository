import torch
from typing import List, Tuple
from torch import nn


class Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
       
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super(Linear, self).__init__()
        # Defining input features 
        self.in_features = in_features 
        # Defining output features 
        self.out_features = out_features
        # Defining weights 
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # Defining bias 
        self.bias = nn.Parameter(torch.Tensor(out_features))
        ##nn.init.xavier_normal_(self.weight)
        ##nn.init.zeros_(self.bias)
    
    def forward(self, input):
        """
            :param input: [bsz, in_features]
            :return result [bsz, out_features]
        """
        # Returning forward function 
        return torch.matmul(input, self.weight.t()) + self.bias

class MLP(torch.nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, activation: str = "relu"):
        super(MLP, self).__init__() 
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        assert len(hidden_sizes) > 1, "You should at least have one hidden layer"
        self.num_classes = num_classes
        self.activation = activation
        assert activation in ['tanh', 'relu', 'sigmoid'], "Invalid choice of activation"
        self.hidden_layers, self.output_layer = self._build_layers(input_size, hidden_sizes, num_classes)
        
        # Initializaton
        self._initialize_linear_layer(self.output_layer)
        for layer in self.hidden_layers:
            self._initialize_linear_layer(layer)
    
    def _build_layers(self, input_size: int, 
                        hidden_sizes: List[int], 
                        num_classes: int) -> Tuple[nn.ModuleList, nn.Module]:
        """
        Build the layers for MLP. Be ware of handlling corner cases.
        :param input_size: An int
        :param hidden_sizes: A list of ints. E.g., for [32, 32] means two hidden layers with 32 each.
        :param num_classes: An int
        :Return:
            hidden_layers: nn.ModuleList. Within the list, each item has type nn.Module
            output_layer: nn.Module
        """
        # Defining Hidden Layers 
        hidden_layers = nn.ModuleList()
        # Defining previous_size 
        previous_size = input_size
        # Iterating through hidden_sizes
        for hidden_size in hidden_sizes:
            # Appending hidden layers 
            hidden_layers.append(Linear(previous_size, hidden_size))
            # Defining previous size as the hidden size 
            previous_size = hidden_size
        # Defining the output layer through calling the previous Linear function 
        output_layer = Linear(previous_size, num_classes)
        # Return hidden layers and output layer 
        return hidden_layers, output_layer 
    
    def activation_fn(self, activation, inputs: torch.Tensor) -> torch.Tensor:
        """ process the inputs through different non-linearity function according to activation name """
        # Defining activation as relu
        if activation == "relu":
            return torch.relu(inputs)
        # Defining activation as tanh 
        elif activation == "tanh":
            return torch.tanh(inputs)
        # Defining activations as sigmoid 
        elif activation == "sigmoid":
            return torch.sigmoid(inputs)
        
    def _initialize_linear_layer(self, module: nn.Linear) -> None:
        """ For bias set to zeros. For weights set to glorot normal """
        # Defining xavier_normal_ for module.weight
        nn.init.xavier_normal_(module.weight)
        # Defining module.bias 
        nn.init.zeros_(module.bias)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """ Forward images and compute logits.
        1. The images are first fattened to vectors. 
        2. Forward the result to each layer in the self.hidden_layer with activation_fn
        3. Finally forward the result to the output_layer.
        
        :param images: [batch, channels, width, height]
        :return logits: [batch, num_classes]
        """
        # Flattening Images 
        batch_size = images.size(0)
        x = images.view(batch_size, -1)

        # Hidden layers with activation 
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation_fn(self.activation, x)
        
        # Defining Output layers 
        logits = self.output_layer(x)
        
        # Return Logits
        return logits 
