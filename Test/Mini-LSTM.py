import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
#       Mini-LSTM Implementation based on the diagram
# ==============================================================================

class MiniLSTMCell(nn.Module):
    """
    Implements a single step of the Mini-LSTM cell based on the provided diagram.
    
    This cell is a simplified recurrent unit that takes an input and a previous
    hidden state to compute the next hidden state and an output. It does not have
    a separate cell state like a standard LSTM.
    """
    def __init__(self, input_size, hidden_size):
        """
        Initializes the layers and parameters of the Mini-LSTM cell.

        Args:
            input_size (int): The number of expected features in the input `x`.
            hidden_size (int): The number of features in the hidden state `h`.
        """
        super(MiniLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # A single linear layer to process the concatenated input and previous hidden state.
        # This is more efficient than separate layers. We'll split the output.
        # It needs to produce outputs for three gates/activations.
        self.W_all = nn.Linear(input_size + hidden_size, hidden_size * 3)

    def forward(self, x, h_prev):
        """
        Defines the forward pass for a single time step of the Mini-LSTM.

        Args:
            x (torch.Tensor): The input for the current time step, with shape 
                              (batch_size, input_size).
            h_prev (torch.Tensor): The previous hidden state, with shape 
                                   (batch_size, hidden_size).

        Returns:
            tuple: A tuple containing:
                   - output (torch.Tensor): The output for the current time step.
                   - h_next (torch.Tensor): The next hidden state.
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)
        
        # Apply the linear transformation
        gates = self.W_all(combined)
        
        # Split the result into the three parts needed for the cell's operations
        g_tanh, g_sigma1, g_sigma2 = gates.chunk(3, 1)

        # --- Operations based on the diagram ---

        # 1. First Tanh and Sigmoid gates acting on the combined input
        a = torch.tanh(g_tanh)
        g1 = torch.sigmoid(g_sigma1)

        # 2. First element-wise multiplication to produce the output
        output = g1 * a
        
        # 3. Second Sigmoid gate (g_sigma2)
        g2 = torch.sigmoid(g_sigma2)

        # 4. Second element-wise multiplication to produce the next hidden state
        h_next = g2 * torch.tanh(output) # Diagram shows a Tanh on the output before the final multiplication

        return output, h_next

class MiniLSTM(nn.Module):
    """
    A wrapper class that processes a sequence of inputs using the MiniLSTMCell.
    This allows it to be used similarly to nn.LSTM or nn.GRU.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        """
        Initializes the multi-layer Mini-LSTM network.
        
        Note: The provided diagram is for a single cell. This class extends it
              to a potentially multi-layer model for practical use.
        """
        super(MiniLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(MiniLSTMCell(layer_input_size, hidden_size))
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 and num_layers > 1 else None

    def forward(self, input_seq, initial_h=None):
        """
        Forward pass for the multi-layer Mini-LSTM network.

        Args:
            input_seq (torch.Tensor): The input sequence, with shape 
                                      (seq_len, batch_size, input_size).
            initial_h (torch.Tensor, optional): A tensor of initial hidden states 
                                                for each layer. If None, zero states are used.
                                                Shape: (num_layers, batch_size, hidden_size).

        Returns:
            tuple: A tuple containing:
                   - outputs (torch.Tensor): The output features from the last layer for each time step.
                                             Shape: (seq_len, batch_size, hidden_size).
                   - final_h (torch.Tensor): The final hidden states for each layer.
                                             Shape: (num_layers, batch_size, hidden_size).
        """
        seq_len, batch_size, _ = input_seq.shape

        # Initialize hidden states if not provided
        if initial_h is None:
            initial_h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=input_seq.device)

        current_h_states = [h for h in initial_h]
        
        current_layer_input = input_seq

        for i, layer in enumerate(self.layers):
            h_prev = current_h_states[i]
            outputs_sequence = []
            
            for t in range(seq_len):
                x_t = current_layer_input[t, :, :]
                output, h_next = layer(x_t, h_prev)
                outputs_sequence.append(output)
                h_prev = h_next # Update hidden state for the next time step
            
            current_h_states[i] = h_prev
            current_layer_input = torch.stack(outputs_sequence, dim=0)
            
            if self.dropout and i < self.num_layers - 1:
                current_layer_input = self.dropout(current_layer_input)

        outputs = current_layer_input
        final_h = torch.stack(current_h_states, dim=0)

        return outputs, final_h

# ==============================================================================
#       Example Usage
# ==============================================================================
if __name__ == '__main__':
    # --- Parameters ---
    batch_size = 8
    seq_len = 12
    input_features = 16
    hidden_features = 32
    num_model_layers = 2

    # --- Dummy Input Data ---
    # Shape: (sequence_length, batch_size, input_features)
    dummy_input = torch.randn(seq_len, batch_size, input_features)

    # --- Mini-LSTM Model Example ---
    print("--- Testing Multi-layer Mini-LSTM Model ---")
    mini_lstm_model = MiniLSTM(
        input_size=input_features, 
        hidden_size=hidden_features, 
        num_layers=num_model_layers,
        dropout=0.1
    )
    print(f"Mini-LSTM Model:\n{mini_lstm_model}\n")

    # Forward pass
    outputs, final_hidden_state = mini_lstm_model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Model Output shape: {outputs.shape}") # (seq_len, batch, hidden)
    print(f"Final hidden state shape: {final_hidden_state.shape}") # (num_layers, batch, hidden)
    print("-" * 40)
