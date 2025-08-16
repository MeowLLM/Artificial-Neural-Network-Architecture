# -----------------------------------------------------------------------------
# Recurrent State Hidden (RSH) Implementation
# -----------------------------------------------------------------------------

class RSHCell(nn.Module):
    """
    An implementation of the Recurrent State Hidden (RSH) cell based on the provided diagram.

    This cell takes an input, a previous hidden state, and a previous memory state
    to compute the next hidden and memory states, along with an output.
    """
    def __init__(self, input_size, hidden_size):
        """
        Initializes the RSH cell.

        Args:
            input_size (int): The number of expected features in the input `x`.
            hidden_size (int): The number of features in the hidden state `h`.
        """
        super(RSHCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define the linear transformations for the various gates and operations.
        # We use a single linear layer for efficiency, splitting the output later.
        # This combines the transformations for the five sigmoid gates and one tanh gate.
        self.W_xh = nn.Linear(input_size, 2 * hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        self.W_c = nn.Linear(hidden_size, hidden_size, bias=False) # For the final output modulation

    def forward(self, x, states):
        """
        Forward pass for the RSH cell.

        Args:
            x (torch.Tensor): The input for the current time step, with shape (batch_size, input_size).
            states (tuple): A tuple containing the previous hidden state and memory state,
                            (h_prev, c_prev), each with shape (batch_size, hidden_size).

        Returns:
            tuple: A tuple containing the output, next hidden state, and next memory state.
                   (output, (h_next, c_next)).
        """
        h_prev, c_prev = states

        # Linear transformations for input and hidden state
        xh_gates = self.W_xh(x)
        hh_gates = self.W_hh(h_prev)

        # Split the gate results for clarity
        x_gate1, x_gate4 = xh_gates.chunk(2, 1)
        h_gate2, h_gate_u, h_gate4, h_gate_out = hh_gates.chunk(4, 1)

        # --- Operations based on the diagram ---

        # 1. First sigmoid gate on input `x` (not explicitly shown but implied for combination)
        # Let's assume the diagram simplifies the standard GRU/LSTM structure where gates
        # depend on both x and h. The connections show this.

        # 2. Gate g2 = sigmoid(W2 * h_prev)
        g2 = torch.sigmoid(h_gate2)

        # 3. Candidate update u = tanh(W3 * h_prev)
        u = torch.tanh(h_gate_u)

        # 4. Next memory state c_next = c_prev + g2 * u
        c_next = c_prev + (g2 * u)

        # 5. Gate g3 = sigmoid(W_c * c_next)
        # This gate modulates the memory to produce the hidden state
        g3 = torch.sigmoid(self.W_c(c_next))

        # 6. Next hidden state h_next = g3 * tanh(c_next)
        h_next = g3 * torch.tanh(c_next)

        # 7. Gate g4 = sigmoid(W_x4 * x + W_h4 * h_prev)
        # This combines inputs (blue and red lines) to modulate the final output
        g4 = torch.sigmoid(x_gate4 + h_gate4)

        # 8. Final output = g4 * h_next
        output = g4 * h_next

        return output, (h_next, c_next)


class RSH(nn.Module):
    """
    A multi-layer Recurrent State Hidden (RSH) network.

    This module processes sequences by applying the RSHCell over each time step.
    It can be stacked to create a deep RNN.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        """
        Initializes the multi-layer RSH network.

        Args:
            input_size (int): The number of expected features in the input `x`.
            hidden_size (int): The number of features in the hidden state `h`.
            num_layers (int): Number of recurrent layers.
            dropout (float): If non-zero, introduces a Dropout layer on the outputs of
                             each RSH layer except the last layer. Default: 0.0
        """
        super(RSH, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(RSHCell(layer_input_size, hidden_size))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, x, initial_states=None):
        """
        Forward pass for the multi-layer RSH network.

        Args:
            x (torch.Tensor): The input sequence, with shape (seq_len, batch_size, input_size).
            initial_states (list of tuples, optional): A list of initial (h, c) states for each layer.
                                                       If None, zero states are used.

        Returns:
            tuple: A tuple containing:
                   - outputs (torch.Tensor): The output features from the last layer for each time step.
                                             Shape: (seq_len, batch_size, hidden_size).
                   - final_states (list of tuples): The final (h, c) states for each layer.
        """
        seq_len, batch_size, _ = x.size()

        # Initialize hidden and memory states if not provided
        if initial_states is None:
            initial_states = []
            for _ in range(self.num_layers):
                h0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
                c0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
                initial_states.append((h0, c0))

        current_states = list(initial_states)
        layer_outputs = []

        # Process sequence step-by-step
        for t in range(seq_len):
            input_t = x[t, :, :]
            for i, layer in enumerate(self.layers):
                states_t = current_states[i]
                output_t, (h_next, c_next) = layer(input_t, states_t)
                current_states[i] = (h_next, c_next)
                input_t = output_t # Input for the next layer is the output of the current one
                if self.dropout and i < self.num_layers - 1:
                    input_t = self.dropout(input_t)
            layer_outputs.append(output_t)

        outputs = torch.stack(layer_outputs, dim=0)
        final_states = current_states

        return outputs, final_states
