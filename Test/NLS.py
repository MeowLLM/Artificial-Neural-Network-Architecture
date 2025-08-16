class CustomLSTMCell(nn.Module):
    """
    Implements a single step of the custom LSTM-like cell based on the provided diagram.
    """
    def __init__(self, input_size: int, hidden_size: int):
        """
        Initializes the layers and parameters of the custom cell.

        Args:
            input_size (int): The number of expected features in the input `x`.
            hidden_size (int): The number of features in the hidden state `h` and memory state `c`.
        """
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation functions used in the cell
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh() # Still available if needed elsewhere

        # === Define Linear transformations for each gate ===

        # Gate Forgot: Takes ReLU(input) and Hidden State
        self.linear_forget = nn.Linear(input_size + hidden_size, hidden_size)

        # Gate Activation Memory: Takes ReLU(input) and Hidden State
        self.linear_activation_memory = nn.Linear(input_size + hidden_size, hidden_size)

        # Gate Update: Takes only Hidden State
        self.linear_update = nn.Linear(hidden_size, hidden_size)

        # Gate Input: Takes Activation Memory and IN Memory State
        self.linear_input = nn.Linear(hidden_size + hidden_size, hidden_size)

        # Gate Output: Takes ReLU(Activation Memory) and new Memory State
        self.linear_output = nn.Linear(hidden_size + hidden_size, hidden_size)

    def forward(self, x_t, states):
        """
        Defines the forward pass for a single time step.

        Args:
            x_t (torch.Tensor): Input tensor for the current time step, shape (batch_size, input_size).
            states (tuple): A tuple containing the previous hidden state and memory state
                            (h_{t-1}, c_{t-1}), each of shape (batch_size, hidden_size).

        Returns:
            tuple: A tuple containing the output/hidden state for the current step (h_t) and
                   a tuple of the new states (h_t, c_t).
        """
        h_prev, c_prev = states

        # 1. Apply ReLU to the input
        relu_x = self.relu(x_t)

        # 2. Concatenate for Forget and Activation Memory gates
        combined_fa = torch.cat((relu_x, h_prev), dim=1)

        # 3. Calculate Gate Forgot (f_t)
        f_t = self.sigmoid(self.linear_forget(combined_fa))

        # 4. Calculate Gate Activation Memory (a_t) using sigmoid as requested.
        a_t = self.sigmoid(self.linear_activation_memory(combined_fa))

        # 5. Calculate Gate Update (u_t)
        u_t = self.sigmoid(self.linear_update(h_prev))

        # 6. Concatenate for Input Gate
        combined_i = torch.cat((a_t, c_prev), dim=1)
        
        # 7. Calculate Gate Input (i_t)
        i_t = self.sigmoid(self.linear_input(combined_i))

        # 8. Calculate the new Memory State (c_t)
        # c_t = (forget_gate * prev_memory) + (input_gate * update_gate)
        c_t = (f_t * c_prev) + (i_t * u_t)

        # 9. Apply ReLU to Activation Memory for the Output Gate
        relu_a = self.relu(a_t)

        # 10. Concatenate for the Output Gate
        combined_o = torch.cat((relu_a, c_t), dim=1)
        
        # 11. Calculate Gate Output, which is also the new Hidden State (h_t)
        # As per the diagram, the output of this gate is the final hidden state.
        h_t = self.sigmoid(self.linear_output(combined_o))

        # The diagram shows 'Output' and 'Out Hidden State' are the same.
        output = h_t
        new_states = (h_t, c_t)
        
        return output, new_states


class CustomLSTM(nn.Module):
    """
    A wrapper class that processes a sequence of inputs using the CustomLSTMCell.
    This mimics the behavior of torch.nn.LSTM.
    """
    def __init__(self, input_size: int, hidden_size: int):
        """
        Initializes the custom LSTM layer.

        Args:
            input_size (int): The number of expected features in the input `x`.
            hidden_size (int): The number of features in the hidden state `h`.
        """
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell = CustomLSTMCell(input_size, hidden_size)

    def forward(self, input_seq, initial_states=None):
        """
        Processes a sequence of inputs.

        Args:
            input_seq (torch.Tensor): The input sequence, shape (seq_len, batch_size, input_size).
            initial_states (tuple, optional): A tuple (h_0, c_0) for the initial states.
                                              If None, states are initialized to zeros. Defaults to None.

        Returns:
            tuple: A tuple containing:
                   - outputs (torch.Tensor): A tensor of all hidden states from the sequence,
                                             shape (seq_len, batch_size, hidden_size).
                   - final_states (tuple): The final (h_n, c_n) states after the last time step.
        """
        seq_len, batch_size, _ = input_seq.shape

        # Initialize hidden and memory states if not provided
        if initial_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=input_seq.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=input_seq.device)
        else:
            h_t, c_t = initial_states
            
        outputs = []

        # Loop over the sequence length
        for t in range(seq_len):
            x_t = input_seq[t, :, :]
            output, (h_t, c_t) = self.cell(x_t, states=(h_t, c_t))
            outputs.append(output.unsqueeze(0))

        # Concatenate outputs along the sequence dimension
        outputs = torch.cat(outputs, dim=0)
        final_states = (h_t, c_t)
        
        return outputs, final_states
