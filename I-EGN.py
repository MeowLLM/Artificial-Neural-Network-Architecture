import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, add_self_loops
import math

# --- 1. Custom EReLU Activation Function ---
# This is the learnable activation function you provided.
class EReLU(nn.Module):
    def __init__(
        self,
        init_a=0.0,
        init_b=1.0,
        init_c=0.5,
        init_d=0.0,
        init_i=1.0,
        eps=1e-6,
        safety_mode="too_close"  # options: "too_close", "close"
    ):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(init_a))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.c = nn.Parameter(torch.tensor(init_c))
        self.d = nn.Parameter(torch.tensor(init_d))
        self.i = nn.Parameter(torch.tensor(init_i))
        self.eps = eps
        self.safety_mode = safety_mode

    def forward(self, x):
        a, b, c, d, i = self.a.to(x), self.b.to(x), self.c.to(x), self.d.to(x), self.i.to(x)
        numerator = a + b * x - c * torch.abs(x) + d

        if self.safety_mode == "too_close":
            # only replace if |i| is almost zero
            safe_i = torch.where(torch.abs(i) < self.eps,
                                 torch.tensor(torch.e, device=x.device, dtype=x.dtype),
                                 i)
        elif self.safety_mode == "close":
            # smoothly push i away from zero: add eps
            safe_i = i + self.eps * torch.sign(i)  # nudges i away from 0
        else:
            raise ValueError(f"Unknown safety_mode: {self.safety_mode}")

        return numerator / safe_i

# --- 2. Custom Message Passing Layer with EReLU ---
# UPDATE: Replaced all ReLU activations with the new EReLU module.

class EIGNConv(MessagePassing):
    """
    EIGN Convolutional Layer
    This layer is inspired by the diagram provided and implements a flexible
    message passing scheme using the custom EReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(EIGNConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels

        # --- Learnable Message Networks with EReLU ---
        self.msg_net_forward = nn.Sequential(
            nn.Linear(2 * in_channels, in_channels),
            EReLU(), # Replaced ReLU
            nn.Linear(in_channels, out_channels)
        )
        self.msg_net_backward = nn.Sequential(
            nn.Linear(2 * in_channels, in_channels),
            EReLU(), # Replaced ReLU
            nn.Linear(in_channels, out_channels)
        )

        # Recurrent message and final update layer
        self.lin_recurrent = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(out_channels * 3 + in_channels, out_channels)

        # Attention mechanism
        self.attention_net = nn.Sequential(
            nn.Linear(2 * in_channels, 1),
            nn.LeakyReLU(0.2)
        )
        
        # Final activation for the node update
        self.final_activation = EReLU() # Replaced F.relu

    def forward(self, x, edge_index):
        # Add self-loops for stability, especially important in GNNs
        edge_index_with_self_loops, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index_with_self_loops
        
        # Compute attention scores
        attention_input = torch.cat([x[row], x[col]], dim=-1)
        alpha = self.attention_net(attention_input)
        # The attention mechanism here is simplified; a more robust implementation
        # might use scatter_softmax for proper normalization per node.
        # For this example, we'll use a global softmax.
        alpha = F.softmax(alpha, dim=0)

        # Propagate messages
        forward_msg = self.propagate(edge_index_with_self_loops, x=x, message_type='forward', attention=alpha)
        backward_msg = self.propagate(edge_index_with_self_loops[[1, 0]], x=x, message_type='backward', attention=alpha)
        recurrent_msg = self.lin_recurrent(x)

        # Update node features
        update_input = torch.cat([x, forward_msg, backward_msg, recurrent_msg], dim=1)
        updated_x = self.lin_update(update_input)
        
        # Apply final EReLU activation
        return self.final_activation(updated_x)


    def message(self, x_j, x_i, message_type, attention):
        tmp = torch.cat([x_i, x_j], dim=1)
        if message_type == 'forward':
            msg = self.msg_net_forward(tmp)
        elif message_type == 'backward':
            msg = self.msg_net_backward(tmp)
        else:
            # Fallback for safety, though should not be reached with current logic
            msg = torch.zeros((tmp.size(0), self.out_channels), device=tmp.device)
        
        # Apply attention to the message
        return attention * msg

# --- 3. The Main EIGN Model ---
# This class orchestrates the stages shown in your diagram.

class EIGN(nn.Module):
    def __init__(self, num_nodes, num_features, num_classes):
        super(EIGN, self).__init__()
        self.num_nodes = num_nodes
        self.conv1 = EIGNConv(num_features, 16)
        self.conv2 = EIGNConv(16, 32)

        # A layer to learn edge scores for refinement
        # CORRECTED: The input dimension now correctly reflects the output of conv1 (16 features).
        # Concatenating two nodes' features gives 16 * 2 = 32.
        self.edge_refiner = nn.Sequential(
            nn.Linear(16 * 2, 1),
            nn.Sigmoid()
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # --- Stage 1 & 2: Initial Convolution and Feature Learning ---
        print("Initial number of edges (Stage 2):", edge_index.shape[1])
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)

        # --- Stage 3: Edge Refinement ---
        # Refine edges based on the learned node features from the first layer
        if edge_index.numel() > 0:
            row, col = edge_index
            # Create edge features by concatenating features of connected nodes
            edge_features = torch.cat([x[row], x[col]], dim=-1)
            edge_scores = self.edge_refiner(edge_features).squeeze()
            
            # Create a mask to keep edges with a score above a threshold (e.g., 0.5)
            mask = edge_scores > 0.5
            refined_edge_index = edge_index[:, mask]
            print(f"Refined number of edges (Stage 3): {refined_edge_index.shape[1]} (kept {mask.sum().item()})")
        else:
            # If there are no edges to begin with, keep it that way
            refined_edge_index = edge_index

        # --- Stage 4: Final Convolution and Classification ---
        x = self.conv2(x, refined_edge_index)
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)


# --- 4. Example Usage & Training Loop ---
if __name__ == '__main__':
    # --- Define Graph Properties ---
    num_nodes = 10
    num_features = 8
    num_classes = 3

    # --- Create a Fully Connected Graph for Demonstration ---
    # This is a more robust way to create sample data than using deprecated functions.
    print("--- Creating a simple fully-connected graph for demonstration ---")
    x = torch.randn(num_nodes, num_features)
    
    # Create a dense adjacency matrix where all nodes are connected
    adj = torch.ones(num_nodes, num_nodes)
    # Remove self-loops from the initial adjacency matrix; they will be added in the model
    adj.fill_diagonal_(0) 
    
    # Convert the dense adjacency matrix to the sparse edge_index format
    row, col = torch.where(adj > 0)
    edge_index = torch.stack([row, col], dim=0)
    
    # Create random labels for the nodes
    y = torch.randint(0, num_classes, (num_nodes,))
    
    # Combine into a PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    print("Sample Data:", data)

    # --- Model Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EIGN(num_nodes=num_nodes, num_features=num_features, num_classes=num_classes).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:02d}, Loss: {loss.item():.4f}')

    # --- Evaluation ---
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred == data.y).sum()
    acc = int(correct) / int(num_nodes)
    print(f'\n--- Evaluation ---')
    print(f'Accuracy: {acc:.4f}')
