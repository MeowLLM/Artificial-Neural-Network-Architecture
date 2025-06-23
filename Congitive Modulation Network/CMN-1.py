import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

# ---------------------------- Memory Management System ----------------------------
class MemoryManager:
    def __init__(self, memory_size, similarity_threshold=0.9):
        self.long_term_memory = []
        self.working_memory = {}
        self.similarity_threshold = similarity_threshold
        self.memory_size = memory_size
    
    def write_working_memory(self, key, value):
        self.working_memory[key] = value
    
    def read_working_memory(self, key):
        return self.working_memory.get(key, None)
    
    def remove_working_memory(self, key):
        if key in self.working_memory:
            del self.working_memory[key]
    
    def consolidate_memory(self, new_data):
        if not self.long_term_memory:
            self.long_term_memory.append(new_data)
            return
        
        # Check similarity and replace most similar if above threshold
        similarities = [F.cosine_similarity(new_data, old_data, dim=0) 
                       for old_data in self.long_term_memory]
        if similarities:  # Handle empty list case
            max_sim, max_idx = torch.max(torch.stack(similarities), dim=0)
            if max_sim > self.similarity_threshold:
                self.long_term_memory[max_idx] = new_data
                return
        
        if len(self.long_term_memory) >= self.memory_size:
            self.long_term_memory.pop(0)
        self.long_term_memory.append(new_data)

# ---------------------------- Attention Modules ----------------------------
class FlashAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.memory_manager = MemoryManager(memory_size=100)
    
    # Add memory access methods
    def read_memory(self, key):
        return self.memory_manager.read_working_memory(key)
    
    def write_memory(self, key, value):
        self.memory_manager.write_working_memory(key, value)
    
    def remove_memory(self, key):
        self.memory_manager.remove_working_memory(key)
    
    def forward(self, x, memory_key=None, operation=None):
        # Memory operations
        if operation == 'read' and memory_key:
            mem_data = self.read_memory(memory_key)
            if mem_data is not None:
                x = x + mem_data
        elif operation == 'write' and memory_key:
            self.write_memory(memory_key, x)
        elif operation == 'remove' and memory_key:
            self.remove_memory(memory_key)
        
        # Attention mechanism
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scale = torch.tensor(np.sqrt(x.size(-1)), device=x.device, dtype=x.dtype)
        attn_weights = F.softmax(Q @ K.T / scale, dim=-1)
        attn_output = attn_weights @ V
        
        # Gated modulation
        combined = torch.cat([x, attn_output], dim=-1)
        gate_signal = torch.sigmoid(self.gate(combined))
        return gate_signal * attn_output

# ---------------------------- Network Nodes ----------------------------
class SubWorkerNode(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc_hidden = nn.Linear(input_dim, output_dim)
        self.fc_input = nn.Linear(input_dim, output_dim)
        self.output_layer = nn.Linear(output_dim * 2, output_dim)
        self.hebbian_weights = nn.Parameter(torch.randn(output_dim, output_dim))
        
    def forward(self, hidden_input, external_input, flash_attn, memory_key):
        # Process inputs
        h_out = torch.tanh(self.fc_hidden(hidden_input))
        x_out = torch.tanh(self.fc_input(external_input))
        
        # Combine inputs
        combined = torch.cat([h_out, x_out], dim=-1)
        output = torch.tanh(self.output_layer(combined))
        
        # Memory interaction - use write_memory method
        if memory_key:
            flash_attn.write_memory(memory_key, output)
        
        # Hebbian learning update (per-sample)
        self.update_hebbian_weights(h_out, x_out)
        
        return output, output  # (output, hidden_output)

    def update_hebbian_weights(self, h_out, x_out):
        with torch.no_grad():
            # Calculate Hebbian update per sample and average
            batch_size = h_out.size(0)
            hebb_updates = torch.zeros_like(self.hebbian_weights)
            
            for i in range(batch_size):
                h_sample = h_out[i]
                x_sample = x_out[i]
                hebb_updates += torch.outer(h_sample, x_sample)
            
            hebb_updates /= batch_size  # Average over batch
            self.hebbian_weights.data = 0.95 * self.hebbian_weights + 0.05 * hebb_updates

class CoWorkerNode(nn.Module):
    def __init__(self, input_dim, output_dim, num_subworkers):
        super().__init__()
        self.subworkers = nn.ModuleList([
            SubWorkerNode(input_dim, output_dim) for _ in range(num_subworkers)
        ])
        self.critical_thinking = nn.Linear(output_dim * num_subworkers, output_dim)
        self.pattern_thinking = nn.LSTM(output_dim * num_subworkers, output_dim, batch_first=True)
        self.flash_attn = FlashAttention(output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        sub_outputs = []
        for i, worker in enumerate(self.subworkers):
            out, _ = worker(x, x, self.flash_attn, f"sub_{i}")
            sub_outputs.append(out)
        
        # Critical thinking pathway
        combined = torch.cat(sub_outputs, dim=-1)
        critical_out = torch.relu(self.critical_thinking(combined))
        
        # Pattern thinking pathway - add sequence dimension
        pattern_in = combined.unsqueeze(1)  # Add sequence dimension
        pattern_out, _ = self.pattern_thinking(pattern_in)
        pattern_out = pattern_out.squeeze(1)  # Remove sequence dimension
        
        # Merge pathways
        merged = critical_out + pattern_out
        attn_out = self.flash_attn(merged)
        return self.norm(attn_out)

class BigNode(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_co_workers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.flash_attn = FlashAttention(hidden_dim)
        self.co_workers = nn.ModuleList([
            CoWorkerNode(hidden_dim, hidden_dim, num_subworkers=4) 
            for _ in range(num_co_workers)
        ])
        self.output_layer = nn.Linear(hidden_dim * num_co_workers, hidden_dim)
        self.validator = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, hidden_state=None, memory_key=None):
        # Initialize hidden state if None
        if hidden_state is None:
            hidden_state = torch.zeros(x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)
        
        # Process inputs
        x_proj = self.input_proj(x)
        h_proj = self.hidden_proj(hidden_state)
        
        # Memory retrieval - handle missing memory properly
        mem_data = self.flash_attn.read_memory(memory_key) if memory_key else None
        
        # Process memory data only if it exists
        if mem_data is not None:
            # Ensure memory data matches dimensions
            if mem_data.shape != x_proj.shape:
                mem_data = mem_data.expand_as(x_proj).clone()
            combined_input = x_proj + h_proj + mem_data
        else:
            combined_input = x_proj + h_proj
        
        # Process through co-workers
        co_worker_outputs = []
        for i, worker in enumerate(self.co_workers):
            co_out = worker(combined_input)
            co_worker_outputs.append(co_out)
        
        # Combine co-worker outputs
        merged_output = torch.cat(co_worker_outputs, dim=-1)
        output = torch.tanh(self.output_layer(merged_output))
        
        # Memory consolidation and validation
        if memory_key:
            self.flash_attn.write_memory(memory_key, output)  # Write to memory
        valid = torch.sigmoid(self.validator(output))
        
        if valid.mean().item() < 0.5:  # Use mean for batch validation
            return None, hidden_state
        return output, output

# ---------------------------- Main Network ----------------------------
class MemoryAugmentedNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_co_workers=2):
        super().__init__()
        self.big_node = BigNode(input_dim, hidden_dim, num_co_workers)
        self.memory_key = "central_memory"
        
    def forward(self, x, hidden_state=None):
        output, new_hidden = self.big_node(x, hidden_state, self.memory_key)
        
        # Retry mechanism for invalid outputs
        if output is None:
            for _ in range(3):  # Max 3 retries
                output, new_hidden = self.big_node(x, hidden_state, self.memory_key)
                if output is not None:
                    break
            if output is None:  # Fallback to input projection
                output = self.big_node.input_proj(x)
                new_hidden = output
        
        # Clean working memory after processing
        if self.memory_key:
            self.big_node.flash_attn.remove_memory(self.memory_key)
        return output, new_hidden
