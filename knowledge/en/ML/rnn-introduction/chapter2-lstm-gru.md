---
title: "Chapter 2: LSTM and GRU (Long Short-Term Memory and Gated Recurrent Unit)"
chapter_title: "Chapter 2: LSTM and GRU (Long Short-Term Memory and Gated Recurrent Unit)"
subtitle: Theory and implementation of gated RNN architectures for handling long-term dependencies
reading_time: 25-30 minutes
difficulty: Intermediate
code_examples: 8
exercises: 5
---

This chapter covers LSTM and GRU (Long Short. You will learn limitations of Vanilla RNN (vanishing, LSTM's cell state, and GRU architecture.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the limitations of Vanilla RNN (vanishing and exploding gradient problems)
  * ✅ Explain LSTM's cell state and gate mechanisms (forget, input, and output gates)
  * ✅ Understand GRU architecture and its differences from LSTM
  * ✅ Implement LSTM and GRU in PyTorch and apply them to real-world problems
  * ✅ Understand the mechanism and advantages of Bidirectional RNN
  * ✅ Build practical LSTM models for IMDb sentiment analysis tasks

* * *

## 2.1 Limitations of Vanilla RNN

### Vanishing and Exploding Gradient Problems

The standard RNN (Vanilla RNN) learned in Chapter 1 can theoretically handle sequences of arbitrary length, but in practice it struggles to learn **long-term dependencies**.

In RNN's BPTT (Backpropagation Through Time), gradients at time $t$ propagate backward through time as follows:

$$ \frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}} $$ 

The product term is the problem:

  * **Vanishing gradients** : When $\left\|\frac{\partial h_t}{\partial h_{t-1}}\right\| < 1$, gradients decrease exponentially as time steps increase
  * **Exploding gradients** : When $\left\|\frac{\partial h_t}{\partial h_{t-1}}\right\| > 1$, gradients increase exponentially

> "Due to vanishing gradients, Vanilla RNN can barely learn long-term dependencies beyond 10 steps"

### Visualizing the Problem
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Visualizing the Problem
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Observe gradient magnitude in Vanilla RNN
    class SimpleRNN(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(SimpleRNN, self).__init__()
            self.hidden_size = hidden_size
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    
        def forward(self, x):
            output, hidden = self.rnn(x)
            return output, hidden
    
    # Create model
    input_size = 10
    hidden_size = 20
    sequence_length = 50
    
    model = SimpleRNN(input_size, hidden_size)
    
    # Random input
    x = torch.randn(1, sequence_length, input_size, requires_grad=True)
    
    # Forward pass
    output, hidden = model(x)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Calculate gradient norm at each time step
    gradients = []
    for t in range(sequence_length):
        grad = x.grad[0, t, :].norm().item()
        gradients.append(grad)
    
    print("=== Gradient Propagation in Vanilla RNN ===")
    print(f"Gradient norm at initial time: {gradients[0]:.6f}")
    print(f"Gradient norm at middle time: {gradients[25]:.6f}")
    print(f"Gradient norm at final time: {gradients[49]:.6f}")
    print(f"\nGradient decay rate: {gradients[0] / gradients[49]:.2f}x")
    print("→ Gradients get smaller as we go back in time (vanishing gradients)")
    

### Concrete Example: Long-Term Dependency Task
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    
    # Task: Predict the first element at the last time step
    def create_long_dependency_task(batch_size=32, seq_length=50):
        """
        Task where important information is at the first time step and needs to be used at the end
        Example: [5, 0, 0, ..., 0] → predict 5 at the end
        """
        x = torch.zeros(batch_size, seq_length, 10)
        targets = torch.randint(0, 10, (batch_size,))
    
        # Encode correct label at first time step
        for i in range(batch_size):
            x[i, 0, targets[i]] = 1.0
    
        return x, targets
    
    # Train with Vanilla RNN
    class VanillaRNNClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(VanillaRNNClassifier, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
    
        def forward(self, x):
            output, hidden = self.rnn(x)
            # Use output at last time step
            logits = self.fc(output[:, -1, :])
            return logits
    
    # Experiment
    model = VanillaRNNClassifier(input_size=10, hidden_size=32, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    num_epochs = 100
    for epoch in range(num_epochs):
        x, targets = create_long_dependency_task(batch_size=32, seq_length=50)
    
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
    
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                x_test, targets_test = create_long_dependency_task(batch_size=100, seq_length=50)
                logits_test = model(x_test)
                _, predicted = logits_test.max(1)
                accuracy = (predicted == targets_test).float().mean().item()
                print(f"Epoch {epoch+1}: Accuracy = {accuracy*100:.2f}%")
    
    print("\n→ Vanilla RNN cannot learn long-term dependencies and has low accuracy (similar to random prediction)")
    

### Solution: Gate Mechanisms

To solve this problem, **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** were proposed. They can effectively learn long-term dependencies by controlling information flow through **gate mechanisms**.

* * *

## 2.2 LSTM (Long Short-Term Memory)

### Overview of LSTM

**LSTM** is a gated RNN architecture proposed by Hochreiter and Schmidhuber in 1997. It is characterized by having a **cell state** for long-term memory in addition to the hidden state of Vanilla RNN.
    
    
    ```mermaid
    graph LR
        A["Input x_t"] --> B["LSTM Cell"]
        C["Previous hidden state h_{t-1}"] --> B
        D["Previous cell state c_{t-1}"] --> B
    
        B --> E["Output h_t"]
        B --> F["New cell state c_t"]
    
        style A fill:#e1f5ff
        style B fill:#b3e5fc
        style D fill:#fff9c4
        style E fill:#4fc3f7
        style F fill:#ffeb3b
    ```

### Four Components of LSTM

An LSTM cell consists of the following four elements:

Gate | Role | Formula  
---|---|---  
**Forget Gate** | How much past memory to retain | $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$  
**Input Gate** | How much new information to add | $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$  
**Candidate Cell** | Content of new information to add | $\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$  
**Output Gate** | How much to output from cell state | $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$  
  
### Mathematical Definition of LSTM

The complete LSTM update equations are as follows:

$$ \begin{align} f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) \quad &\text{(Forget gate)} \\\ i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) \quad &\text{(Input gate)} \\\ \tilde{C}_t &= \tanh(W_C [h_{t-1}, x_t] + b_C) \quad &\text{(Candidate value)} \\\ C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad &\text{(Cell state update)} \\\ o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) \quad &\text{(Output gate)} \\\ h_t &= o_t \odot \tanh(C_t) \quad &\text{(Hidden state update)} \end{align} $$ 

Where:

  * $\sigma$: Sigmoid function (outputs values between 0 and 1, used for gate control)
  * $\odot$: Element-wise product (Hadamard product)
  * $[h_{t-1}, x_t]$: Vector concatenation
  * $W_*, b_*$: Learnable parameters

### Role of Cell State

The cell state $C_t$ functions as an information highway:

$$ C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t $$ 

  * $f_t \odot C_{t-1}$: Selectively retain past memory (controlled by forget gate)
  * $i_t \odot \tilde{C}_t$: Selectively add new information (controlled by input gate)

> "Gradients flow directly through the cell state, making vanishing gradients less likely!"

### Manual Implementation of LSTM
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class LSTMCell(nn.Module):
        """Manual implementation of LSTM cell (for educational purposes)"""
        def __init__(self, input_size, hidden_size):
            super(LSTMCell, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
    
            # Forget gate
            self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
            # Input gate
            self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
            # Candidate value
            self.W_C = nn.Linear(input_size + hidden_size, hidden_size)
            # Output gate
            self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
    
        def forward(self, x_t, h_prev, C_prev):
            """
            x_t: (batch_size, input_size) - Current input
            h_prev: (batch_size, hidden_size) - Previous hidden state
            C_prev: (batch_size, hidden_size) - Previous cell state
            """
            # Concatenate input and hidden state
            combined = torch.cat([h_prev, x_t], dim=1)
    
            # Forget gate: which information to forget
            f_t = torch.sigmoid(self.W_f(combined))
    
            # Input gate: which information to add
            i_t = torch.sigmoid(self.W_i(combined))
    
            # Candidate value: content of information to add
            C_tilde = torch.tanh(self.W_C(combined))
    
            # Cell state update
            C_t = f_t * C_prev + i_t * C_tilde
    
            # Output gate: which information to output
            o_t = torch.sigmoid(self.W_o(combined))
    
            # Hidden state update
            h_t = o_t * torch.tanh(C_t)
    
            return h_t, C_t
    
    
    class ManualLSTM(nn.Module):
        """LSTM processing over multiple time steps"""
        def __init__(self, input_size, hidden_size):
            super(ManualLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.cell = LSTMCell(input_size, hidden_size)
    
        def forward(self, x, init_states=None):
            """
            x: (batch_size, seq_length, input_size)
            """
            batch_size, seq_length, _ = x.size()
    
            # Initial state
            if init_states is None:
                h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
                C_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            else:
                h_t, C_t = init_states
    
            # Store outputs at each time step
            outputs = []
    
            # Process sequence time step by time step
            for t in range(seq_length):
                h_t, C_t = self.cell(x[:, t, :], h_t, C_t)
                outputs.append(h_t.unsqueeze(1))
    
            # Concatenate outputs
            outputs = torch.cat(outputs, dim=1)
    
            return outputs, (h_t, C_t)
    
    
    # Operation check
    batch_size = 4
    seq_length = 10
    input_size = 8
    hidden_size = 16
    
    model = ManualLSTM(input_size, hidden_size)
    x = torch.randn(batch_size, seq_length, input_size)
    
    outputs, (h_final, C_final) = model(x)
    
    print("=== Manual LSTM Implementation Check ===")
    print(f"Input size: {x.shape}")
    print(f"Output size: {outputs.shape}")
    print(f"Final hidden state: {h_final.shape}")
    print(f"Final cell state: {C_final.shape}")
    

### Using PyTorch's nn.LSTM

In actual development, we use PyTorch's optimized `nn.LSTM`:
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: In actual development, we use PyTorch's optimizednn.LSTM:
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # PyTorch's LSTM
    lstm = nn.LSTM(
        input_size=10,      # Input dimension
        hidden_size=20,     # Hidden state dimension
        num_layers=2,       # Number of LSTM layers
        batch_first=True,   # (batch, seq, feature) order
        dropout=0.2,        # Dropout between layers
        bidirectional=False # Whether bidirectional
    )
    
    # Dummy data
    batch_size = 32
    seq_length = 15
    input_size = 10
    
    x = torch.randn(batch_size, seq_length, input_size)
    
    # Forward pass
    output, (h_n, c_n) = lstm(x)
    
    print("=== Using PyTorch nn.LSTM ===")
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}  # (batch, seq, hidden_size)")
    print(f"Final hidden state: {h_n.shape}  # (num_layers, batch, hidden_size)")
    print(f"Final cell state: {c_n.shape}  # (num_layers, batch, hidden_size)")
    
    # Check number of parameters
    total_params = sum(p.numel() for p in lstm.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("→ Many parameters because each layer has 4 gates (f, i, C, o)")
    

### LSTM and Long-Term Dependencies
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: LSTM and Long-Term Dependencies
    
    Purpose: Demonstrate optimization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Solve the previous long-term dependency task with LSTM
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(LSTMClassifier, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
    
        def forward(self, x):
            output, (h_n, c_n) = self.lstm(x)
            # Use output at last time step
            logits = self.fc(output[:, -1, :])
            return logits
    
    # Task creation function (same as before)
    def create_long_dependency_task(batch_size=32, seq_length=50):
        x = torch.zeros(batch_size, seq_length, 10)
        targets = torch.randint(0, 10, (batch_size,))
        for i in range(batch_size):
            x[i, 0, targets[i]] = 1.0
        return x, targets
    
    # Train with LSTM
    model = LSTMClassifier(input_size=10, hidden_size=32, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("=== Learning Long-Term Dependencies with LSTM ===")
    num_epochs = 100
    for epoch in range(num_epochs):
        x, targets = create_long_dependency_task(batch_size=32, seq_length=50)
    
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
    
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                x_test, targets_test = create_long_dependency_task(batch_size=100, seq_length=50)
                logits_test = model(x_test)
                _, predicted = logits_test.max(1)
                accuracy = (predicted == targets_test).float().mean().item()
                print(f"Epoch {epoch+1}: Accuracy = {accuracy*100:.2f}%")
    
    print("\n→ LSTM can effectively learn long-term dependencies and achieve high accuracy!")
    

* * *

## 2.3 GRU (Gated Recurrent Unit)

### Overview of GRU

**GRU (Gated Recurrent Unit)** is a simplified architecture of LSTM proposed by Cho et al. in 2014. It often achieves equal or better performance than LSTM with fewer parameters.

### Differences Between LSTM and GRU

Item | LSTM | GRU  
---|---|---  
**Number of gates** | 3 (forget, input, output) | 2 (reset, update)  
**States** | Hidden state $h_t$ and cell state $C_t$ | Hidden state $h_t$ only  
**Parameters** | More | Fewer (about 75% of LSTM)  
**Computation speed** | Somewhat slower | Somewhat faster  
**Performance** | Task dependent | Task dependent (advantageous for short sequences)  
  
### Mathematical Definition of GRU

The GRU update equations are as follows:

$$ \begin{align} r_t &= \sigma(W_r [h_{t-1}, x_t] + b_r) \quad &\text{(Reset gate)} \\\ z_t &= \sigma(W_z [h_{t-1}, x_t] + b_z) \quad &\text{(Update gate)} \\\ \tilde{h}_t &= \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h) \quad &\text{(Candidate hidden state)} \\\ h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad &\text{(Hidden state update)} \end{align} $$ 

Role of each gate:

  * **Reset gate $r_t$** : How much to ignore past information (ignores past when close to 0)
  * **Update gate $z_t$** : How much to mix past and present information (retains past when close to 0, adopts new information when close to 1)

> "GRU integrates LSTM's forget gate and input gate with the update gate $z_t$"

### Manual Implementation of GRU
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    
    class GRUCell(nn.Module):
        """Manual implementation of GRU cell"""
        def __init__(self, input_size, hidden_size):
            super(GRUCell, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
    
            # Reset gate
            self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
            # Update gate
            self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
            # Candidate hidden state
            self.W_h = nn.Linear(input_size + hidden_size, hidden_size)
    
        def forward(self, x_t, h_prev):
            """
            x_t: (batch_size, input_size)
            h_prev: (batch_size, hidden_size)
            """
            # Concatenate input and hidden state
            combined = torch.cat([h_prev, x_t], dim=1)
    
            # Reset gate
            r_t = torch.sigmoid(self.W_r(combined))
    
            # Update gate
            z_t = torch.sigmoid(self.W_z(combined))
    
            # Candidate hidden state (filter past with reset gate)
            combined_reset = torch.cat([r_t * h_prev, x_t], dim=1)
            h_tilde = torch.tanh(self.W_h(combined_reset))
    
            # Hidden state update (mix past and present with update gate)
            h_t = (1 - z_t) * h_prev + z_t * h_tilde
    
            return h_t
    
    
    class ManualGRU(nn.Module):
        """GRU processing over multiple time steps"""
        def __init__(self, input_size, hidden_size):
            super(ManualGRU, self).__init__()
            self.hidden_size = hidden_size
            self.cell = GRUCell(input_size, hidden_size)
    
        def forward(self, x, init_state=None):
            batch_size, seq_length, _ = x.size()
    
            # Initial state
            if init_state is None:
                h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            else:
                h_t = init_state
    
            outputs = []
    
            for t in range(seq_length):
                h_t = self.cell(x[:, t, :], h_t)
                outputs.append(h_t.unsqueeze(1))
    
            outputs = torch.cat(outputs, dim=1)
            return outputs, h_t
    
    
    # Operation check
    model = ManualGRU(input_size=8, hidden_size=16)
    x = torch.randn(4, 10, 8)
    
    outputs, h_final = model(x)
    
    print("=== Manual GRU Implementation Check ===")
    print(f"Input: {x.shape}")
    print(f"Output: {outputs.shape}")
    print(f"Final hidden state: {h_final.shape}")
    print("→ GRU has no cell state, only hidden state")
    

### Using PyTorch's nn.GRU
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Using PyTorch's nn.GRU
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # PyTorch's GRU
    gru = nn.GRU(
        input_size=10,
        hidden_size=20,
        num_layers=2,
        batch_first=True,
        dropout=0.2,
        bidirectional=False
    )
    
    x = torch.randn(32, 15, 10)
    output, h_n = gru(x)
    
    print("=== Using PyTorch nn.GRU ===")
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Final hidden state: {h_n.shape}")
    
    # Parameter count comparison with LSTM
    lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
    gru_params = sum(p.numel() for p in gru.parameters())
    lstm_params = sum(p.numel() for p in lstm.parameters())
    
    print(f"\nGRU parameters: {gru_params:,}")
    print(f"LSTM parameters: {lstm_params:,}")
    print(f"Difference: {lstm_params - gru_params:,} (GRU has {(lstm_params/gru_params - 1)*100:.1f}% fewer)")
    

### Performance Comparison Between LSTM and GRU
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import time
    
    class SequenceClassifier(nn.Module):
        """Generic sequence classifier"""
        def __init__(self, input_size, hidden_size, num_classes, rnn_type='lstm'):
            super(SequenceClassifier, self).__init__()
    
            if rnn_type == 'lstm':
                self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
            elif rnn_type == 'gru':
                self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
            else:
                raise ValueError("rnn_type must be 'lstm' or 'gru'")
    
            self.fc = nn.Linear(hidden_size, num_classes)
            self.rnn_type = rnn_type
    
        def forward(self, x):
            if self.rnn_type == 'lstm':
                output, (h_n, c_n) = self.rnn(x)
            else:
                output, h_n = self.rnn(x)
    
            logits = self.fc(output[:, -1, :])
            return logits
    
    # Comparison experiment
    def compare_models(seq_length=50):
        print(f"\n=== Comparison with sequence length={seq_length} ===")
    
        # Create models
        lstm_model = SequenceClassifier(10, 32, 10, rnn_type='lstm')
        gru_model = SequenceClassifier(10, 32, 10, rnn_type='gru')
    
        # Generate data
        x, targets = create_long_dependency_task(batch_size=32, seq_length=seq_length)
    
        criterion = nn.CrossEntropyLoss()
    
        # LSTM training
        optimizer_lstm = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
        start = time.time()
        for _ in range(50):
            optimizer_lstm.zero_grad()
            logits = lstm_model(x)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer_lstm.step()
        lstm_time = time.time() - start
    
        # GRU training
        optimizer_gru = torch.optim.Adam(gru_model.parameters(), lr=0.001)
        start = time.time()
        for _ in range(50):
            optimizer_gru.zero_grad()
            logits = gru_model(x)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer_gru.step()
        gru_time = time.time() - start
    
        # Accuracy evaluation
        x_test, targets_test = create_long_dependency_task(batch_size=100, seq_length=seq_length)
    
        with torch.no_grad():
            logits_lstm = lstm_model(x_test)
            logits_gru = gru_model(x_test)
    
            _, pred_lstm = logits_lstm.max(1)
            _, pred_gru = logits_gru.max(1)
    
            acc_lstm = (pred_lstm == targets_test).float().mean().item()
            acc_gru = (pred_gru == targets_test).float().mean().item()
    
        print(f"LSTM - Accuracy: {acc_lstm*100:.2f}%, Training time: {lstm_time:.2f}s")
        print(f"GRU  - Accuracy: {acc_gru*100:.2f}%, Training time: {gru_time:.2f}s")
    
    # Compare at different sequence lengths
    compare_models(seq_length=20)
    compare_models(seq_length=50)
    compare_models(seq_length=100)
    
    print("\n→ GRU tends to be more efficient for short sequences, LSTM advantageous for longer sequences")
    

* * *

## 2.4 Bidirectional RNN

### What is Bidirectional RNN?

**Bidirectional RNN** processes sequences from both forward (front to back) and backward (back to front) directions and integrates information from both directions.
    
    
    ```mermaid
    graph LR
        A["x_1"] --> B["Forward→"]
        B --> C["x_2"]
        C --> D["Forward→"]
        D --> E["x_3"]
    
        E --> F["Backward←"]
        F --> C
        C --> G["Backward←"]
        G --> A
    
        B --> H["h_1"]
        D --> I["h_2"]
        F --> J["h_3 (backward)"]
        G --> K["h_2 (backward)"]
    
        style B fill:#b3e5fc
        style D fill:#b3e5fc
        style F fill:#ffab91
        style G fill:#ffab91
    ```

### Advantages of Bidirectional RNN

  * **Complete context capture** : Can consider both past and future context at each position
  * **Part-of-speech tagging** : Determine parts of speech by looking at before and after words
  * **Sentiment analysis** : Judge sentiment by looking at entire sentence
  * **Machine translation** : Use as encoder

> "Bidirectional RNN cannot be used for real-time processing because the output at time $t$ depends on future information. It is suitable for offline processing (when entire sequence is available)."

### Implementation of Bidirectional LSTM
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Implementation of Bidirectional LSTM
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # In PyTorch, just specify bidirectional=True
    class BidirectionalLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(BidirectionalLSTM, self).__init__()
    
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                batch_first=True,
                bidirectional=True  # Enable bidirectional
            )
    
            # hidden_size * 2 because bidirectional
            self.fc = nn.Linear(hidden_size * 2, num_classes)
    
        def forward(self, x):
            # output: (batch, seq, hidden_size * 2)
            output, (h_n, c_n) = self.lstm(x)
    
            # Use output at last time step
            logits = self.fc(output[:, -1, :])
            return logits
    
    # Operation check
    model = BidirectionalLSTM(input_size=10, hidden_size=32, num_classes=10)
    x = torch.randn(4, 15, 10)
    
    logits = model(x)
    
    print("=== Bidirectional LSTM Operation Check ===")
    print(f"Input: {x.shape}")
    print(f"Output: {logits.shape}")
    
    # Parameter count comparison
    uni_lstm = nn.LSTM(10, 32, batch_first=True, bidirectional=False)
    bi_lstm = nn.LSTM(10, 32, batch_first=True, bidirectional=True)
    
    uni_params = sum(p.numel() for p in uni_lstm.parameters())
    bi_params = sum(p.numel() for p in bi_lstm.parameters())
    
    print(f"\nUnidirectional LSTM: {uni_params:,} parameters")
    print(f"Bidirectional LSTM: {bi_params:,} parameters")
    print(f"→ Bidirectional has about 2x the parameters")
    

### Performance Comparison: Bidirectional vs Unidirectional
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Performance Comparison: Bidirectional vs Unidirectional
    
    Purpose: Demonstrate optimization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    class DirectionalClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, bidirectional=False):
            super(DirectionalClassifier, self).__init__()
    
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                batch_first=True,
                bidirectional=bidirectional
            )
    
            fc_input_size = hidden_size * 2 if bidirectional else hidden_size
            self.fc = nn.Linear(fc_input_size, num_classes)
    
        def forward(self, x):
            output, _ = self.lstm(x)
            logits = self.fc(output[:, -1, :])
            return logits
    
    # Comparison experiment
    def compare_directionality():
        print("\n=== Unidirectional vs Bidirectional Comparison ===")
    
        # Create models
        uni_model = DirectionalClassifier(10, 32, 10, bidirectional=False)
        bi_model = DirectionalClassifier(10, 32, 10, bidirectional=True)
    
        criterion = nn.CrossEntropyLoss()
    
        # Training
        for model, name in [(uni_model, "Unidirectional"), (bi_model, "Bidirectional")]:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
            for epoch in range(100):
                x, targets = create_long_dependency_task(batch_size=32, seq_length=50)
    
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
    
            # Evaluation
            x_test, targets_test = create_long_dependency_task(batch_size=100, seq_length=50)
            with torch.no_grad():
                logits_test = model(x_test)
                _, predicted = logits_test.max(1)
                accuracy = (predicted == targets_test).float().mean().item()
    
            print(f"{name} LSTM - Accuracy: {accuracy*100:.2f}%")
    
    compare_directionality()
    print("\n→ In this task, information is at the beginning, so bidirectional advantage is small")
    print("  Bidirectional is advantageous for tasks where both past and future context are important, such as part-of-speech tagging")
    

* * *

## 2.5 Practice: IMDb Sentiment Analysis

### IMDb Dataset

**IMDb (Internet Movie Database)** is a sentiment analysis dataset of movie reviews:

  * 50,000 movie reviews (25,000 training, 25,000 test)
  * 2-class classification: Positive, Negative
  * Each review is English text

### Data Preparation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Data Preparation
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchtext.datasets import IMDB
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
    from collections import Counter
    
    # Tokenizer
    tokenizer = get_tokenizer('basic_english')
    
    # Load dataset
    train_iter = IMDB(split='train')
    
    # Build vocabulary
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
    
    # Build vocabulary (top 10,000 most frequent words)
    vocab = build_vocab_from_iterator(
        yield_tokens(IMDB(split='train')),
        specials=['<unk>', '<pad>'],
        max_tokens=10000
    )
    vocab.set_default_index(vocab['<unk>'])
    
    print("=== IMDb Dataset Preparation ===")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"<pad> token index: {vocab['<pad>']}")
    print(f"<unk> token index: {vocab['<unk>']}")
    
    # Sample tokenization
    sample_text = "This movie is great!"
    tokens = tokenizer(sample_text)
    indices = [vocab[token] for token in tokens]
    print(f"\nSample: '{sample_text}'")
    print(f"Tokens: {tokens}")
    print(f"Indices: {indices}")
    </unk></unk></pad></pad></unk></pad></unk>

### Dataset Class
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Dataset Class
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence
    
    class IMDbDataset(Dataset):
        def __init__(self, split='train'):
            self.data = list(IMDB(split=split))
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            label, text = self.data[idx]
    
            # Convert label to number (neg=0, pos=1)
            label = 1 if label == 'pos' else 0
    
            # Tokenize text and convert to indices
            tokens = tokenizer(text)
            indices = [vocab[token] for token in tokens]
    
            return torch.tensor(indices), torch.tensor(label)
    
    def collate_batch(batch):
        """
        Pad sequences in batch to same length
        """
        texts, labels = zip(*batch)
    
        # Padding
        texts_padded = pad_sequence(texts, batch_first=True, padding_value=vocab['<pad>'])
        labels = torch.stack(labels)
    
        return texts_padded, labels
    
    # Create data loaders
    train_dataset = IMDbDataset(split='train')
    test_dataset = IMDbDataset(split='test')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_batch
    )
    
    print("\n=== Data Loader Check ===")
    texts, labels = next(iter(train_loader))
    print(f"Batch size: {texts.shape[0]}")
    print(f"Sequence length (max): {texts.shape[1]}")
    print(f"Labels: {labels[:5]}")
    </pad>

### LSTM Sentiment Analysis Model
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: LSTM Sentiment Analysis Model
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    class LSTMSentimentClassifier(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
            super(LSTMSentimentClassifier, self).__init__()
    
            # Embedding layer
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab['<pad>'])
    
            # LSTM layer
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
    
            # Classification layer
            self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 because bidirectional
    
            # Dropout
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, text):
            # text: (batch, seq_len)
    
            # Embedding: (batch, seq_len, embedding_dim)
            embedded = self.dropout(self.embedding(text))
    
            # LSTM: output (batch, seq_len, hidden_dim * 2)
            output, (hidden, cell) = self.lstm(embedded)
    
            # Use output at last time step
            # Or concatenate final hidden states of forward and backward
            # hidden: (num_layers * 2, batch, hidden_dim)
    
            # Concatenate forward and backward of final layer
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)
    
            # Dropout + classification
            hidden_concat = self.dropout(hidden_concat)
            logits = self.fc(hidden_concat)
    
            return logits
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = LSTMSentimentClassifier(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=256,
        num_classes=2,
        num_layers=2,
        dropout=0.5
    ).to(device)
    
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    </pad>

### Training Loop
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Training Loop
    
    Purpose: Demonstrate optimization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
    
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
    
            optimizer.zero_grad()
            logits = model(texts)
            loss = criterion(logits, labels)
    
            loss.backward()
            # Gradient clipping (prevent gradient explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def test_epoch(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
    
        with torch.no_grad():
            for texts, labels in loader:
                texts, labels = texts.to(device), labels.to(device)
    
                logits = model(texts)
                loss = criterion(logits, labels)
    
                running_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
    
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    # Execute training
    num_epochs = 5
    best_acc = 0
    
    print("\n=== Training Start ===")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
    
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_imdb_lstm.pth')
    
    print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")
    

### Inference and Interpretation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    
    def predict_sentiment(model, text, vocab, tokenizer, device):
        """Predict sentiment of single text"""
        model.eval()
    
        # Tokenize
        tokens = tokenizer(text)
        indices = [vocab[token] for token in tokens]
    
        # Convert to tensor
        text_tensor = torch.tensor(indices).unsqueeze(0).to(device)  # (1, seq_len)
    
        # Predict
        with torch.no_grad():
            logits = model(text_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(1).item()
    
        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = probs[0, pred].item()
    
        return sentiment, confidence
    
    # Test
    test_reviews = [
        "This movie is absolutely amazing! I loved every moment.",
        "Terrible film. Waste of time and money.",
        "It was okay, nothing special but not bad either.",
        "One of the best movies I've ever seen!",
        "Boring and predictable. Would not recommend."
    ]
    
    print("\n=== Sentiment Analysis Prediction Results ===")
    for review in test_reviews:
        sentiment, confidence = predict_sentiment(model, review, vocab, tokenizer, device)
        print(f"\nReview: {review}")
        print(f"Prediction: {sentiment} (Confidence: {confidence*100:.2f}%)")
    

* * *

## 2.6 Guidelines for Choosing Between LSTM and GRU

### Selection Criteria

Situation | Recommendation | Reason  
---|---|---  
Long sequences (>100) | LSTM | Cell state retains long-term memory  
Short sequences (<50) | GRU | Fewer parameters, more efficient  
Computational constraints | GRU | About 25% fewer parameters  
High accuracy essential | LSTM | Higher expressive power  
Real-time processing | GRU | Faster computation  
Both contexts needed | Bidirectional LSTM/GRU | Utilizes information from both directions  
Uncertain | Try both | High task dependency  
  
### Choosing Hyperparameters

  * **Hidden layer size** : 64-512 (depending on task complexity)
  * **Number of layers** : 1-3 layers (too deep risks overfitting)
  * **Dropout** : 0.2-0.5 (prevent overfitting)
  * **Embedding dimension** : 50-300 (depending on vocabulary size)
  * **Learning rate** : 0.0001-0.001 (Adam recommended)
  * **Batch size** : 32-128 (depending on memory)

### Implementation Best Practices
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    
    class BestPracticeLSTM(nn.Module):
        """LSTM model incorporating best practices"""
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
            super(BestPracticeLSTM, self).__init__()
    
            # 1. Specify padding_idx in Embedding layer
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    
            # 2. Bidirectional LSTM
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.3,  # Inter-layer Dropout
                bidirectional=True
            )
    
            # 3. Batch Normalization (optional)
            self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
    
            # 4. Dropout
            self.dropout = nn.Dropout(0.5)
    
            # 5. Classification layer
            self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
        def forward(self, x):
            embedded = self.dropout(self.embedding(x))
            output, (hidden, cell) = self.lstm(embedded)
    
            # Concatenate final hidden states of forward and backward
            hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
    
            # Batch Norm (optional)
            hidden_concat = self.batch_norm(hidden_concat)
    
            # Dropout + classification
            hidden_concat = self.dropout(hidden_concat)
            logits = self.fc(hidden_concat)
    
            return logits
    
    # Training considerations
    def train_with_best_practices(model, train_loader, num_epochs=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
    
        for epoch in range(num_epochs):
            model.train()
            for texts, labels in train_loader:
                optimizer.zero_grad()
                logits = model(texts)
                loss = criterion(logits, labels)
                loss.backward()
    
                # Gradient clipping (essential)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    
                optimizer.step()
    
            # Adjust learning rate based on validation loss
            val_loss = evaluate(model, val_loader)
            scheduler.step(val_loss)
    
    print("=== Best Practices ===")
    print("1. Specify padding_idx to exclude <pad> from training")
    print("2. Bidirectional LSTM for complete context capture")
    print("3. Dropout to prevent overfitting")
    print("4. Gradient clipping to prevent gradient explosion")
    print("5. Learning rate scheduler to improve optimization")
    </pad>

* * *

## Exercises

**Exercise 1: Observe LSTM Gate Operations**

Visualize the values of each LSTM gate (forget, input, output) and observe how they control information.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Visualize the values of each LSTM gate (forget, input, outpu
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Exercise: Plot each gate's values over time for simple sequence data
    # Hint: Record f_t, i_t, o_t values and graph with matplotlib
    

**Exercise 2: Compare Convergence Speed of GRU and LSTM**

Train GRU and LSTM on the same task and compare training curves (loss and accuracy).
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Train GRU and LSTM on the same task and compare training cur
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    
    # Exercise: Create GRU and LSTM models
    # Exercise: Train on same data and record loss and accuracy for each epoch
    # Exercise: Plot training curves
    # Evaluation metrics: convergence speed, final accuracy, training time
    

**Exercise 3: Verify Effect of Bidirectional RNN**

Compare performance of unidirectional and bidirectional RNN on part-of-speech tagging task.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Compare performance of unidirectional and bidirectional RNN 
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Exercise: Implement unidirectional and bidirectional LSTM models
    # Exercise: Compare performance on part-of-speech tagging task (predict part of speech for each word)
    # Hint: Use datasets like UD_English from torchtext.datasets
    # Verify bidirectional advantage on tasks where both past and future context are important
    

**Exercise 4: Relationship Between Sequence Length and Performance**

Compare LSTM and GRU performance at different sequence lengths (10, 50, 100, 200) and determine which is stronger at long-term dependencies.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Compare LSTM and GRU performance at different sequence lengt
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Exercise: Compare accuracy with LSTM and GRU
    # Exercise: Create graph of sequence length vs accuracy
    # Analyze at what sequence length performance differences become significant
    

**Exercise 5: Improve IMDb Sentiment Analysis**

Improve the basic LSTM model to increase test accuracy.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Improve the basic LSTM model to increase test accuracy.
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # 1. Use pre-trained embeddings (GloVe, Word2Vec)
    # 2. Add Attention mechanism
    # 3. Adjust number of layers and hidden_size
    # 4. Data Augmentation (back-translation, etc.)
    # 5. Ensemble learning
    
    # Goal: Improve accuracy by +2% or more from baseline
    

* * *

## Summary

In this chapter, we learned about LSTM, GRU, and their applications.

### Key Points

  * **Vanilla RNN limitations** : Difficulty learning long-term dependencies due to vanishing/exploding gradients
  * **LSTM** : Realizes long-term memory with cell state and gate mechanisms (forget, input, output)
  * **GRU** : Simplified LSTM, operates efficiently with 2 gates (reset, update)
  * **LSTM vs GRU differences** : Trade-offs in parameter count, computation speed, and performance
  * **Bidirectional RNN** : Processes from both directions to fully capture context
  * **Practice** : Applied to real-world NLP task with IMDb sentiment analysis
  * **Best practices** : Gradient clipping, Dropout, learning rate scheduling

### Next Steps

In the next chapter, we will learn about **Sequence-to-Sequence (Seq2Seq)** and **Attention mechanisms**. You will master techniques essential for sequence transformation tasks such as machine translation and summarization.
