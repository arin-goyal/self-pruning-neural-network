import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: Setup and Data Loading
# ==========================================

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 64

# Transformations for CIFAR-10 images
# We convert images to PyTorch Tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

# Load the test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

print(f"Number of training batches: {len(trainloader)}")
print(f"Number of test batches: {len(testloader)}")

# --- End of Step 1 ---

# ==========================================
# STEP 2: The PrunableLinear Layer
# ==========================================

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. Standard weight and bias parameters
        # We initialize them similarly to how nn.Linear does it
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # 2. The learnable gate_scores parameter
        # It has the exact same shape as the weight tensor.
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization for weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Initialize bias
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Initialize gate_scores to be slightly positive
        # This ensures sigmoid(gate_scores) starts close to 1,
        # meaning the network starts out almost fully connected.
        nn.init.normal_(self.gate_scores, mean=1.5, std=0.1)

    def forward(self, x):
        # 3a. Apply Sigmoid to gate_scores to turn them into gates (values between 0 and 1)
        gates = torch.sigmoid(self.gate_scores)
        
        # 3b. Calculate pruned weights by element-wise multiplication
        pruned_weights = self.weight * gates
        
        # 3c. Perform standard linear layer operation using the pruned weights
        return nn.functional.linear(x, pruned_weights, self.bias)

# --- End of Step 2 ---

# ==========================================
# STEP 3: Building the Network Architecture
# ==========================================

class PrunableNet(nn.Module):
    def __init__(self):
        super(PrunableNet, self).__init__()
        # CIFAR-10 images are 3 channels, 32x32 pixels
        # Flattened size = 3 * 32 * 32 = 3072
        
        # We use our custom PrunableLinear instead of standard nn.Linear
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10) # 10 output classes for CIFAR-10

    def forward(self, x):
        # Flatten the image: (batch_size, 3, 32, 32) -> (batch_size, 3072)
        x = x.view(x.size(0), -1)
        
        # Pass through layers with ReLU activations
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Output layer (no activation here, CrossEntropyLoss handles softmax)
        x = self.fc3(x)
        return x

# Instantiate the model and move it to the configured device
model = PrunableNet().to(device)
print("\n--- Network Architecture ---")
print(model)

# --- End of Step 3 ---

# ==========================================
# STEP 4: Loss Formulation
# ==========================================

# Standard classification loss for multi-class problems like CIFAR-10
criterion = nn.CrossEntropyLoss()

def compute_sparsity_loss(model):
    """
    Calculates the L1 penalty on the gate values to encourage sparsity.
    We iterate through all modules in the network, find the PrunableLinear
    layers, apply sigmoid to their gate_scores to get the actual gate values (0 to 1),
    and sum them up. Since gate values are positive, L1 norm is just the sum.
    """
    l1_loss = 0.0
    for module in model.modules():
        # Check if the layer is our custom prunable layer
        if isinstance(module, PrunableLinear):
            # Calculate actual gate values from scores
            gates = torch.sigmoid(module.gate_scores)
            # Add the sum of these gates to our penalty
            l1_loss += torch.sum(gates)
    return l1_loss

# --- End of Step 4 ---

# ==========================================
# STEP 5: Training and Evaluation Loops
# ==========================================

def train_and_evaluate(lambda_val, epochs=5, lr=0.001, sparsity_threshold=1e-2):
    """
    Trains a fresh model from scratch using the specified lambda value.
    Returns the final accuracy, sparsity level, and the trained model.
    """
    print(f"\n{'='*40}")
    print(f"Starting experiment with Lambda = {lambda_val}")
    print(f"{'='*40}")
    
    # 1. Initialize a fresh model
    model = PrunableNet().to(device)
    
    # 2. Setup Optimizer
    # We use Adam. To encourage pruning, we can use a slightly higher learning rate
    # for the gate_scores to help them move quickly if they need to.
    gate_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'gate_scores' in name:
            gate_params.append(param)
        else:
            other_params.append(param)
            
    optimizer = optim.Adam([
        {'params': other_params, 'lr': lr},
        {'params': gate_params, 'lr': 0.005}
    ])

    # 3. The Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            
            # Calculate standard classification loss
            class_loss = criterion(outputs, labels)
            
            # Calculate sparsity loss
            sparse_loss = compute_sparsity_loss(model)
            
            # TOTAL LOSS Formulation
            total_loss = class_loss + (lambda_val * sparse_loss)

            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Avg Total Loss: {running_loss / len(trainloader):.4f}")

    # 4. Evaluation Loop (Accuracy)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): # No gradients needed for evaluation
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    test_accuracy = 100 * correct / total
    
    # 5. Calculate Sparsity Level
    # What percentage of gates are basically zero?
    total_gates = 0
    pruned_gates = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                # Count total gates
                total_gates += gates.numel()
                # Count gates that are below our tiny threshold
                pruned_gates += (gates < sparsity_threshold).sum().item()
                
    sparsity_level = 100 * pruned_gates / total_gates
    
    print(f"Test Accuracy:  {test_accuracy:.2f}%")
    print(f"Sparsity Level: {sparsity_level:.2f}% (Gates < {sparsity_threshold})")
    
    return test_accuracy, sparsity_level, model

# --- End of Step 5 ---

# ==========================================
# STEP 6: Running Experiments
# ==========================================

if __name__ == '__main__':
    # We will test a baseline (no pruning) and two different penalty strengths
    lambda_values = [0.0, 0.0001, 0.001]
    
    results = []
    best_model = None
    best_lambda = 0.0
    highest_sparsity_with_good_acc = 0.0
    
    for l_val in lambda_values:
        # Train for 5 epochs for the sake of time.
        # In a real scenario, you'd train for 50-100 epochs.
        acc, sparsity, model = train_and_evaluate(lambda_val=l_val, epochs=5)
        
        results.append({
            'Lambda': l_val,
            'Test Accuracy': acc,
            'Sparsity Level (%)': sparsity
        })
        
        # Heuristic to save a "best" model: good sparsity and decent accuracy
        # We will use this best model to plot the gate distribution in Step 7
        if sparsity > highest_sparsity_with_good_acc and acc > 40.0:
            highest_sparsity_with_good_acc = sparsity
            best_model = model
            best_lambda = l_val
            
    # If no model hit the heuristic, just save the last one
    if best_model is None:
        best_model = model
        best_lambda = l_val
        
    print("\n" + "="*40)
    print("FINAL RESULTS SUMMARY")
    print("="*40)
    print(f"{'Lambda':<10} | {'Test Accuracy':<15} | {'Sparsity Level (%)':<20}")
    print("-" * 50)
    for res in results:
        print(f"{res['Lambda']:<10} | {res['Test Accuracy']:<15.2f} | {res['Sparsity Level (%)']:<20.2f}")

    # ==========================================
    # STEP 7: Generating the Plot
    # ==========================================
    print(f"\nGenerating plot for the best model (Lambda = {best_lambda})...")
    
    # Collect all gate values from the best model
    all_gates = []
    with torch.no_grad():
        for module in best_model.modules():
            if isinstance(module, PrunableLinear):
                # We need to apply sigmoid because the raw parameter is gate_scores
                gates = torch.sigmoid(module.gate_scores).cpu().numpy().flatten()
                all_gates.extend(gates)
                
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_gates, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Gate Value Distribution (Lambda = {best_lambda})')
    plt.xlabel('Gate Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    
    # Save the plot
    plt.savefig('gate_distribution.png')
    print("Plot saved as 'gate_distribution.png'")

    # ==========================================
    # STEP 8: Generating Trade-off Plot
    # ==========================================
    print("\nGenerating Sparsity vs. Accuracy trade-off plot...")
    lambdas_str = [str(r['Lambda']) for r in results]
    accuracies = [r['Test Accuracy'] for r in results]
    sparsities = [r['Sparsity Level (%)'] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot Accuracy on left Y axis
    color = 'tab:blue'
    ax1.set_xlabel('Lambda Penalty')
    ax1.set_ylabel('Test Accuracy (%)', color=color, fontweight='bold')
    ax1.plot(lambdas_str, accuracies, marker='o', color=color, linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 100)

    # Plot Sparsity on right Y axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Sparsity Level (%)', color=color, fontweight='bold')
    ax2.plot(lambdas_str, sparsities, marker='x', color=color, linewidth=2, linestyle='--', markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 100)

    plt.title('Accuracy vs. Sparsity Trade-off Across Lambdas')
    fig.tight_layout()
    plt.savefig('tradeoff_plot.png')
    print("Plot saved as 'tradeoff_plot.png'")

# --- End of Assignment Script ---
