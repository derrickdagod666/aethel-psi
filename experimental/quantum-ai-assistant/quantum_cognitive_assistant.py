#!/usr/bin/env python3
"""
Quantum-Cognitive AI Assistant: Unified Implementation
Incorporating concepts from Quantum Geometric Harmonic Resonance paper
and Qiskit Psi-encoded entanglement example.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# Qiskit is used conceptually for Parameter, but Pennylane handles execution
# from qiskit.circuit import Parameter as QiskitParameter # Renamed to avoid conflict
import pennylane as qml
# Use Pennylane's numpy for compatibility within qnodes
from pennylane import numpy as pnp
from loguru import logger
import faiss
import pyvista as pv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer # Using a specific model for insights
import matplotlib.pyplot as plt # For histogram plotting
import os # To check for error model file
import time # For visualization updates

# Configure logging
logger.add("quantum_cognitive.log", rotation="10 MB", retention="7 days", level="INFO")

# --- Constants inspired by the paper/example ---
# Represents the abstract parameter encoding the Ψ waveform's influence (e.g., phase, amplitude effect)
# Corresponds to 'psi' in the Qiskit example. Made learnable.
INITIAL_PSI_VALUE = np.pi / 3
N_QUBITS = 4 # Q0: Data, Q1: Transmon/Psi, Q2: Ancilla, Q3: Extra/Environment
N_LAYERS_QUANTUM = 2 # Number of trainable quantum layers in feature map
SHOTS = 1024 # Default shots for simulation
FAISS_DIM = 2**N_QUBITS # Dimension for FAISS index (probability vector)

# --------------------------
# Quantum Core Components
# --------------------------

class QuantumProcessor:
    """
    Integrated quantum processing unit.
    Includes:
    - Pennylane QNode for quantum feature mapping.
    - A parameterized gate sequence inspired by the Qiskit example to encode Ψ waveform influence.
    - Neural network based error mitigation (simulated).
    - FAISS memory for quantum state similarity (probability distributions).
    """

    def __init__(self, n_qubits=N_QUBITS, shots=SHOTS, faiss_dim=FAISS_DIM):
        self.n_qubits = n_qubits
        # Use Pennylane's default simulator with specified shots
        self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=shots)
        self.shots = shots
        self.error_model = self._build_error_model(input_dim=faiss_dim)
        # FAISS memory dimension
        self.faiss_dim = faiss_dim
        self.memory = faiss.IndexFlatL2(self.faiss_dim)
        self.memory_count = 0 # Track number of items in memory
        logger.info(f"QuantumProcessor initialized with {n_qubits} qubits, {shots} shots.")

        # --- Ψ Waveform Parameter ---
        # This parameter is intended to be learned by the classical optimizer.
        # It represents the abstract influence of the Ψ waveform, analogous to 'psi'
        # in the Qiskit example and conceptually linked to A, ω, or φ from the paper.
        # We make it an nn.Parameter so PyTorch's autograd handles it.
        self.psi = nn.Parameter(torch.tensor(INITIAL_PSI_VALUE, dtype=torch.float32))
        logger.info(f"Initial Ψ parameter (psi): {self.psi.item():.4f}")

        # --- Define the QNode ---
        # Wrap the circuit definition in the qnode decorator
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def quantum_feature_map_circuit(inputs, weights, psi_param):
            """
            Quantum circuit for feature mapping.
            Args:
                inputs (torch.Tensor): Classical input data (flattened). Shape (n_features,).
                                       Number of features should be <= n_qubits.
                weights (torch.Tensor): Learnable weights for the quantum layers. Shape (n_layers, n_qubits, 3).
                psi_param (torch.Tensor): Learnable parameter representing Ψ influence.
            Returns:
                torch.Tensor: Expectation values of PauliZ on each qubit. Shape (n_qubits,).
            """
            n_inputs = inputs.shape[0]
            if n_inputs > self.n_qubits:
                 raise ValueError(f"Number of inputs ({n_inputs}) cannot exceed number of qubits ({self.n_qubits})")

            # 1. Input Encoding (using first n_inputs qubits)
            # Example: Amplitude embedding or angle embedding
            # qml.AmplitudeEmbedding(features=inputs, wires=range(n_inputs), normalize=True, pad_with=0.)
            for i in range(n_inputs):
                 qml.RY(inputs[i] * pnp.pi, wires=i) # Scale input to be angle-like

            # 2. Prepare Ancilla/Environment Qubits (optional initialization)
            # Example: Hadamard on ancilla (wire 2) and extra qubit (wire 3)
            if self.n_qubits > 2:
                 qml.Hadamard(wires=2)
            if self.n_qubits > 3:
                 qml.Hadamard(wires=3)


            # 3. Encode Ψ Waveform Influence & Entangle (inspired by Qiskit example)
            # Map: Qiskit 0 -> Wire 0 (Data)
            #      Qiskit 1 -> Wire 1 (Transmon/Psi)
            #      Qiskit 2 -> Wire 2 (Ancilla)
            qml.RY(psi_param, wires=1) # Encode psi on transmon-like qubit
            qml.CNOT(wires=[1, 0])     # Entangle transmon(1) with data(0)
            if self.n_qubits > 2:
                 qml.CNOT(wires=[1, 2]) # Entangle transmon(1) with ancilla(2)

            # 4. Trainable Quantum Layers (Basic Entangler Layers)
            for layer_weights in weights:
                # Single qubit rotations
                for i in range(self.n_qubits):
                    qml.Rot(layer_weights[i, 0], layer_weights[i, 1], layer_weights[i, 2], wires=i)
                # Entanglement
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.n_qubits > 1: # Add entanglement between last and first qubit for cyclic boundary
                     qml.CNOT(wires=[self.n_qubits - 1, 0])

            # 5. Measurement: Expectation value of Pauli Z for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # Assign the decorated function to the instance
        self.quantum_feature_map_qnode = quantum_feature_map_circuit


    def _build_error_model(self, input_dim, hidden_dim=32):
        """Builds or loads a simple neural error mitigation network."""
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1) # Ensure output is a probability distribution
        )
        model_path = 'error_model.pth'
        # Ensure the directory exists if saving for the first time
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)

        if os.path.exists(model_path):
            try:
                # Load model state, making sure it's loaded to the correct device (e.g., CPU)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                logger.info(f"Loaded pre-trained error model from {model_path}")
            except Exception as e:
                 logger.warning(f"Could not load error model from {model_path}: {e}. Using untrained model.")
                 # Save initial weights if creating for the first time
                 # torch.save(model.state_dict(), model_path)
                 # logger.info(f"Saved initial untrained error model to {model_path}")
        else:
            logger.warning(f"Error model file {model_path} not found. Using untrained model.")
            # Save initial weights if creating for the first time
            # torch.save(model.state_dict(), model_path)
            # logger.info(f"Saved initial untrained error model to {model_path}")

        model.eval() # Set to evaluation mode as we are not training it here
        return model

    def get_probabilities(self, qnode_func, *args):
        """Helper function to get probabilities from a QNode execution."""
        # Execute the QNode to get expectation values or samples
        # To get probabilities, we need to change the return type of the QNode
        # or run a separate QNode that returns probabilities.
        # Let's create a temporary QNode for probabilities.

        @qml.qnode(self.dev, interface='torch')
        def prob_circuit(*inner_args):
            qnode_func(*inner_args) # Run the original circuit structure
            return qml.probs(wires=range(self.n_qubits))

        with torch.no_grad(): # Don't track gradients for probability calculation
            probs = prob_circuit(*args)
        return probs.cpu().numpy() # Return as numpy array on CPU

    def execute_circuit_with_mitigation(self, inputs, weights):
        """
        Executes the quantum feature map circuit, applies error mitigation,
        and stores the result in FAISS memory.
        Args:
            inputs (torch.Tensor): Input features for the QNode.
            weights (torch.Tensor): Quantum layer weights for the QNode.
        Returns:
            np.ndarray: Error-mitigated probability distribution.
            torch.Tensor: Original expectation values from the QNode (for gradient calculation).
        """
        # 1. Execute QNode to get expectation values (needed for gradients)
        exp_values = self.quantum_feature_map_qnode(inputs, weights, self.psi)

        # 2. Get Raw Probabilities (without tracking gradients for this part)
        raw_probs = self.get_probabilities(self.quantum_feature_map_qnode, inputs.detach(), weights.detach(), self.psi.detach())

        # 3. Apply Neural Error Mitigation
        # Ensure raw_probs is float32 for the NN model
        raw_probs_tensor = torch.tensor(raw_probs, dtype=torch.float32).unsqueeze(0) # Add batch dimension
        with torch.no_grad(): # No gradients for error model inference
            corrected_probs_tensor = self.error_model(raw_probs_tensor)
        corrected_probs = corrected_probs_tensor.squeeze(0).cpu().numpy() # Remove batch dim

        # 4. Store corrected probabilities in FAISS memory
        # Ensure the vector is C-contiguous and float32
        corrected_probs_faiss = np.ascontiguousarray(corrected_probs, dtype=np.float32).reshape(1, -1)
        try:
            self.memory.add(corrected_probs_faiss)
            self.memory_count += 1
        except Exception as e:
            logger.error(f"Failed to add vector to FAISS: {e}")
            logger.error(f"Vector shape: {corrected_probs_faiss.shape}, dtype: {corrected_probs_faiss.dtype}")
            logger.error(f"FAISS index dimension: {self.memory.d}, count: {self.memory.ntotal}")


        # Return both the expectation values (for training) and corrected probs (for analysis/memory)
        return corrected_probs, exp_values

    def retrieve_similar_states(self, query_probs, k=3):
        """Retrieves k most similar states from FAISS memory."""
        if self.memory_count == 0:
            logger.warning("FAISS memory is empty. Cannot retrieve states.")
            return [], []
        if self.memory_count < k:
            k = self.memory_count
            logger.warning(f"Memory contains only {self.memory_count} states. Retrieving {k}.")

        query_probs_faiss = np.ascontiguousarray(query_probs, dtype=np.float32).reshape(1, -1)
        try:
            distances, indices = self.memory.search(query_probs_faiss, k)
            # Note: FAISS returns distances (L2 squared) and indices.
            # We might want to retrieve the actual vectors later if stored separately,
            # but for now, indices and distances are informative.
            return distances[0], indices[0]
        except Exception as e:
            logger.error(f"Failed to search FAISS index: {e}")
            return [], []


# --------------------------
# Cognitive Neural Network
# --------------------------

class QuantumCognitiveNN(nn.Module):
    """
    Hybrid quantum-classical neural architecture using Transformer components.
    """

    def __init__(self, n_qubits=N_QUBITS, n_layers_quantum=N_LAYERS_QUANTUM, classical_dim=256, n_heads=8, n_encoder_layers=3, n_decoder_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.classical_dim = classical_dim
        self.quantum_processor = QuantumProcessor(n_qubits=n_qubits)

        # Learnable weights for the quantum circuit layers
        # Shape: (n_layers, n_qubits, 3) for qml.Rot (3 parameters per rotation)
        self.quantum_weights = nn.Parameter(torch.randn(n_layers_quantum, n_qubits, 3) * 0.1) # Small initial weights

        # Classical components (Transformer-based)
        encoder_layer = nn.TransformerEncoderLayer(d_model=classical_dim, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=classical_dim, nhead=n_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # Attention mechanism to combine classical and quantum info (optional)
        # self.attention = nn.MultiheadAttention(classical_dim, n_heads, batch_first=True)

        # Interface layers: Project quantum features to classical dim and vice-versa
        self.quantum_projection = nn.Linear(n_qubits, classical_dim)
        # Output layer: Project final classical state to desired output dimension (e.g., number of classes)
        self.output_projection = nn.Linear(classical_dim, 2) # Example: Binary classification

        # Input embedding layer for classical features
        self.input_embedding = nn.Linear(1, classical_dim) # Example: if input features are 1D per step

        logger.info("QuantumCognitiveNN initialized.")
        logger.info(f"Quantum weights shape: {self.quantum_weights.shape}")


    def forward(self, x):
        """
        Forward pass through the hybrid network.
        Args:
            x (torch.Tensor): Input tensor. Shape (batch_size, seq_len, feature_dim).
                              For simplicity, let's assume feature_dim=1 for now.
        Returns:
            torch.Tensor: Output predictions.
        """
        batch_size, seq_len, _ = x.shape

        # 1. Classical Encoding
        # Embed input features to classical_dim
        embedded_x = self.input_embedding(x) # Shape: (batch_size, seq_len, classical_dim)
        encoded_classical = self.encoder(embedded_x) # Shape: (batch_size, seq_len, classical_dim)

        # 2. Quantum Feature Extraction (Process each item in the sequence)
        quantum_features_list = []
        all_corrected_probs = [] # Store corrected probs for potential analysis/retrieval

        # Process each sequence element - this might be slow!
        # Consider processing only the last element or an aggregate for efficiency.
        # Here, we process the *last* element of the sequence as an example.
        last_elements = x[:, -1, :] # Shape: (batch_size, feature_dim=1)

        for i in range(batch_size):
            # Select features for the quantum circuit (e.g., first n_qubits features)
            # Assuming feature_dim >= n_qubits for this example encoding
            # If feature_dim < n_qubits, padding or different encoding needed.
            # Let's use the single feature from last_elements and replicate/scale it.
            # A more sophisticated mapping from seq_len*feature_dim to quantum inputs is needed for real tasks.
            quantum_input = last_elements[i].repeat(self.n_qubits) * 0.5 # Example scaling/mapping

            # Execute quantum circuit and get error-mitigated probs + expectation values
            corrected_probs, exp_values = self.quantum_processor.execute_circuit_with_mitigation(
                quantum_input, self.quantum_weights
            )
            quantum_features_list.append(exp_values) # Use expectation values for backprop
            all_corrected_probs.append(corrected_probs) # Store for potential use

        # Stack the quantum features (expectation values)
        quantum_features = torch.stack(quantum_features_list) # Shape: (batch_size, n_qubits)

        # 3. Hybrid Integration
        # Project quantum features to the classical dimension
        projected_quantum = self.quantum_projection(quantum_features) # Shape: (batch_size, classical_dim)

        # Combine with classical features (e.g., add to the last element's encoding)
        # We need to align dimensions. encoded_classical is (batch, seq, dim), projected_quantum is (batch, dim)
        # Let's use the encoding of the last sequence element.
        last_classical_encoded = encoded_classical[:, -1, :] # Shape: (batch_size, classical_dim)

        # Simple combination: Add quantum features to the last classical encoding
        combined_features = last_classical_encoded + projected_quantum

        # Alternative: Use attention (if defined)
        # combined_features, _ = self.attention(projected_quantum.unsqueeze(1), encoded_classical, encoded_classical)
        # combined_features = combined_features.squeeze(1) # Remove sequence dim if needed

        # 4. Classical Decoding (Optional - depends on task)
        # If the task requires sequence generation, use the decoder.
        # For classification based on the final state, we can skip the decoder.
        # decoded_output = self.decoder(combined_features.unsqueeze(1), encoded_classical) # Example decoder usage
        # final_representation = decoded_output[:, -1, :] # Use last decoder output

        # Use the combined features directly for classification
        final_representation = combined_features

        # 5. Output Projection
        output = self.output_projection(final_representation) # Shape: (batch_size, num_classes)

        # Store corrected probabilities for potential retrieval/analysis outside forward pass
        self._last_corrected_probs = all_corrected_probs

        return output

# --------------------------
# Training Infrastructure
# --------------------------

class HybridTrainer:
    """Unified quantum-classical training manager"""

    def __init__(self, model: QuantumCognitiveNN, learning_rate_classical=0.001, learning_rate_quantum=0.01):
        self.model = model
        # Separate optimizers for classical and quantum parameters
        classical_params = [p for name, p in model.named_parameters() if 'quantum_weights' not in name and 'psi' not in name]
        quantum_params = [model.quantum_weights, model.quantum_processor.psi] # Include psi here

        self.classical_optimizer = optim.AdamW(classical_params, lr=learning_rate_classical)
        # Use Pennylane's optimizer for quantum parameters handled by Pennylane's interface
        # Note: We manually handle the psi parameter's gradient via PyTorch's autograd
        self.quantum_optimizer = qml.AdamOptimizer(stepsize=learning_rate_quantum) # For Pennylane weights if needed separately

        self.loss_fn = nn.CrossEntropyLoss() # Example loss for classification

        # LLM for insights (load model and tokenizer)
        try:
            # Using a smaller, manageable model like gpt2
            logger.info("Loading LLM for insights (gpt2)...")
            self.llm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.llm_model = AutoModelForCausalLM.from_pretrained("gpt2")
            # Set pad token if not set
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                self.llm_model.config.pad_token_id = self.llm_model.config.eos_token_id
            self.llm_pipeline = pipeline("text-generation", model=self.llm_model, tokenizer=self.llm_tokenizer, device=-1) # Use CPU
            logger.info("LLM loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}. Insights will be disabled.")
            self.llm_pipeline = None


    def train(self, dataloader, epochs=10): # Reduced epochs for faster testing
        logger.info("Starting hybrid training")
        training_start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            num_batches = 0
            self.model.train() # Set model to training mode

            for batch_data, batch_targets in dataloader:
                # --- Training Step ---
                self.classical_optimizer.zero_grad()

                # Forward pass - Quantum execution happens inside model.forward
                outputs = self.model(batch_data)

                # Calculate loss
                loss = self.loss_fn(outputs, batch_targets)

                # Backward pass (calculates gradients for classical and quantum parameters via PyTorch interface)
                loss.backward()

                # Optimize classical parameters (including psi)
                self.classical_optimizer.step()

                # Optimize quantum weights using Pennylane optimizer (if needed separately, but handled by PyTorch interface here)
                # The PyTorch interface calculates gradients for quantum_weights, so classical_optimizer handles it.
                # If using a different diff_method or optimizer, manual stepping might be needed:
                # self.model.quantum_weights = self.quantum_optimizer.step(lambda w: self._calculate_loss_for_quantum(w, batch_data, batch_targets), self.model.quantum_weights)

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches
            epoch_duration = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_epoch_loss:.4f} | Duration: {epoch_duration:.2f}s")
            logger.info(f"Current Psi parameter: {self.model.quantum_processor.psi.item():.4f}")

            # Generate insights periodically
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1: # Generate every 5 epochs and on the last epoch
                 self._generate_insights(epoch + 1, avg_epoch_loss)

        total_training_time = time.time() - training_start_time
        logger.info(f"Training finished in {total_training_time:.2f} seconds.")

    # def _calculate_loss_for_quantum(self, q_weights, batch_data, batch_targets):
    #     """Helper function for Pennylane optimizer step if needed."""
    #     # Temporarily assign new weights to calculate loss
    #     original_weights = self.model.quantum_weights.data.clone()
    #     self.model.quantum_weights.data = torch.tensor(q_weights)
    #     with torch.no_grad(): # Don't track gradients here, just calculate loss value
    #         outputs = self.model(batch_data)
    #         loss = self.loss_fn(outputs, batch_targets)
    #     self.model.quantum_weights.data = original_weights # Restore original weights
    #     return loss.item()


    @torch.no_grad() # No gradients needed for insight generation
    def _generate_insights(self, epoch, loss_value):
        """LLM-powered training analysis"""
        if not self.llm_pipeline:
            logger.warning("LLM pipeline not available. Skipping insight generation.")
            return

        logger.info("Generating training insights...")
        prompt = (
            f"Quantum-Cognitive AI Training Analysis:\n"
            f"Epoch: {epoch}\n"
            f"Average Loss: {loss_value:.4f}\n"
            f"Current Psi Parameter (Waveform Influence): {self.model.quantum_processor.psi.item():.4f}\n"
            f"Quantum Layers: {self.model.quantum_weights.shape[0]}\n"
            f"Qubits: {self.model.n_qubits}\n\n"
            f"Provide a brief technical analysis (2-3 sentences) focusing on potential interpretations "
            f"of the loss trend and the psi parameter value in the context of a hybrid quantum-classical system. "
            f"Mention potential next steps or areas to monitor."
        )
        try:
            # Generate text, limiting length and handling potential errors
            results = self.llm_pipeline(prompt, max_length=len(prompt.split()) + 100, num_return_sequences=1, truncation=True, pad_token_id=self.llm_tokenizer.pad_token_id)
            insight = results[0]['generated_text'][len(prompt):].strip() # Extract only generated part
            if not insight:
                 insight = "LLM provided no specific insight for this epoch."
            logger.info(f"--- Epoch {epoch} Insights ---\n{insight}\n-------------------------")
        except Exception as e:
            logger.error(f"LLM insight generation failed: {e}")


# --------------------------
# Monitoring & Visualization
# --------------------------

class QuantumMonitor:
    """Real-time quantum state probability visualization using PyVista"""

    def __init__(self, n_qubits=N_QUBITS):
        self.n_qubits = n_qubits
        self.n_states = 2**n_qubits
        self.plotter = pv.Plotter(window_size=[800, 400], title="Quantum State Probabilities")
        # Create initial points for the bar chart
        x_coords = np.arange(self.n_states)
        y_coords = np.zeros(self.n_states) # Initial heights
        z_coords = np.zeros(self.n_states)
        self.points = np.vstack((x_coords, y_coords, z_coords)).T
        # Create a PolyData object representing the bars (initially flat)
        self.bars = pv.PolyData(self.points)
        # Define lines connecting the base points to the top points (for bars)
        lines = []
        for i in range(self.n_states):
            lines.extend([2, i, i]) # Format: [num_points, point_idx1, point_idx2] - creates zero-height lines initially
        self.bars.lines = np.array(lines)
        # Add the bars to the plotter
        self.actor = self.plotter.add_mesh(self.bars, scalars=y_coords, cmap='viridis', line_width=5, render_lines_as_tubes=True, scalar_bar_args={'title': 'Probability'})
        self.plotter.view_xy() # Set initial view
        self.plotter.enable_parallel_projection()
        self.plotter.camera.zoom(1.5)
        # Add labels for states (can be slow for many qubits)
        if n_qubits <= 6: # Only add labels for small number of qubits
             state_labels = [format(i, f'0{n_qubits}b') for i in range(self.n_states)]
             self.plotter.add_point_labels(self.points, state_labels, point_size=1, font_size=10, shape=None)

        logger.info("QuantumMonitor initialized with PyVista.")
        self.plotter.show(interactive=False, auto_close=False) # Show window but don't block

    def update_visualization(self, probs):
        """Updates the PyVista plot with new probabilities."""
        if len(probs) != self.n_states:
            logger.error(f"Probability vector length ({len(probs)}) does not match expected states ({self.n_states}). Cannot update visualization.")
            return

        # Update point heights based on probabilities
        self.points[:, 1] = probs * 10 # Scale heights for visibility
        self.bars.points = self.points # Update the mesh points

        # Update scalars for coloring
        self.bars.point_data['scalars'] = probs

        # Force render update
        self.plotter.render()
        # Add a small pause to allow the window to refresh
        # time.sleep(0.05) # Causes blocking, better handled by event loop if available

    def close(self):
        """Closes the PyVista plotter window."""
        self.plotter.close()
        logger.info("QuantumMonitor closed.")


# --------------------------
# Data Generation
# --------------------------

def generate_synthetic_data(n_samples=1000, seq_len=10, n_features=1):
    """
    Generate synthetic sequential data with patterns potentially influenced
    by a simulated waveform.
    """
    logger.info(f"Generating synthetic dataset ({n_samples} samples, seq_len={seq_len}, features={n_features})...")
    # Simple sine wave pattern for features
    time_steps = torch.arange(seq_len).float().unsqueeze(0).unsqueeze(-1).repeat(n_samples, 1, n_features)
    base_signal = torch.randn(n_samples, 1, n_features) * 0.5 # Random base level
    frequency = torch.randn(n_samples, 1, n_features) * 0.1 + 0.05 # Random frequency
    phase = torch.randn(n_samples, 1, n_features) * np.pi # Random phase

    # Simulate Ψ waveform influence (e.g., modulating amplitude)
    psi_influence = torch.sin(time_steps * 0.1 + INITIAL_PSI_VALUE) * 0.2 + 1.0 # Modulating factor

    X = (torch.sin(time_steps * frequency + phase) * psi_influence + base_signal)

    # Generate labels based on some property of the sequence (e.g., final value)
    y = (X[:, -1, 0] > X[:, -1, 0].mean()).long() # Binary label based on final value threshold

    logger.info("Synthetic dataset generated.")
    return TensorDataset(X, y)

# --------------------------
# Main Execution
# --------------------------

if __name__ == "__main__":
    logger.info("--- Quantum-Cognitive AI Assistant Initializing ---")

    # Initialize components
    qcnn = QuantumCognitiveNN(n_qubits=N_QUBITS, classical_dim=128, n_heads=4, n_encoder_layers=2, n_decoder_layers=2) # Smaller classical model
    trainer = HybridTrainer(qcnn, learning_rate_classical=0.002, learning_rate_quantum=0.02)
    monitor = QuantumMonitor(n_qubits=N_QUBITS)

    # Prepare data
    # Using smaller dataset for quicker testing
    dataset = generate_synthetic_data(n_samples=200, seq_len=8, n_features=1)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True) # Smaller batch size

    # --- Training Loop ---
    try:
        trainer.train(dataloader, epochs=10) # Reduced epochs for testing
    except Exception as e:
        logger.exception("An error occurred during training.") # Log traceback
    finally:
        # --- Post-Training ---
        logger.info("--- Post-Training Analysis ---")

        # Save final model
        model_save_path = "quantum_cognitive_assistant.pth"
        try:
            torch.save(qcnn.state_dict(), model_save_path)
            logger.info(f"Training completed and model saved to {model_save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

        # Example inference
        logger.info("Running example inference...")
        try:
            # Create a single sample with correct dimensions
            test_input_raw, _ = next(iter(dataloader)) # Get a batch
            test_input = test_input_raw[0].unsqueeze(0) # Take first sample, add batch dim -> (1, seq_len, features)

            qcnn.eval() # Set model to evaluation mode
            with torch.no_grad():
                prediction = qcnn(test_input)
                predicted_class = torch.argmax(prediction, dim=1).item()
                # Retrieve the probabilities generated during this forward pass
                last_probs = qcnn._last_corrected_probs[0] if hasattr(qcnn, '_last_corrected_probs') else None

            logger.info(f"Prediction (logits) for sample: {prediction.numpy()}")
            logger.info(f"Predicted class: {predicted_class}")

            # Visualize final quantum state probabilities from inference
            if last_probs is not None:
                logger.info("Updating monitor with probabilities from inference...")
                monitor.update_visualization(last_probs)
                # Keep plot open for a bit
                # time.sleep(5) # Causes blocking

                # Retrieve similar states from memory
                logger.info("Retrieving similar states from FAISS memory...")
                distances, indices = qcnn.quantum_processor.retrieve_similar_states(last_probs, k=5)
                if indices:
                    logger.info(f"Found {len(indices)} similar states (indices): {indices}")
                    logger.info(f"Distances (L2^2): {distances}")
                else:
                    logger.info("No similar states found or memory empty.")

            else:
                 logger.warning("Could not retrieve last probabilities for visualization.")

        except Exception as e:
            logger.exception("An error occurred during inference or visualization.")

        # Close the monitor window
        logger.info("Closing visualization...")
        monitor.close()
        logger.info("--- Quantum-Cognitive AI Assistant Finished ---")