#@title Install Dependencies (Run this cell first)
!pip install numpy matplotlib scipy qutip gradio plotly transformers torch faiss qiskit beautifulsoup4 requests
!pip install gTTS playsound
!pip install ipywidgets
!jupyter nbextension enable --py widgetsnbextension

#@title Setup Kaggle (Optional - Uncomment and fill in details if using)

# # Upload your kaggle.json file here
# from google.colab import files
# files.upload()
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# # Example dataset URL - REPLACE with a real one if using Kaggle.
# # You will also need to adapt the preprocessing and instruction
# # generation functions to match the specific dataset you choose.
# EXAMPLE_KAGGLE_DATASET_URL = "" # PLACEHOLDER


#@title Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal, expm
import qutip as qt
import logging
import gradio as gr
import json
import asyncio
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_bloch_multivector
import argparse
from typing import Callable, Union, List, Tuple, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer  # For Gemma
import torch  # Import PyTorch
from gtts import gTTS #Text to speech
import os # File operations
import subprocess # Running shell commands
import random # Random numbers
import pandas as pd # DataFrames
from bs4 import BeautifulSoup
import requests

#@title Constants
HBAR = 1.0
MASS = 1.0
A = 54  # Dimensionless amplitude
OMEGA = 2 * np.pi / 945  # Angular frequency
PHI = 0  # Phase offset

#@title core/quantum_system.py
from abc import ABC, abstractmethod


class QuantumSystem(ABC):
    """Abstract base class for quantum systems."""
    def __init__(self):
        pass

    @abstractmethod
    def time_evolve(self, initial_state: Union[np.ndarray, List], time_steps: int, dt: float) -> Union[np.ndarray,List]:
        """Evolves the system in time."""
        pass

    @abstractmethod
    def visualize(self, **kwargs):
       """Visualizes the system state."""
       pass

#@title core/utils.py

def psi_waveform(x: Union[np.ndarray, float], A: float, omega: float, phi: float) -> Union[np.ndarray, float]:
    """Represents the spatial part of the universal waveform."""
    return A * np.cos(omega * x + phi)

def generate_variations(instruction_template, parameters):
    variations = [instruction_template.format(**parameters)]
    if 'system' in parameters:
        variations.append(f"Can you show me the {parameters['system']}?")
        variations.append(f"Please simulate the {parameters['system']}.")
    if 'time_steps' in parameters:
        variations.append(f"What happens after {parameters['time_steps']} steps?")
    # Question variations:
    if 'omega0' in parameters:
        variations.append(f"What is the effect of changing omega0 to {parameters['omega0']}?")
    #Imperative variations
    if 'coupling_strength' in parameters:
        variations.append(f"Set coupling strength to {parameters['coupling_strength']}.")
    return variations

#@title systems/harmonic_oscillator.py
class HarmonicOscillator(QuantumSystem):
    """
    Quantum harmonic oscillator with spin and time-varying parameters,
    and coupling to the Psi waveform.
    """
    def __init__(self, A: float = A, omega: float = OMEGA, alpha: float = 0.1,
                 spin: Optional[float] = None, B_field: Optional[float] = None,
                 coupling_strength: float = 0.1): # Added coupling
        super().__init__()
        self.omega0: float = 1.0  # Natural frequency
        self.A: float = A  # For Psi Waveform
        self.omega: float = omega # For Psi Waveform
        self.alpha: float = alpha  # Coupling constant (if applicable)
        self.spin: Optional[float] = spin
        self.B_field: Optional[float] = B_field
        self.x: Optional[np.ndarray] = None
        self.eigvals: Optional[np.ndarray] = None
        self.eigvecs: Optional[np.ndarray] = None
        self.coupling_strength: float = coupling_strength
        self.states = []


        if spin is not None and B_field is None:
            raise ValueError("B_field (magnetic field) is required when spin is specified.")

        # Pauli matrices
        self.pauli_x: np.ndarray = np.array([[0, 1], [1, 0]])
        self.pauli_y: np.ndarray = np.array([[0, -1j], [1j, 0]])
        self.pauli_z: np.ndarray = np.array([[1, 0], [0, -1]])

        # Time-varying parameters (initialized as constant functions)
        self.A_func: Callable[[float], float] = lambda t: self.A
        self.omega_func: Callable[[float], float] = lambda t: self.omega
        self.phi_func: Callable[[float], float] = lambda t: 0.0

    def set_parameter_functions(self, A_func: Optional[Callable[[float], float]] = None,
                                omega_func: Optional[Callable[[float], float]] = None,
                                phi_func: Optional[Callable[[float], float]] = None):
        """Set time-dependent parameter functions."""
        if A_func: self.A_func = A_func
        if omega_func: self.omega_func = omega_func
        if phi_func: self.phi_func = phi_func

    def _hamiltonian_matrix(self, x: np.ndarray, dx: float, t: float = 0.0):
        """Construct Hamiltonian matrix (with or without spin).  Includes Psi coupling."""
        V = self.V_total(x, t)  # Harmonic potential
        factor = - (HBAR**2) / (2 * MASS)
        T_diag = factor * (-2.0 / dx**2) * np.ones(len(x))
        T_offdiag = factor * (1.0 / dx**2) * np.ones(len(x) - 1)

        # --- Coupling to Psi(x) ---
        psi_values = psi_waveform(x, self.A_func(t), self.omega_func(t), self.phi_func(t))
        V_psi = self.coupling_strength * psi_values  # Simple linear coupling
        H_diag = T_diag + V + V_psi  # Add Psi to the *potential*

        if self.spin is not None:
            H_spin = -self.B_field * self.pauli_z  # Zeeman interaction
            H_total = np.kron(np.diag(H_diag), np.eye(2)) + np.kron(np.eye(len(x)), H_spin)
            T_offdiag_total = np.kron(np.diag(T_offdiag, k=1), np.eye(2))
            return H_total, T_offdiag_total  # Return both matrices

        return np.diag(H_diag), T_offdiag  # Return diagonal and off-diagonal

    def V_total(self, x: np.ndarray, t: float) -> np.ndarray:
        """Total potential energy (harmonic potential ONLY, no psi here)."""
        V_harmonic = 0.5 * MASS * self.omega0**2 * x**2
        return V_harmonic # Only the harmonic potential


    def solve_system(self, x_min: float = -10.0, x_max: float = 10.0, N_points: int = 1000):
        """Solve time-independent Schrödinger equation."""
        x = np.linspace(x_min, x_max, N_points)
        dx = x[1] - x[0]

        H_diag, T_offdiag = self._hamiltonian_matrix(x, dx)

        if self.spin is not None:
            # Assemble full Hamiltonian (ensure Hermitian)
            H = H_diag + np.diag(T_offdiag, k=1) + np.diag(T_offdiag.conj(), k=-1)
            self.eigvals, self.eigvecs = np.linalg.eigh(H)
            self.eigvecs = self.eigvecs.reshape(N_points, 2, -1)  # Reshape for spin components
        else:
            self.eigvals, self.eigvecs = eigh_tridiagonal(H_diag, T_offdiag)

        self.x = x  # Store x for later use (e.g., visualization)
        return x, self.eigvals, self.eigvecs

    def time_evolve(self, initial_state: np.ndarray, time_steps: int, dt: float) -> np.ndarray:
        """Time evolution using matrix exponentiation."""
        if self.x is None:
            raise ValueError("Must call solve_system() before time_evolve().")

        x = self.x
        dx = x[1] - x[0]
        if not isinstance(initial_state, qt.Qobj):
            initial_state = qt.Qobj(initial_state)

        initial_state = initial_state.unit()
        states = []
        current_state = initial_state
        for t in range(time_steps):
            H_diag, H_offdiag = self._hamiltonian_matrix(x, dx, t * dt)
            if self.spin is not None:
                H = H_diag + np.diag(H_offdiag, k=1) + np.diag(H_offdiag.conj(), k=-1)
            else: # No spin
                H = np.diag(H_diag) + np.diag(T_offdiag, k=1) + np.diag(T_offdiag.conj(), k=-1)

            U = expm(-1j * H * dt / HBAR)
            if self.spin is not None:
                current_state = U @ current_state.full()
                states.append(qt.Qobj(current_state))
            else:
                current_state = U @ current_state.full()
                states.append(qt.Qobj(current_state))
        self.states = states
        return states

    def visualize(self, states: Optional[List[qt.Qobj]] = None, time_index: int = -1):
        """Visualizes the probability density using Plotly."""
        if self.x is None:
            raise ValueError("Run solve_system first.")
        if states is None:
            states = self.states

        x = self.x
        if not states:
            if self.x is None or self.eigvecs is None:
                raise ValueError("Run solve_system first.")

            x = self.x
            if self.spin is not None:
                psi = self.eigvecs[:, :, 0]  # Example: First state ,  # Get both spin components
                prob_density = np.sum(np.abs(psi)**2, axis=1)  # Sum over spin
            else:
                prob_density = np.abs(self.eigvecs[:, 0])**2  # Example: First state

            fig = go.Figure(data=[go.Scatter(x=x, y=prob_density, mode='lines')])
            fig.update_layout(title="Harmonic Oscillator Probability Density",
                            xaxis_title="Position (x)",
                            yaxis_title="Probability Density")
            return fig  # Return the Plotly figure

        else:
            if self.spin is not None:
                psi = states[time_index].full()  # Example: First state ,  # Get both spin components
                prob_density = np.abs(psi.reshape(len(self.x),2))
                prob_density = np.sum(prob_density**2, axis=1)
            else:
                prob_density = np.abs(states[time_index].full().flatten())**2  # Example: First state

            fig = go.Figure(data=[go.Scatter(x=x, y=prob_density, mode='lines')])
            fig.update_layout(title=f"Harmonic Oscillator Probability Density (Time Index {time_index})",
                            xaxis_title="Position (x)",
                            yaxis_title="Probability Density")
            return fig  # Return the Plotly figure

#@title core/quantum_classical_module.py
class QuantumClassicalModule:
    """Manages interaction between quantum systems and classical AI (LLM)."""
    def __init__(self, model_name: str = "google/gemma-2b-it"):
        self.tools: Dict[str, Callable] = {}  # Tools for the LLM
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
        except Exception as e:
            print(f"Could not load Gemma: {e}")
            self.tokenizer = None
            self.model = None

        self.system_instances = {} # Keep track of the instances


    def register_tool(self, function: Callable, name: str):
        """Register tools for the language model."""
        self.tools[name] = function

    def text_to_audio(self, text):
        """Converts text to an audio file using gTTS."""
        try:
            tts = gTTS(text=text, lang='en')
            audio_file = "temp_audio.mp3" # Using a temporary file.
            tts.save(audio_file)
            return audio_file
        except Exception as e:
            print(f"Error in converting to audio: {e}")
            return None

    async def process_input(self, user_input: str, current_system_name:str) -> List[Tuple[str, str]]:
      """Process input using Gemma LLM

      Args:
          user_input: The user's input string.
          current_system_name: Name of the current system

      Returns:
          A list of (user input, response) pairs to display in the chatbot.
      """
      if not user_input:
          return [("", "Please provide a valid input.")]

      try:
          # --- More Structured Prompt ---
          prompt = f"""
    System: QCE AI Assistant
    Task: Translate the following user instruction into a JSON object representing a command for a quantum simulation.
    The available tools are: run_simulation, time_evolution, visualize_system.
    The available systems are: Harmonic Oscillator.

    User Instruction: {user_input}
    Current System: {current_system_name}

    JSON:
    """
          # --- Gemma Integration ---
          # 1. Tokenize the input
          if self.tokenizer:
            inputs = self.tokenizer(prompt, return_tensors="pt") #Added prompt
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            # 2. Generate a response with Gemma
            outputs = self.model.generate(**inputs, max_new_tokens=200) # Limit Output
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
          else:
              decoded_output = "{'tool':'','system':'','parameters':{}}" #Dummy output

          # 3. Parse Gemma's output (expecting JSON)
          try:

              # Find the JSON part of the output (Gemma might add text before/after)
              json_start = decoded_output.find("{")
              json_end = decoded_output.rfind("}") + 1
              if json_start == -1 or json_end == -1:
                  raise json.JSONDecodeError("No valid JSON found in output", decoded_output, 0)

              response_json_str = decoded_output[json_start:json_end]
              response_json = json.loads(response_json_str)


              tool_name = response_json.get("tool")  # Use .get() for safety
              parameters = response_json.get("parameters", {})  # Default to empty dict
              system_name = response_json.get("system")

              if system_name and system_name != current_system_name:
                if system_name not in self.system_instances:
                    # Create a new instance if necessary
                    if system_name == "Harmonic Oscillator":
                        self.system_instances[system_name] = HarmonicOscillator(**parameters)
                    #Add more systems
                    else:
                        response_text = f"System '{system_name}' not supported."
                        return [("",response_text)] # No valid JSON
                current_system = self.system_instances[system_name]
              else: # System is the same
                current_system = self.system_instances.get(current_system_name)
                if current_system is None:
                    response_text = "Current system not found please select or create one"
                    return[(user_input,response_text)]

              if tool_name and tool_name in self.tools:
                  # 4. Call the tool with the extracted parameters

                  # Handle special cases and type conversions
                if tool_name == "time_evolution":
                    # Convert initial state string
                    initial_state_str = parameters.get("initial_state", "[1, 0]")
                    try:
                        initial_state = np.array(json.loads(initial_state_str), dtype=complex)
                    except json.JSONDecodeError:
                        return [("", "Invalid initial state format.  Use, e.g., [1, 0]")]
                    parameters["initial_state"] = initial_state  # Use the converted array
                    # Convert time steps, and dt
                    parameters["time_steps"] = int(parameters.get("time_steps", 100))
                    parameters["dt"] = float(parameters.get("dt", 0.1))

                    # Handle Coupled Oscillators separately
                    if current_system_name == "Coupled Oscillators":
                            result = self.tools[tool_name](current_system, **parameters)
                    else:
                        result = self.tools[tool_name](current_system, **parameters)
                elif tool_name == "run_simulation":
                    result = self.tools[tool_name](current_system)
                else:
                    result = self.tools[tool_name](current_system, **parameters) # Pass the current system


                if isinstance(result, go.Figure):
                    response_text = "Here is the requested Plot"
                    response_audio = self.text_to_audio(response_text)
                    return [(user_input, (response_audio, result))] # Audio and Plot
                else:
                    response_text = f"Tool '{tool_name}' executed. Result: {result}"
                    response_audio = self.text_to_audio(response_text)  # Convert text to audio
                    return [(user_input, (response_audio, response_text))] # Both audio and text
              elif tool_name:  # Tool name provided, but not found
                  response_text = f"Tool '{tool_name}' is not available."
                  response_audio = self.text_to_audio(response_text)
                  return [(user_input, (response_audio,response_text))]
              else:
                  response_text = "I'm sorry, I couldn't determine which tool to use." # No tool name
                  response_audio = self.text_to_audio(response_text)
                  return [(user_input, (response_audio, response_text))]

          except (json.JSONDecodeError, KeyError, TypeError) as e:
              response_text = f"Error processing LLM output: {e}. Output: {decoded_output}"
              logging.exception("Error processing LLM output:")
              response_audio = self.text_to_audio(response_text)
              return[(user_input,(response_audio,response_text))]

      except Exception as e:
          response_text = f"An unexpected error occurred: {e}"
          logging.exception("Unexpected error in process_input:")
          response_audio = self.text_to_audio(response_text)
          return[(user_input,(response_audio, response_text))]
#@title interface/gradio_app.py
class QuantumAssistant:
    """Integrated quantum AI assistant."""
    def __init__(self):
        self.qcm = QuantumClassicalModule()
        self.available_systems = {
            "Harmonic Oscillator": HarmonicOscillator,
            # "Coupled Oscillators": CoupledOscillators,
            # "Particle in a Box": ParticleInABox,
            # "Hydrogen Atom": HydrogenAtom,  # Enable later
        }
        self.current_system_name = "Harmonic Oscillator"
        # Initialize the instance
        self.qcm.system_instances[self.current_system_name] = self.available_systems[self.current_system_name]()
        self._register_tools()
        self.ho_params = None #Keep track of parameter visibility


    def _register_tools(self):
        """Registers all quantum tools with the language model."""
        tools = [
            ("run_simulation", self.run_simulation_tool),
            ("time_evolution", self.run_time_evolution_tool),
            ("visualize_system", self.visualize_system_tool),
            # ("calculate_entropy", self.calculate_entanglement_entropy_tool), # Removed, example
            # ("quantum_memory", self.retrieve_memories_tool),  # Removed, example
        ]

        for name, func in tools:
            self.qcm.register_tool(func, name=name)

    # --- Tool Implementations (Corrected and using Plotly) ---

    def run_simulation_tool(self, system):
        """Runs a simulation for the selected system."""
        if isinstance(system, str):
            system_instance = self.available_systems.get(system)
            if system_instance is None:
                return f"Error: System '{system}' not found."
            system_instance = system_instance() # Create an instance
            self.qcm.system_instances[system] = system_instance # Store the instance
        else:
            system_instance = system


        if hasattr(system_instance, 'solve_system'):
            system_instance.solve_system()
        return f"Simulation completed for {type(system_instance).__name__}."

    def run_time_evolution_tool(self, system, initial_state_str: str, time_steps: int, dt: float):
        """Runs time evolution for the selected system."""
        try:
            if isinstance(initial_state_str,str):
                initial_state = np.array(json.loads(initial_state_str), dtype=complex)
            else: # Already an array
                initial_state = initial_state_str


            if isinstance(system, str):
                system_instance = self.available_systems.get(system)
                if system_instance is None:
                    return f"Error: System '{system}' not found."
                system_instance = system_instance() # Create Instance
                self.qcm.system_instances[system] = system_instance # Store the instance

            else: # It's already an instance
                system_instance = system

            #No need to do this here, it's handled by Gemma.
            # if isinstance(system_instance, HarmonicOscillator) and system_instance.spin is not None:
            #     if len(initial_state) != 2 * len(system_instance.x):
            #          initial_state = np.tile(initial_state, len(system_instance.x) // (len(initial_state) // 2) + 1)[:len(system_instance.x)*2]

            states = system_instance.time_evolve(initial_state, int(time_steps), float(dt))

            return f"Time evolution completed for {type(system_instance).__name__}." #Return system name

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print("Error in run_time_evolution_tool:") #Debugging
            return f"Error in time evolution: {e}"
        except Exception as e:
            print("Error in run_time_evolution")
            return f"Error: {e}"

    def visualize_system_tool(self, system):
        """Generates a visualization for the selected system using Plotly."""
        try:

            if isinstance(system, str):
                system_instance = self.available_systems.get(system)
                if system_instance is None:
                    return f"Error: System '{system}' not found"
                system_instance = self.qcm.system_instances.get(system) # Get the instance
                if system_instance is None:
                    return f"Error: System Instance not found"
            else:
                system_instance = system

            if hasattr(system_instance, 'solve_system'):
                try:
                    system_instance.solve_system()  # Ensure solve_system is called
                except:
                    pass # If solve_system doesn't exist, that's ok

            if isinstance(system_instance, HarmonicOscillator):
                return system_instance.visualize() # Call directly

            # elif isinstance(system_instance, CoupledOscillators):
            #     return system_instance.visualize()
            # elif isinstance(system_instance, ParticleInABox):
            #     return system_instance.visualize()
            # # elif isinstance(system_instance, HydrogenAtom):  # Enable later
            # #     return system_instance.visualize()
            else:
                return "Visualization not implemented for this system."

        except Exception as e:
            print("Error in visualize_system_tool:") # Debugging
            return f"Error in visualization: {e}"
    def create_interface(self):
      """Creates the Gradio interface."""
      with gr.Blocks() as interface:
          gr.Markdown("# Quantum AI Assistant")
          with gr.Tab("Chat"):
              chatbot = gr.Chatbot()
              user_input = gr.Textbox(label="Input")
              submit_btn = gr.Button("Submit")
              audio_output = gr.Audio(label="Assistant Response", autoplay=True) # Add audio output
              # system_select MUST be defined before it is used in submit_btn.click
              # It was originally defined later within a different gr.Tab
              system_select = gr.Dropdown(
                  label="Select Quantum System",
                  choices=list(self.available_systems.keys()),
                  value=self.current_system_name
              )
              submit_btn.click(self.qcm.process_input, [user_input, system_select], [chatbot, audio_output])


          with gr.Tab("System Parameters"):
              with gr.Row():
                  with gr.Column():
                      with gr.Accordion(label="Harmonic Oscillator Parameters", open=False) as ho_params:
                          ho_spin_checkbox = gr.Checkbox(label="Include Spin", value=False)
                          ho_b_field_input = gr.Slider(label="Magnetic Field Strength", minimum=0, maximum=10, step=0.1, value=0)
                          ho_alpha_input = gr.Slider(label="Alpha (Coupling)", minimum=0, maximum=1, step=0.01, value=0.1)
                          ho_A_expr_input = gr.Textbox(label="A(t) (e.g., 54 + 10*sin(0.1*t))", value="54")
                          ho_omega_expr_input = gr.Textbox(label="ω(t) (e.g., 0.0066)", value="0.0066")
                          ho_phi_expr_input = gr.Textbox(label="φ(t) (e.g., 0)", value="0")
                          ho_coupling_strength_input = gr.Slider(label="Coupling Strength", minimum=0, maximum=1, step=0.01, value=0.1)
                          ho_omega0_input = gr.Number(label="ω₀ (Natural Frequency)", value=1.0, precision=2, minimum=0.1) #Added
                          ho_update_params_button = gr.Button("Update HO Parameters")

                  with gr.Column():
                      param_status_output = gr.Textbox(label="Status")

          with gr.Tab("Simulation Results"):
              with gr.Row():
                  with gr.Column():
                      run_sim_button = gr.Button("Run Simulation")
                      run_time_evol_button = gr.Button("Run Time Evolution")
                      initial_state_input = gr.Textbox(label="Initial State (e.g., [1, 0])", value="[1, 0]")
                      time_steps_input = gr.Slider(label="Time Steps", minimum=10, maximum=1000, step=10, value=100)
                      dt_input = gr.Slider(label="dt", minimum=0.01, maximum=1.0, step=0.01, value=0.1)
                  with gr.Column():
                      sim_plot_output = gr.Plot(label="Simulation Results")  # Use gr.Plot for Plotly
                      time_evol_plot_output = gr.Plot(label="Time Evolution") # Use gr.Plot


          # --- Event Handlers ---
          system_select.change(self.update_system, inputs=system_select, outputs=[ho_params])

          ho_update_params_button.click(
              self.update_ho_parameters,
              inputs=[ho_A_expr_input, ho_omega_expr_input, ho_phi_expr_input, ho_spin_checkbox, ho_b_field_input, ho_alpha_input, ho_coupling_strength_input, ho_omega0_input],
              outputs=param_status_output
          )

          run_sim_button.click(self.run_simulation_tool_wrapper, [], sim_plot_output)
          run_time_evol_button.click(self.run_time_evolution_tool_wrapper, [initial_state_input, time_steps_input, dt_input], time_evol_plot_output)
          submit_btn.click(self.qcm.process_input, [user_input, system_select], [chatbot, audio_output])

      return interface

      # --- Helper methods for Gradio ---
    def update_system(self, system_name: str):
        """Updates the current system and shows/hides parameter blocks."""
        self.current_system_name = system_name
        # self.current_system = self.available_systems[system_name]() # Create a NEW INSTANCE
        if system_name not in self.qcm.system_instances:
            self.qcm.system_instances[system_name] = self.available_systems[system_name]()
        # Show/hide parameter blocks as needed
        return {
            ho_params: gr.update(visible=(system_name == "Harmonic Oscillator")),
        }

    def update_ho_parameters(self, A_expr: str, omega_expr: str, phi_expr: str, use_spin: bool, B_field: float, alpha:float, coupling_strength: float, omega0: float):
        """Updates HO parameters."""
        try:
            A_func = lambda t: eval(A_expr)
            omega_func = lambda t: eval(omega_expr)
            phi_func = lambda t: eval(phi_expr)

            # Update the instance in the dictionary
            self.qcm.system_instances["Harmonic Oscillator"] = HarmonicOscillator(
                spin=0.5 if use_spin else None,
                B_field=B_field if use_spin else None,
                alpha = alpha,
                coupling_strength=coupling_strength,
                omega0 = omega0
            )
            self.qcm.system_instances["Harmonic Oscillator"].set_parameter_functions(A_func, omega_func, phi_func)
            return "Harmonic Oscillator parameters updated."
        except Exception as e:
            return f"Error: Invalid function expression: {e}"

    def run_simulation_tool_wrapper(self):
      """Wrapper for run_simulation_tool to handle Plotly figures."""
      result = self.run_simulation_tool(self.current_system_name)
      if isinstance(result, str):  # If it's an error message
          return result
      else:  # Assume it's a Plotly figure if not an error
          return result

    def run_time_evolution_tool_wrapper(self, initial_state_input, time_steps_input, dt_input):
      result = self.run_time_evolution_tool(self.current_system_name, initial_state_input, time_steps_input, dt_input)
      if isinstance(result, str):
          return result
      #Since time evolution doesn't generate a plot, we call visualize
      elif "Time evolution completed" in result:
          return self.visualize_system_tool(self.current_system_name)
      else:
          return result

# --- Main Execution ---

def main():
    assistant = QuantumAssistant()
    interface = assistant.create_interface()
    interface.launch(debug=True)  # Use debug=True for easier debugging

if __name__ == "__main__":
    main()