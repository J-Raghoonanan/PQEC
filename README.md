# README.md


# Swap_QEC
Leveraging Child's swap purification protocol along with Grafe's non-linear ultrahigh threshold QEC formalism to develop a new QEC scheme

# Streaming Quantum State Purification: Comprehensive Data Generation

Complete data generation and figure creation pipeline for the streaming quantum error correction protocol paper.

## Overview

This package generates ALL data and figures needed for a comprehensive quantum error correction paper on streaming purification protocols. It combines the theoretical advantages of nonlinear encoding with the practical benefits of streaming implementation.

## Key Features

- **Complete Data Pipeline**: Single command generates all paper data
- **Publication-Quality Figures**: Automatic PDF generation for all plots
- **Memory Scaling Analysis**: Demonstrates O(log N) advantage over standard QEC
- **Phase Diagrams**: Success/failure regions for different protocols
- **QEC Comparisons**: Performance against existing codes
- **Noise Model Analysis**: Why depolarizing works better than Pauli errors

## Generated Data Types

### Core Protocol Data:
- **Evolution Data**: Error/fidelity evolution through purification levels
- **Threshold Data**: Error thresholds across dimensions and code sizes  
- **Resource Data**: Memory usage and gate complexity analysis

### Critical Missing Elements Added:
- **Memory Scaling**: O(log N) vs O(N) comparison (YOUR KEY ADVANTAGE)
- **Phase Diagrams**: Success/failure regions (publication standard)
- **QEC Comparisons**: Performance vs existing protocols
- **Convergence Analysis**: Detailed purity evolution validation
- **Noise Model Analysis**: Why depolarizing > Pauli errors
- **Amplification Efficiency**: Amplitude amplification overhead

## Generated Figures

### Main Paper Figures:
1. `memory_scaling_advantage.pdf` - **KEY FIGURE**: Memory scaling advantage
2. `main_figure_combined.pdf` - Combined overview figure
3. `qec_protocol_comparison.pdf` - Competitive positioning
4. `phase_diagram_depolarizing_d2.pdf` - Operating regime

### Supporting Figures:
- Error/fidelity evolution plots
- Convergence analysis
- Threshold behavior analysis  
- Noise model comparisons
- Resource overhead analysis
- Amplification efficiency plots

## Directory Structure

```
project/
├── src/                                    # Your existing source code
│   ├── streaming_protocol.py
│   ├── quantum_states.py
│   ├── noise_models.py
│   └── swap_operations.py
│   └── comprehensive_data_generation.py   # NEW: Complete data generation
├── requirements.txt                       # Dependencies
├── README.md                              # This file
├── data/                                  # Generated data (auto-created)
│   ├── evolution/
│   ├── threshold/
│   ├── memory_scaling/
│   ├── phase_diagrams/
│   ├── qec_comparisons/
│   ├── convergence/
│   ├── noise_analysis/
│   ├── amplification/
│   ├── resource/
│   └── metadata/
└── figures/                               # Generated figures (auto-created)
    ├── figure_generation.py
    ├── memory_scaling_advantage.pdf 
    ├── main_figure_combined.pdf
    ├── qec_protocol_comparison.pdf
    └── ... (12 total figures)
```

## Key Advantages Demonstrated

### 1. Memory Scaling (Your Main Advantage)
- **Streaming QEC**: O(log N) memory
- **Standard QEC**: O(N) memory  
- **Advantage**: Up to 64× memory reduction for large systems

### 2. High Error Thresholds
- **Depolarizing noise**: Up to 50% error rates
- **Surface codes**: ~1% error rates
- **Advantage**: 50× higher error tolerance

### 3. Streaming Implementation
- **Constant memory**: Independent of computation length
- **Practical benefit**: Enables large-scale quantum computing

### 4. Deterministic Operation
- **Amplitude amplification**: Eliminates probabilistic failures
- **Reliability**: Predictable error correction performance

## Limitations Identified

### 1. Noise Model Dependence
- **Perfect for depolarizing**: Theoretical optimality
- **Limited for Pauli**: Cross-coherence terms reduce effectiveness

### 2. Implementation Requirements
- **Identical states**: Requires multiple copies of same quantum state
- **Gate overhead**: Amplitude amplification adds complexity.
