"""
Unit tests for IBMQ_components.py - Batch SWAP Purification Implementation

Tests cover:
- Configuration validation
- Circuit construction functions
- Measurement and analysis logic  
- Error handling and edge cases
- Integration functionality

Run with: pytest test_IBMQ_components.py -v
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from qiskit import QuantumCircuit
from qiskit.result import Result

# Import the module we're testing
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from IBMQ_components import (
    PurificationConfig,
    create_hadamard_target_circuit,
    apply_depolarizing_noise,
    create_batch_purification_circuit,
    add_measurements,
    analyze_results,
    execute_with_retry,
    setup_ibm_backend,
    transpile_circuit_for_backend,
    run_complete_purification_experiment,
)


# =============================================================================
# Configuration Tests
# =============================================================================

class TestPurificationConfig:
    """Test the PurificationConfig dataclass."""
    
    def test_valid_config(self):
        """Test that valid configurations pass validation."""
        config = PurificationConfig(M=2, N=4, p=0.1)
        config.validate()  # Should not raise
        
        assert config.M == 2
        assert config.N == 4
        assert config.p == 0.1
        assert config.synthesize_run_id() == "batch_M2_N4_p0.1000"
    
    def test_invalid_M(self):
        """Test validation fails for invalid M values."""
        with pytest.raises(ValueError, match="M must be in"):
            PurificationConfig(M=0, N=4, p=0.1).validate()
        
        with pytest.raises(ValueError, match="M must be in"):
            PurificationConfig(M=10, N=4, p=0.1).validate()
    
    def test_invalid_N(self):
        """Test validation fails for invalid N values."""
        # N must be power of 2
        with pytest.raises(ValueError, match="N must be a power of 2"):
            PurificationConfig(M=2, N=3, p=0.1).validate()
        
        # N must be > 1
        with pytest.raises(ValueError, match="N must be a power of 2"):
            PurificationConfig(M=2, N=1, p=0.1).validate()
    
    def test_invalid_p(self):
        """Test validation fails for invalid p values."""
        with pytest.raises(ValueError, match="p must be in"):
            PurificationConfig(M=2, N=4, p=-0.1).validate()
        
        with pytest.raises(ValueError, match="p must be in"):
            PurificationConfig(M=2, N=4, p=1.5).validate()
    
    def test_qubit_limit(self):
        """Test validation fails when too many qubits required."""
        # This should fail due to qubit limits
        # Use M=8 (max allowed) but large N to exceed total qubit limit
        with pytest.raises(ValueError, match="exceeds backend limit"):
            PurificationConfig(M=8, N=16, p=0.1).validate()  # 8*16 + 4 = 132 qubits > 127
    
    def test_edge_case_configurations(self):
        """Test edge cases that should be valid."""
        # Minimum valid config
        config1 = PurificationConfig(M=1, N=2, p=0.0)
        config1.validate()
        
        # Maximum reasonable config
        config2 = PurificationConfig(M=3, N=8, p=1.0)
        config2.validate()
        
        # Powers of 2 for N
        for N in [2, 4, 8, 16]:
            config = PurificationConfig(M=2, N=N, p=0.1)
            config.validate()


# =============================================================================
# Circuit Construction Tests
# =============================================================================

class TestCircuitConstruction:
    """Test circuit construction functions."""
    
    def test_create_hadamard_target_circuit(self):
        """Test Hadamard target state preparation."""
        for M in [1, 2, 3, 4]:
            qc = create_hadamard_target_circuit(M)
            
            assert qc.num_qubits == M
            assert qc.num_clbits == 0
            assert qc.name == f"hadamard_target_M{M}"
            
            # Should have exactly M Hadamard gates
            h_count = sum(1 for instr, qargs, cargs in qc.data if instr.name == 'h')
            assert h_count == M
    
    def test_create_hadamard_invalid_input(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="M must be positive"):
            create_hadamard_target_circuit(0)
        
        with pytest.raises(ValueError, match="M must be positive"):
            create_hadamard_target_circuit(-1)
    
    def test_apply_depolarizing_noise(self):
        """Test depolarizing noise application."""
        qc = QuantumCircuit(3)
        original_depth = qc.depth()
        
        # Test with p=0 (no noise)
        apply_depolarizing_noise(qc, [0, 1, 2], p=0.0, seed=42)
        assert qc.depth() == original_depth  # No gates added
        
        # Test with p>0 (should add some gates)
        qc_noisy = QuantumCircuit(3)
        apply_depolarizing_noise(qc_noisy, [0, 1, 2], p=0.5, seed=42)
        # With p=0.5 and seed=42, should add some Pauli gates
        assert qc_noisy.depth() >= original_depth
        
        # Test reproducibility with same seed
        qc_noisy2 = QuantumCircuit(3) 
        apply_depolarizing_noise(qc_noisy2, [0, 1, 2], p=0.5, seed=42)
        assert qc_noisy.depth() == qc_noisy2.depth()
    
    def test_apply_depolarizing_noise_invalid_p(self):
        """Test error handling for invalid p values."""
        qc = QuantumCircuit(2)
        
        with pytest.raises(ValueError, match="p must be in"):
            apply_depolarizing_noise(qc, [0, 1], p=-0.1)
        
        with pytest.raises(ValueError, match="p must be in"):
            apply_depolarizing_noise(qc, [0, 1], p=1.5)
    
    def test_create_batch_purification_circuit(self):
        """Test batch purification circuit construction."""
        config = PurificationConfig(M=2, N=4, p=0.1)
        qc = create_batch_purification_circuit(config)
        
        # Check basic structure
        expected_data_qubits = config.N * config.M  # 4 * 2 = 8
        expected_ancillas = int(np.log2(config.N))  # log2(4) = 2
        expected_total_qubits = expected_data_qubits + expected_ancillas  # 8 + 2 = 10
        
        assert qc.num_qubits == expected_total_qubits
        assert qc.name == "batch_purification_M2_N4"
        
        # Should have Hadamard gates for state preparation
        h_count = sum(1 for instr in qc.data if instr.operation.name == 'h')
        # M hadamards per copy + 2 hadamards per SWAP test 
        # For N=4: 3 SWAP tests total (2 in level 0, 1 in level 1)
        # So: 4*2 + 2*3 = 8 + 6 = 14 hadamards total
        expected_h = config.N * config.M + 2 * (config.N - 1)  # N-1 total SWAP tests in tree
        assert h_count == expected_h
        
        # Should have CSWAP gates for purification
        cswap_count = sum(1 for instr, qargs, cargs in qc.data if instr.name == 'cswap')
        # Each SWAP test needs M CSWAPs, total levels is log2(N)
        expected_cswaps = config.M * (config.N - 1)  # Tree structure
        assert cswap_count == expected_cswaps
    
    def test_add_measurements(self):
        """Test measurement addition to circuits."""
        config = PurificationConfig(M=2, N=4, p=0.1)
        base_circuit = create_batch_purification_circuit(config)
        
        measured_circuit = add_measurements(base_circuit, config)
        
        # Should add classical registers
        assert len(measured_circuit.cregs) == 2
        
        # Check classical register sizes
        final_state_reg = None
        ancilla_reg = None
        for creg in measured_circuit.cregs:
            if creg.name == 'final_state':
                final_state_reg = creg
            elif creg.name == 'ancillas':
                ancilla_reg = creg
        
        assert final_state_reg is not None
        assert ancilla_reg is not None
        assert final_state_reg.size == config.M
        assert ancilla_reg.size == int(np.log2(config.N))
        
        # Should have measurement operations
        measure_count = sum(1 for instr, qargs, cargs in measured_circuit.data if instr.name == 'measure')
        expected_measures = config.M + int(np.log2(config.N))  # final state + ancillas
        assert measure_count == expected_measures


# =============================================================================
# Analysis Tests
# =============================================================================

class TestAnalysis:
    """Test result analysis functions."""
    
    def test_analyze_results_perfect_case(self):
        """Test analysis with perfect results."""
        config = PurificationConfig(M=2, N=4, p=0.1)
        
        # Perfect case: all ancillas=0, final state=00
        counts = {
            "0000": 1000  # final_state=00, ancillas=00
        }
        
        fidelity, success_prob = analyze_results(counts, config)
        
        assert fidelity == 1.0  # Perfect fidelity
        assert success_prob == 1.0  # All SWAP tests succeeded
    
    def test_analyze_results_partial_success(self):
        """Test analysis with partial success."""
        config = PurificationConfig(M=2, N=4, p=0.1)
        
        # Mixed case: some successes, some failures
        counts = {
            "0000": 300,  # Perfect: final=00, ancillas=00
            "0100": 200,  # Imperfect: final=01, ancillas=00  
            "0001": 500   # Failed SWAP: ancillas=01
        }
        
        fidelity, success_prob = analyze_results(counts, config)
        
        # Success probability: (300 + 200) / 1000 = 0.5
        assert success_prob == 0.5
        
        # Fidelity given success: 300 / (300 + 200) = 0.6
        assert fidelity == 0.6
    
    def test_analyze_results_no_success(self):
        """Test analysis when no SWAP tests succeed."""
        config = PurificationConfig(M=2, N=4, p=0.1)
        
        # All SWAP tests failed
        counts = {
            "0001": 400,  # ancillas=01
            "0010": 300,  # ancillas=10
            "0011": 300   # ancillas=11
        }
        
        fidelity, success_prob = analyze_results(counts, config)
        
        assert success_prob == 0.0
        assert fidelity == 0.0
    
    def test_analyze_results_empty_counts(self):
        """Test analysis with empty results."""
        config = PurificationConfig(M=2, N=4, p=0.1)
        
        fidelity, success_prob = analyze_results({}, config)
        
        assert fidelity == 0.0
        assert success_prob == 0.0
    
    def test_analyze_results_different_configurations(self):
        """Test analysis with different M and N values."""
        # Test M=1, N=2
        config1 = PurificationConfig(M=1, N=2, p=0.1)
        counts1 = {"00": 500, "01": 500}  # final=0, ancilla=0 vs ancilla=1
        fidelity1, success_prob1 = analyze_results(counts1, config1)
        assert success_prob1 == 0.5
        assert fidelity1 == 1.0  # Perfect when successful
        
        # Test M=3, N=8  
        config2 = PurificationConfig(M=3, N=8, p=0.1)
        # For M=3, N=8: final_state=3 bits, ancillas=3 bits, total=6 bits
        counts2 = {"000000": 800, "000001": 200}  # final=000, ancillas=000 vs ancillas=001
        fidelity2, success_prob2 = analyze_results(counts2, config2)
        assert success_prob2 == 0.8  # 800/1000 successful (ancillas=000)
        assert fidelity2 == 1.0      # Perfect final state when successful


# =============================================================================
# Integration Tests (Mocked)
# =============================================================================

class TestIntegration:
    """Test integration functionality with mocked IBM components."""
    
    @patch('IBMQ_components.QiskitRuntimeService')
    def test_setup_ibm_backend(self, mock_service_class):
        """Test IBM backend setup."""
        # Mock the service and backend
        mock_service = Mock()
        mock_backend = Mock()
        mock_backend.status.return_value = "operational"
        mock_service.backend.return_value = mock_backend
        mock_service_class.return_value = mock_service
        
        service, backend = setup_ibm_backend("ibm_torino")
        
        assert service == mock_service
        assert backend == mock_backend
        mock_service.backend.assert_called_once_with("ibm_torino")
    
    @patch('IBMQ_components.generate_preset_pass_manager')
    def test_transpile_circuit_for_backend(self, mock_pass_manager_gen):
        """Test circuit transpilation."""
        # Create mock components
        qc = QuantumCircuit(2)
        mock_backend = Mock()
        mock_pass_manager = Mock()
        mock_transpiled_circuit = Mock()
        
        mock_pass_manager.run.return_value = mock_transpiled_circuit
        mock_pass_manager_gen.return_value = mock_pass_manager
        
        result = transpile_circuit_for_backend(qc, mock_backend, optimization_level=2)
        
        assert result == mock_transpiled_circuit
        mock_pass_manager_gen.assert_called_once_with(
            optimization_level=2,
            backend=mock_backend
        )
        mock_pass_manager.run.assert_called_once_with(qc)
    
    @patch('IBMQ_components.SamplerV2')
    def test_execute_with_retry_success(self, mock_sampler_class):
        """Test successful execution with retry logic."""
        config = PurificationConfig(M=2, N=4, p=0.1, shots=1000, min_success_rate=0.1)
        circuit = QuantumCircuit(10, 4)  # 10 qubits, 4 classical bits
        
        # Mock successful execution
        mock_job = Mock()
        mock_result_item = Mock()
        mock_data = Mock()
        mock_meas = Mock()
        mock_meas.get_counts.return_value = {
            "0000": 200,  # Successful cases
            "0100": 300,  
            "0001": 500   # Failed cases
        }
        mock_data.meas = mock_meas
        mock_result_item.data = mock_data
        
        # result is a list-like object where result[0] gives the data
        mock_job.result.return_value = [mock_result_item]
        
        mock_sampler = Mock()
        mock_sampler.run.return_value = mock_job
        mock_sampler_class.return_value = mock_sampler
        
        mock_service = Mock()
        mock_backend = Mock()
        
        # Mock transpilation
        with patch('IBMQ_components.transpile_circuit_for_backend') as mock_transpile:
            mock_transpile.return_value = circuit
            
            fidelity, success_prob, counts = execute_with_retry(
                circuit, config, mock_service, mock_backend)
        
        # Should succeed on first attempt
        assert success_prob == 0.5  # 500/1000
        assert fidelity == 0.4  # 200/500
        assert mock_sampler.run.call_count == 1
        
        # Verify SamplerV2 was called correctly (backend only, no session)
        mock_sampler_class.assert_called_once_with(backend=mock_backend)
    
    @patch('IBMQ_components.setup_ibm_backend')
    @patch('IBMQ_components.execute_with_retry') 
    def test_run_complete_purification_experiment(self, mock_execute, mock_setup):
        """Test the complete experiment workflow."""
        config = PurificationConfig(M=2, N=4, p=0.1)
        
        # Mock backend setup
        mock_service = Mock()
        mock_backend = Mock()
        mock_setup.return_value = (mock_service, mock_backend)
        
        # Mock execution
        mock_execute.return_value = (0.8, 0.6, {"0000": 800, "0001": 200})
        
        result = run_complete_purification_experiment(config)
        
        # Check result structure
        assert result['run_id'] == config.synthesize_run_id()
        assert result['M'] == config.M
        assert result['N'] == config.N
        assert result['p'] == config.p
        assert result['max_purification_level'] == int(np.log2(config.N))
        assert result['final_fidelity'] == 0.8
        assert result['swap_success_probability'] == 0.6
        assert result['backend_name'] == config.backend_name
        
        # Verify functions were called
        mock_setup.assert_called_once_with(config.backend_name)
        mock_execute.assert_called_once()


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_small_circuit(self):
        """Test minimum viable configuration."""
        config = PurificationConfig(M=1, N=2, p=0.0)
        qc = create_batch_purification_circuit(config)
        
        assert qc.num_qubits == 3  # 2 data qubits + 1 ancilla
        measured_qc = add_measurements(qc, config)
        
        # Should work without errors
        counts = {"00": 1000}  # Perfect case
        fidelity, success_prob = analyze_results(counts, config)
        assert fidelity == 1.0
        assert success_prob == 1.0
    
    def test_larger_configuration(self):
        """Test larger but still reasonable configuration."""
        config = PurificationConfig(M=3, N=8, p=0.2)
        qc = create_batch_purification_circuit(config)
        
        expected_qubits = 8 * 3 + 3  # 8 registers of 3 qubits + 3 ancillas
        assert qc.num_qubits == expected_qubits
        
        # Should create valid circuit structure
        measured_qc = add_measurements(qc, config)
        assert measured_qc.num_clbits == 3 + 3  # final state + ancillas
    
    def test_noise_edge_cases(self):
        """Test noise application edge cases."""
        qc = QuantumCircuit(2)
        
        # p=1.0 should always apply errors
        np.random.seed(42)
        original_depth = qc.depth()
        apply_depolarizing_noise(qc, [0, 1], p=1.0, seed=42)
        assert qc.depth() > original_depth
        
        # Empty qubit list should do nothing
        qc_empty = QuantumCircuit(2)
        apply_depolarizing_noise(qc_empty, [], p=0.5, seed=42)
        assert qc_empty.depth() == 0
    
    def test_malformed_measurement_outcomes(self):
        """Test handling of malformed measurement results."""
        config = PurificationConfig(M=2, N=4, p=0.1)
        
        # Test with wrong outcome length
        counts = {
            "000": 100,     # Too short
            "00000": 200,   # Too long  
            "0000": 700     # Correct length
        }
        
        fidelity, success_prob = analyze_results(counts, config)
        
        # Should only use the correctly formatted outcome
        assert success_prob == 0.7  # Only the "0000" outcome counted
    
    def test_configuration_boundary_conditions(self):
        """Test configuration at boundary conditions."""
        # Maximum M we allow
        config_max_M = PurificationConfig(M=8, N=2, p=0.1)
        config_max_M.validate()
        
        # Various powers of 2 for N
        for N in [2, 4, 8, 16]:
            config = PurificationConfig(M=1, N=N, p=0.5)
            config.validate()
            qc = create_batch_purification_circuit(config)
            assert qc.num_qubits == N + int(np.log2(N))


# =============================================================================
# Performance and Stress Tests  
# =============================================================================

class TestPerformance:
    """Test performance-related aspects (not actual timing, just structure)."""
    
    def test_circuit_scaling(self):
        """Test that circuit size scales as expected."""
        base_config = PurificationConfig(M=2, N=2, p=0.1)
        base_circuit = create_batch_purification_circuit(base_config)
        base_qubits = base_circuit.num_qubits
        base_depth = base_circuit.depth()
        
        # Doubling N should add more qubits and depth
        larger_config = PurificationConfig(M=2, N=4, p=0.1)
        larger_circuit = create_batch_purification_circuit(larger_config)
        
        assert larger_circuit.num_qubits > base_qubits
        assert larger_circuit.depth() > base_depth
    
    def test_deterministic_circuit_construction(self):
        """Test that circuit construction is deterministic."""
        config = PurificationConfig(M=2, N=4, p=0.1)
        
        circuit1 = create_batch_purification_circuit(config)
        circuit2 = create_batch_purification_circuit(config) 
        
        # Should generate identical circuits
        assert circuit1.num_qubits == circuit2.num_qubits
        assert circuit1.depth() == circuit2.depth()
        assert len(circuit1.data) == len(circuit2.data)


# =============================================================================
# Fixtures and Helpers
# =============================================================================

@pytest.fixture
def sample_config():
    """Provide a standard test configuration."""
    return PurificationConfig(M=2, N=4, p=0.1)

@pytest.fixture
def sample_counts():
    """Provide sample measurement counts for testing."""
    return {
        "0000": 300,  # Perfect success
        "0100": 200,  # Imperfect success  
        "1000": 100,  # Imperfect success
        "0001": 250,  # SWAP failure
        "0010": 150   # SWAP failure
    }


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])