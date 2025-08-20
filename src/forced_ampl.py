"""
Forced Amplitude Amplification - Making the Protocol Actually Use Amplification

Instead of N_iter = max(0, floor(...)), we use N_iter = max(1, floor(...))
to force at least one iteration and see what happens.
"""

import numpy as np
from typing import Dict, List, Tuple

class ForcedAmplificationSimulator:
    """
    Simulator that forces amplitude amplification to see the Q operator in action.
    """
    
    def __init__(self, dimension: int = 2, force_minimum_iterations: int = 1, verbose: bool = True):
        self.dimension = dimension
        self.force_minimum_iterations = force_minimum_iterations
        self.verbose = verbose
    
    def calculate_iterations_comparison(self, purity: float) -> Dict:
        """
        Compare theoretical optimal vs forced iterations.
        """
        lambda_val = purity
        delta = 1 - lambda_val
        d = self.dimension
        
        # Calculate success probability
        success_prob = 0.5 * (1 + (1-delta)**2 + delta*(2-delta)/d)
        
        if success_prob <= 0.001 or success_prob >= 0.999:
            return {
                'success_prob': success_prob,
                'theoretical_iters': 0,
                'forced_iters': 0,
                'error': 'Success probability out of range'
            }
        
        # Theoretical optimal iterations
        theoretical_iters = max(0, int(np.floor(np.pi / (4*np.arcsin(np.sqrt(success_prob))) - 0.5)))
        
        # Forced minimum iterations
        forced_iters = max(self.force_minimum_iterations, theoretical_iters)
        
        # Calculate amplitude evolution for both cases
        theta = 2 * np.arcsin(np.sqrt(success_prob))
        
        # Theoretical case
        if theoretical_iters == 0:
            theoretical_final_prob = success_prob
            theoretical_amplitude_evolution = [np.sqrt(success_prob)]
        else:
            theoretical_final_amplitude = np.sin((2*theoretical_iters + 1) * theta / 2)
            theoretical_final_prob = theoretical_final_amplitude**2
            theoretical_amplitude_evolution = [np.sin((2*k + 1) * theta / 2) for k in range(theoretical_iters + 1)]
        
        # Forced case
        forced_amplitude_evolution = [np.sin((2*k + 1) * theta / 2) for k in range(forced_iters + 1)]
        forced_final_amplitude = forced_amplitude_evolution[-1]
        forced_final_prob = forced_final_amplitude**2
        
        # Compare improvement
        improvement_theoretical = theoretical_final_prob - success_prob
        improvement_forced = forced_final_prob - success_prob
        
        return {
            'purity': purity,
            'success_prob': success_prob,
            'theta': theta,
            'theoretical_iters': theoretical_iters,
            'forced_iters': forced_iters,
            'theoretical_final_prob': theoretical_final_prob,
            'forced_final_prob': forced_final_prob,
            'theoretical_amplitude_evolution': theoretical_amplitude_evolution,
            'forced_amplitude_evolution': forced_amplitude_evolution,
            'improvement_theoretical': improvement_theoretical,
            'improvement_forced': improvement_forced,
            'forced_is_better': improvement_forced > improvement_theoretical,
            'amplitude_amplification_factor': forced_final_prob / success_prob if success_prob > 0 else 1
        }
    
    def test_forced_amplification_range(self) -> Dict:
        """
        Test forced amplification across different parameter ranges.
        """
        
        print(f"TESTING FORCED AMPLITUDE AMPLIFICATION")
        print(f"Minimum iterations forced: {self.force_minimum_iterations}")
        print(f"Dimension: {self.dimension}")
        print("="*60)
        
        # Test across different purity levels
        purity_levels = np.concatenate([
            np.linspace(0.1, 0.9, 20),      # Standard range
            np.linspace(0.01, 0.1, 20),    # Low purity
            np.linspace(0.9, 0.99, 10),    # High purity
        ])
        
        results = []
        cases_where_forced_helps = []
        cases_where_forced_hurts = []
        
        print(f"\nTesting {len(purity_levels)} purity levels:")
        
        for i, purity in enumerate(purity_levels):
            result = self.calculate_iterations_comparison(purity)
            
            if 'error' in result:
                continue
                
            results.append(result)
            
            # Categorize results
            if result['forced_is_better']:
                cases_where_forced_helps.append(result)
            elif result['improvement_forced'] < result['improvement_theoretical']:
                cases_where_forced_hurts.append(result)
            
            # Print interesting cases
            if (result['theoretical_iters'] == 0 and result['forced_iters'] > 0) or result['forced_is_better']:
                print(f"  λ={purity:.3f}: P_success={result['success_prob']:.4f}")
                print(f"    Theoretical: {result['theoretical_iters']} iter → P={result['theoretical_final_prob']:.6f}")
                print(f"    Forced:      {result['forced_iters']} iter → P={result['forced_final_prob']:.6f}")
                print(f"    Forced better: {result['forced_is_better']}")
                print(f"    Amplification factor: {result['amplitude_amplification_factor']:.3f}")
                print()
        
        # Analysis
        total_cases = len(results)
        forced_helps_count = len(cases_where_forced_helps)
        forced_hurts_count = len(cases_where_forced_hurts)
        
        print(f"FORCED AMPLIFICATION ANALYSIS:")
        print(f"  Total cases: {total_cases}")
        print(f"  Cases where forced helps: {forced_helps_count} ({100*forced_helps_count/total_cases:.1f}%)")
        print(f"  Cases where forced hurts: {forced_hurts_count} ({100*forced_hurts_count/total_cases:.1f}%)")
        
        if cases_where_forced_helps:
            best_case = max(cases_where_forced_helps, key=lambda x: x['improvement_forced'])
            print(f"  Best forced case: λ={best_case['purity']:.3f}")
            print(f"    Improvement: {best_case['improvement_forced']:.6f}")
            print(f"    Amplification factor: {best_case['amplitude_amplification_factor']:.3f}")
        
        return {
            'results': results,
            'cases_where_forced_helps': cases_where_forced_helps,
            'cases_where_forced_hurts': cases_where_forced_hurts,
            'summary': {
                'total_cases': total_cases,
                'forced_helps_count': forced_helps_count,
                'forced_hurts_count': forced_hurts_count,
                'forced_helps_rate': forced_helps_count / total_cases if total_cases > 0 else 0
            }
        }
    
    def demonstrate_q_operator_evolution(self, purity: float, max_iterations: int = 10) -> Dict:
        """
        Show the Q operator evolution step by step.
        """
        
        print(f"\nDEMONSTRATING Q OPERATOR EVOLUTION")
        print(f"Purity λ = {purity:.6f}, max iterations = {max_iterations}")
        print("-"*50)
        
        lambda_val = purity
        delta = 1 - lambda_val
        d = self.dimension
        
        # Calculate success probability
        success_prob = 0.5 * (1 + (1-delta)**2 + delta*(2-delta)/d)
        theta = 2 * np.arcsin(np.sqrt(success_prob))
        
        print(f"Initial success probability: {success_prob:.6f}")
        print(f"θ = 2*arcsin(√P) = {theta:.6f}")
        print(f"Theoretical optimal iterations: {max(0, int(np.floor(np.pi / (4*np.arcsin(np.sqrt(success_prob))) - 0.5)))}")
        print()
        
        # Calculate evolution step by step
        evolution_data = []
        
        print("Q Operator Evolution:")
        print("Iter | Amplitude | Probability | Improvement")
        print("-----|-----------|-------------|------------")
        
        for k in range(max_iterations + 1):
            if k == 0:
                amplitude = np.sqrt(success_prob)
                probability = success_prob
            else:
                amplitude = np.sin((2*k + 1) * theta / 2)
                probability = amplitude**2
            
            improvement = probability - success_prob
            
            evolution_data.append({
                'iteration': k,
                'amplitude': amplitude,
                'probability': probability,
                'improvement': improvement
            })
            
            print(f"{k:4d} | {amplitude:9.6f} | {probability:11.6f} | {improvement:+11.6f}")
        
        # Find the best iteration
        best_iteration = max(evolution_data, key=lambda x: x['probability'])
        
        print(f"\nBest iteration: {best_iteration['iteration']}")
        print(f"Best probability: {best_iteration['probability']:.6f}")
        print(f"Maximum improvement: {best_iteration['improvement']:.6f}")
        
        return {
            'purity': purity,
            'success_prob': success_prob,
            'theta': theta,
            'evolution_data': evolution_data,
            'best_iteration': best_iteration
        }

def test_different_forced_minimums() -> Dict:
    """
    Test different forced minimum iteration counts.
    """
    
    print("="*70)
    print("TESTING DIFFERENT FORCED MINIMUM ITERATIONS")
    print("="*70)
    
    forced_minimums = [1, 2, 3, 5, 10]
    dimensions = [2, 4, 8, 16]
    test_purities = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    all_results = {}
    
    for min_iters in forced_minimums:
        print(f"\n" + "="*50)
        print(f"TESTING FORCED MINIMUM: {min_iters} ITERATIONS")
        print("="*50)
        
        for d in dimensions:
            print(f"\nDimension d = {d}:")
            
            sim = ForcedAmplificationSimulator(
                dimension=d, 
                force_minimum_iterations=min_iters, 
                verbose=False
            )
            
            dimension_results = []
            
            for purity in test_purities:
                result = sim.calculate_iterations_comparison(purity)
                if 'error' not in result:
                    dimension_results.append(result)
                    
                    print(f"  λ={purity:.1f}: P_init={result['success_prob']:.4f} → "
                          f"P_forced={result['forced_final_prob']:.6f} "
                          f"(factor: {result['amplitude_amplification_factor']:.3f})")
            
            all_results[f'min_{min_iters}_d_{d}'] = dimension_results
    
    return all_results

def run_forced_amplification_experiments() -> Dict:
    """
    Run comprehensive experiments with forced amplitude amplification.
    """
    
    print("="*80)
    print("FORCED AMPLITUDE AMPLIFICATION EXPERIMENTS")
    print("Making the protocol actually use amplitude amplification!")
    print("="*80)
    
    all_experiments = {}
    
    # Experiment 1: Basic forced amplification test
    print("\n🔬 EXPERIMENT 1: Basic Forced Amplification")
    sim = ForcedAmplificationSimulator(dimension=2, force_minimum_iterations=1, verbose=True)
    all_experiments['basic_forced'] = sim.test_forced_amplification_range()
    
    # Experiment 2: Q operator evolution demonstration  
    print("\n🔬 EXPERIMENT 2: Q Operator Step-by-Step Evolution")
    test_cases = [
        {'purity': 0.7, 'description': 'High purity case'},
        {'purity': 0.3, 'description': 'Medium purity case'}, 
        {'purity': 0.1, 'description': 'Low purity case'}
    ]
    
    evolution_results = []
    for case in test_cases:
        print(f"\n{case['description']}:")
        result = sim.demonstrate_q_operator_evolution(case['purity'], max_iterations=10)
        evolution_results.append(result)
    
    all_experiments['q_operator_evolution'] = evolution_results
    
    # Experiment 3: Different forced minimums
    print("\n🔬 EXPERIMENT 3: Different Forced Minimum Iterations")
    all_experiments['different_minimums'] = test_different_forced_minimums()
    
    # Summary
    print("\n" + "="*80)
    print("FORCED AMPLIFICATION EXPERIMENTS SUMMARY")
    print("="*80)
    
    basic_results = all_experiments['basic_forced']
    helps_rate = basic_results['summary']['forced_helps_rate']
    
    print(f"📊 BASIC FORCED AMPLIFICATION RESULTS:")
    print(f"   Cases where forcing 1 iteration helps: {helps_rate:.1%}")
    print(f"   Total cases analyzed: {basic_results['summary']['total_cases']}")
    
    if basic_results['cases_where_forced_helps']:
        print(f"   Forced amplification can improve success probability!")
        best_case = max(basic_results['cases_where_forced_helps'], 
                       key=lambda x: x['improvement_forced'])
        print(f"   Best improvement: {best_case['improvement_forced']:.6f}")
        print(f"   Best amplification factor: {best_case['amplitude_amplification_factor']:.3f}")
    
    print(f"\n🎯 KEY INSIGHT:")
    if helps_rate > 0:
        print(f"   Forcing amplitude amplification DOES help in {helps_rate:.1%} of cases!")
        print(f"   This suggests the theoretical 'optimal' formula may be too conservative.")
    else:
        print(f"   Forcing amplitude amplification never helps.")
        print(f"   The theoretical optimal formula appears correct.")
    
    return all_experiments

if __name__ == "__main__":
    results = run_forced_amplification_experiments()
    
    print("\n" + "🚀 " + "="*70)
    print("FORCED AMPLIFICATION: MAKING THE Q OPERATOR WORK!")
    print("="*80)
    
    # Show the most interesting results
    if 'q_operator_evolution' in results:
        print("\n📈 Q OPERATOR EVOLUTION HIGHLIGHTS:")
        for i, evolution in enumerate(results['q_operator_evolution']):
            best = evolution['best_iteration']
            print(f"   Case {i+1} (λ={evolution['purity']:.1f}): Best at iteration {best['iteration']}")
            print(f"     P_success: {evolution['success_prob']:.6f} → {best['probability']:.6f}")
            print(f"     Improvement: {best['improvement']:.6f}")