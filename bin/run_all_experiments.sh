#!/bin/bash

# Run all SI Partial Observability experiments
# This script runs three different experiments comparing initialization methods

echo "=========================================="
echo "SI Partial Observability Experiment Suite"
echo "=========================================="

# Change to the MATLAB codes directory
cd "/home/vaishnavi-shivkumar/Documents/4-1/IS/matlab codes"

echo ""
echo "Starting experiments..."
echo ""

# Experiment 1: Compare initialization methods
echo "--- EXPERIMENT 1: Initialization Comparison ---"
echo "Running random vs least squares initialization comparison..."
echo "Expected time: ~2-3 minutes"
echo ""

matlab -batch "PO_compare_init_methods" 2>&1 | grep -E "(===|Random|Least|Winner|error|Time|Results)"

echo ""
echo "Experiment 1 completed. Results saved to init_comparison_results.mat"
echo ""

# Experiment 2: Vary hidden nodes
echo "--- EXPERIMENT 2: Varying Hidden Nodes ---"
echo "Testing N_obs=5, N_hidden=[1,2,3], K=N_hidden..."
echo "Expected time: ~5-8 minutes"
echo ""

matlab -batch "PO_vary_hidden_nodes" 2>&1 | grep -E "(===|Testing|Random|LS|Results)"

echo ""
echo "Experiment 2 completed. Results saved to vary_hidden_nodes_results.mat"
echo ""

# Experiment 3: Vary observed nodes  
echo "--- EXPERIMENT 3: Varying Observed Nodes ---"
echo "Testing N_hidden=1, N_obs=[2,3,4,5,6,7,8], K=1..."
echo "Expected time: ~8-12 minutes"
echo ""

matlab -batch "PO_vary_observed_nodes" 2>&1 | grep -E "(===|Testing|Random|LS|Results)"

echo ""
echo "Experiment 3 completed. Results saved to vary_observed_nodes_results.mat"
echo ""

echo "=========================================="
echo "All experiments completed!"
echo ""
echo "Generated files:"
echo "  - init_comparison_results.mat"
echo "  - vary_hidden_nodes_results.mat" 
echo "  - vary_observed_nodes_results.mat"
echo "  - plots/init_comparison.png"
echo "  - plots/vary_hidden_nodes.png"
echo "  - plots/vary_observed_nodes.png"
echo ""
echo "Plots saved to ./plots/ folder"
echo "=========================================="
