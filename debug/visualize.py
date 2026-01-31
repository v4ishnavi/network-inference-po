import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import loadmat
import argparse
import os
from pathlib import Path

def load_graph_from_mat(mat_path):
    """
    Load adjacency matrix from .mat file
    Returns A, D, L matrices
    """
    mat_data = loadmat(mat_path)
    
    # Try to find the adjacency matrix in the .mat file
    # Common variable names: A, A_full_orig, adjacency, adj_matrix
    possible_keys = ['A', 'A_full_orig', 'adjacency', 'adj_matrix']
    
    A = None
    for key in possible_keys:
        if key in mat_data:
            A = mat_data[key]
            print(f"Found adjacency matrix under key: '{key}'")
            break
    
    # If not found, look for any matrix that's not metadata
    if A is None:
        for key, value in mat_data.items():
            if not key.startswith('__') and isinstance(value, np.ndarray) and value.ndim == 2:
                A = value
                print(f"Found adjacency matrix under key: '{key}'")
                break
    
    if A is None:
        raise ValueError(f"Could not find adjacency matrix in {mat_path}")
    
    # Calculate degree and Laplacian matrices
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    
    return A, D, L

def visualize_graph_from_mat(mat_path, output_path=None, node_limit=100, show_weights=False):
    """
    Visualize graph from .mat file
    
    Args:
        mat_path: Path to .mat file containing adjacency matrix
        output_path: Path to save visualization (optional)
        node_limit: Maximum nodes to visualize in network graph (for performance)
        show_weights: Whether to show edge weights on network graph
    """
    # Load matrices
    A, D, L = load_graph_from_mat(mat_path)
    N = A.shape[0]
    
    print(f"\nGraph loaded successfully!")
    print(f"Number of nodes: {N}")
    print(f"Adjacency matrix shape: {A.shape}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Network Graph Visualization (only if N is reasonable)
    if N <= node_limit:
        ax1 = plt.subplot(2, 3, 1)
        G = nx.from_numpy_array(A)
        
        # Use spring layout with appropriate parameters
        pos = nx.spring_layout(G, k=2/np.sqrt(N), iterations=50, seed=42)
        
        # Draw edges with varying thickness based on weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        if len(weights) > 0:
            # Normalize weights for visualization
            max_weight = max(weights) if max(weights) > 0 else 1
            normalized_weights = [w/max_weight for w in weights]
            
            nx.draw_networkx_edges(G, pos, width=[w*3 for w in normalized_weights], 
                                  alpha=0.4, edge_color='gray', ax=ax1)
        
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=300, ax=ax1)
        
        if N <= 30:  # Only show labels for small graphs
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)
        
        ax1.set_title(f'Graph Visualization (N={N})', fontsize=14, fontweight='bold')
        ax1.axis('off')
    else:
        ax1 = plt.subplot(2, 3, 1)
        ax1.text(0.5, 0.5, f'Graph too large to visualize\n(N={N} nodes)\n\nSee other panels for matrix views', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax1.set_title(f'Graph Visualization (N={N})', fontsize=14, fontweight='bold')
        ax1.axis('off')
    
    # 2. Adjacency Matrix Heatmap
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(A, cmap='YlOrRd', interpolation='nearest', aspect='auto')
    ax2.set_title('Adjacency Matrix (A)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Node')
    ax2.set_ylabel('Node')
    plt.colorbar(im2, ax=ax2, label='Edge Weight')
    
    # 3. Degree Matrix Heatmap
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(D, cmap='Blues', interpolation='nearest', aspect='auto')
    ax3.set_title('Degree Matrix (D)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Node')
    ax3.set_ylabel('Node')
    plt.colorbar(im3, ax=ax3, label='Degree')
    
    # 4. Laplacian Matrix Heatmap
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(L, cmap='RdBu_r', interpolation='nearest', aspect='auto')
    ax4.set_title('Laplacian Matrix (L)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Node')
    ax4.set_ylabel('Node')
    plt.colorbar(im4, ax=ax4, label='Value')
    
    # 5. Degree Distribution
    ax5 = plt.subplot(2, 3, 5)
    degrees = np.diag(D)
    
    if N <= 50:
        ax5.bar(range(N), degrees, color='steelblue', alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Node')
    else:
        # For large graphs, show histogram instead
        ax5.hist(degrees, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Degree')
        ax5.set_ylabel('Frequency')
    
    ax5.set_title('Degree Distribution', fontsize=14, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Graph Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate statistics
    num_edges = np.sum(A > 0) / 2  # Divide by 2 for undirected graph
    density = num_edges / (N * (N - 1) / 2) if N > 1 else 0
    avg_degree = np.mean(degrees)
    
    # Check if graph is symmetric
    is_symmetric = np.allclose(A, A.T)
    
    stats_text = f"""
    Graph Statistics:
    
    Nodes: {N}
    Edges: {int(num_edges)}
    Density: {density:.4f}
    
    Degree Stats:
    Average: {avg_degree:.2f}
    Min: {np.min(degrees):.2f}
    Max: {np.max(degrees):.2f}
    Std: {np.std(degrees):.2f}
    
    Matrix Properties:
    Symmetric: {is_symmetric}
    L = D - A: {np.allclose(L, D - A)}
    
    File: {Path(mat_path).name}
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax6.set_title('Statistics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    else:
        # Auto-generate output path based on input
        output_dir = Path(mat_path).parent
        output_name = Path(mat_path).stem + '_visualization.png'
        output_path = output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    
    return fig, (A, D, L)

def main():
    parser = argparse.ArgumentParser(description='Visualize graph from .mat file')
    parser.add_argument('mat_path', type=str, 
                       help='Path to .mat file (e.g., results/run_1/A_full_orig.mat)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for visualization (optional)')
    parser.add_argument('--node-limit', type=int, default=100,
                       help='Maximum nodes to show in network graph (default: 100)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.mat_path):
        print(f"Error: File not found: {args.mat_path}")
        return
    
    # Visualize
    visualize_graph_from_mat(args.mat_path, args.output, args.node_limit)

if __name__ == "__main__":
    main()