import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Output goes to docs/paper_figures/
os.makedirs("../docs/paper_figures", exist_ok=True)

def create_fig7():
    """Fig. 7: Sample pipeline output: detected pest class with confidence score and corresponding pesticide recommendation."""
    fig, ax = plt.subplots(figsize=(8, 6))
    # OpenCV uses top-left as (0,0)
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.axis('off')
    
    # Simulated Camera Feed Background (Dark grayish-green)
    bg = patches.Rectangle((0, 0), 640, 480, facecolor='#2f4f4f')
    ax.add_patch(bg)
    
    # Simulated Leaf Shape (Light green)
    leaf = patches.Ellipse((320, 240), 300, 400, angle=30, facecolor='#4ade80', alpha=0.6)
    ax.add_patch(leaf)
    
    # UI Overlay Background for text readability
    overlay = patches.Rectangle((5, 5), 630, 80, facecolor='black', alpha=0.5, edgecolor='none')
    ax.add_patch(overlay)
    
    # Simulated OpenCV text overlays
    text1 = "aphids | Imidacloprid 17.8% SL | 0.5 ml per litre of water"
    text2 = "Confidence: 94.2%"
    
    # Plotting text (OpenCV putText style)
    plt.text(15, 35, text1, color='#00ff00', fontsize=12, family='monospace', weight='bold')
    plt.text(15, 70, text2, color='white', fontsize=11, family='monospace', weight='bold')
    
    # Add a bounding box or target reticle around the "pest" area
    reticle = patches.Rectangle((170, 100), 300, 280, fill=False, edgecolor='#00ff00', linewidth=2, linestyle='--')
    ax.add_patch(reticle)
    plt.text(170, 90, "aphids", color='#00ff00', fontsize=10, family='monospace', weight='bold')
    
    plt.title('Sample Pipeline Output', fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../docs/paper_figures/Fig7_Sample_Output.png', dpi=300, bbox_inches='tight')
    print("Created Fig 7")

if __name__ == "__main__":
    create_fig7()
