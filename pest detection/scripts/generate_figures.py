import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Output directory (relative to scripts/ folder)
OUTPUT_DIR = "../docs/paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_fig1():
    """Fig. 1: Distribution of images across pest categories in the dataset."""
    classes = ['Adristyrannus', 'A. spiniferus', 'Ampelophaga', 'Aphis citricola',
               'Apolygus lucorum', 'Alfalfa plant bug', 'Alfalfa seed chalcid',
               'Alfalfa weevil', 'Aphids']
    counts = [186, 414, 458, 210, 228, 393, 111, 314, 2456]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, counts, color='#4f46e5', edgecolor='black')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylabel('Number of Images', fontsize=12, weight='bold')
    plt.title('Distribution of Images Across Pest Categories', fontsize=14, weight='bold')

    # Add count labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 30, int(yval),
                 ha='center', va='bottom', fontsize=10, weight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig1_Class_Distribution.png', dpi=300)
    plt.close()
    print("Created Fig 1")


def create_fig2():
    """Fig. 2: End-to-end system pipeline from image input to pesticide recommendation."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Define boxes
    boxes = [
        (0.5, 1.5, 1.8, 1, "Image\nAcquisition\n(Webcam/Upload)"),
        (2.8, 1.5, 1.8, 1, "Image\nPreprocessing\n(Resize, Rescale)"),
        (5.1, 1.5, 1.8, 1, "Deep Learning\nModel\n(Custom CNN)"),
        (7.4, 1.5, 1.8, 1, "Classification\n(Pest Class\nPrediction)"),
        (9.7, 1.5, 1.8, 1, "Pesticide\nRecommendation\nModule")
    ]

    for x, y, w, h, text in boxes:
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                     edgecolor="black", facecolor="#e0f2fe", linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, weight='bold')

    # Define arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2] + 0.1
        y1 = boxes[i][1] + boxes[i][3]/2
        x2 = boxes[i+1][0] - 0.1
        y2 = boxes[i+1][1] + boxes[i+1][3]/2
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='-|>', lw=2))

    plt.title('End-to-end System Pipeline', fontsize=14, weight='bold', y=0.9)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig2_System_Pipeline.png', dpi=300)
    plt.close()
    print("Created Fig 2")


def create_fig3():
    """Fig. 3: Proposed Custom CNN architecture showing convolutional blocks and classification head."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    layers = [
        (0.5, 2, 1.2, 2, "Input Image\n224x224x3", "#f1f5f9"),
        (2.2, 2.5, 1.2, 1, "Conv2D (32)\n+ ReLU\n3x3", "#bae6fd"),
        (3.6, 2.5, 0.8, 1, "MaxPool\n2x2", "#fecaca"),
        (4.6, 2.5, 1.2, 1, "Conv2D (64)\n+ ReLU\n3x3", "#bae6fd"),
        (6.0, 2.5, 0.8, 1, "MaxPool\n2x2", "#fecaca"),
        (7.0, 2.5, 1.2, 1, "Conv2D (128)\n+ ReLU\n3x3", "#bae6fd"),
        (8.4, 2.5, 0.8, 1, "MaxPool\n2x2", "#fecaca"),
        (9.4, 2.5, 1.0, 1, "Flatten", "#d9f99d"),
        (10.6, 2.5, 1.2, 1, "Dense (128)\n+ Dropout 0.5", "#fde047"),
        (12.0, 2.5, 1.2, 1, "Dense (9)\nSoftmax\nOutput", "#bbf7d0")
    ]

    for x, y, w, h, text, color in layers:
        box = patches.Rectangle((x, y), w, h, edgecolor="black", facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, weight='bold')

    for i in range(len(layers) - 1):
        x1 = layers[i][0] + layers[i][2]
        y1 = layers[i][1] + layers[i][3]/2
        x2 = layers[i+1][0]
        y2 = layers[i+1][1] + layers[i+1][3]/2
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='-|>', lw=1.5))

    plt.title('Custom CNN Architecture', fontsize=14, weight='bold', y=0.9)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig3_CNN_Architecture.png', dpi=300)
    plt.close()
    print("Created Fig 3")


def create_fig4():
    """Fig. 4: Flowchart of the pesticide recommendation module logic."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    boxes = [
        (3.5, 8.5, 3, 1, "Start:\nReceive Predicted Class", "#d9f99d"),
        (3.5, 7.0, 3, 1, "Check Confidence\nScore >= 85%", "#fef08a"),
        (3.5, 5.5, 3, 1, "Query Pesticide DB\nfor Predicted Class", "#bae6fd"),
        (3.5, 4.0, 3, 1, "Retrieve Pesticide\nName & Dosage", "#bae6fd"),
        (3.5, 2.5, 3, 1, "Format Output\nfor Display", "#fbcfe8"),
        (3.5, 1.0, 3, 1, "End:\nReturn Recommendation", "#d9f99d"),
        (7.5, 7.0, 2, 1, "Return\n'Uncertain'", "#fecaca")
    ]

    for x, y, w, h, text, color in boxes:
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                     edgecolor="black", facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, weight='bold')

    # Main path arrows
    for i in range(5):
        x1 = boxes[i][0] + boxes[i][2]/2
        y1 = boxes[i][1] - 0.1
        x2 = boxes[i+1][0] + boxes[i+1][2]/2
        y2 = boxes[i+1][1] + boxes[i+1][3] + 0.1
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='-|>', lw=1.5))
        if i == 1:
            ax.text(x1 + 0.2, y1 - 0.3, "Yes", weight='bold')

    # Branch arrow
    ax.annotate('', xy=(7.5, 7.5), xytext=(6.6, 7.5),
                arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='-|>', lw=1.5))
    ax.text(6.8, 7.6, "No", weight='bold')

    plt.title('Pesticide Recommendation Flowchart', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig4_Recommendation_Flowchart.png', dpi=300)
    plt.close()
    print("Created Fig 4")


def create_fig5():
    """Fig. 5: Training and validation accuracy (top) and loss (bottom) curves."""
    epochs = list(range(1, 16))
    train_acc = [0.5038, 0.5132, 0.5192, 0.5328, 0.5393, 0.5512, 0.5561, 0.5579, 0.5697, 0.5770, 0.5759, 0.5895, 0.5904, 0.5943, 0.5959]
    val_acc   = [0.5172, 0.5172, 0.5325, 0.5699, 0.5561, 0.5760, 0.5882, 0.5859, 0.5890, 0.6050, 0.6119, 0.6142, 0.6173, 0.6157, 0.6387]
    train_loss = [1.8100, 1.6251, 1.5588, 1.5292, 1.4853, 1.4601, 1.4098, 1.3832, 1.3734, 1.3497, 1.3172, 1.2910, 1.2775, 1.2349, 1.2366]
    val_loss   = [1.5835, 1.5316, 1.4587, 1.4007, 1.3972, 1.3365, 1.2788, 1.2993, 1.2386, 1.2251, 1.1916, 1.1682, 1.1388, 1.1193, 1.0983]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Accuracy Plot
    ax1.plot(epochs, train_acc, 'b-o', linewidth=2, label='Train Accuracy')
    ax1.plot(epochs, val_acc,   'r-s', linewidth=2, label='Validation Accuracy')
    ax1.set_ylabel('Accuracy', fontsize=12, weight='bold')
    ax1.set_title('Training and Validation Accuracy', fontsize=14, weight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(epochs)

    # Loss Plot
    ax2.plot(epochs, train_loss, 'b-o', linewidth=2, label='Train Loss')
    ax2.plot(epochs, val_loss,   'r-s', linewidth=2, label='Validation Loss')
    ax2.set_xlabel('Epochs', fontsize=12, weight='bold')
    ax2.set_ylabel('Loss', fontsize=12, weight='bold')
    ax2.set_title('Training and Validation Loss', fontsize=14, weight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xticks(epochs)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig5_Accuracy_Loss_Curves.png', dpi=300)
    plt.close()
    print("Created Fig 5")


def create_fig6():
    """Fig. 6: Pie Chart of Pest Category Proportions."""
    classes = ['Adristyrannus', 'A. spiniferus', 'Ampelophaga', 'Aphis citricola',
               'Apolygus lucorum', 'Alfalfa plant bug', 'Alfalfa seed chalcid',
               'Alfalfa weevil', 'Aphids']
    counts = [186, 414, 458, 210, 228, 393, 111, 314, 2456]

    # Explode the largest slice
    explode = [0] * len(classes)
    explode[-1] = 0.1  # Explode Aphids

    plt.figure(figsize=(10, 8))
    plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=140,
            explode=explode, shadow=True, textprops={'fontsize': 10, 'weight': 'bold'})

    plt.title('Proportion of Images by Pest Category', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig6_Class_Proportions_Pie.png', dpi=300)
    plt.close()
    print("Created Fig 6")


def create_fig7():
    """Fig. 7: Sample pipeline output: detected pest class with confidence score and corresponding pesticide recommendation."""
    fig, ax = plt.subplots(figsize=(8, 6))
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

    plt.text(15, 35, text1, color='#00ff00', fontsize=12, family='monospace', weight='bold')
    plt.text(15, 70, text2, color='white',   fontsize=11, family='monospace', weight='bold')

    # Bounding box / reticle around the "pest" area
    reticle = patches.Rectangle((170, 100), 300, 280, fill=False, edgecolor='#00ff00', linewidth=2, linestyle='--')
    ax.add_patch(reticle)
    plt.text(170, 90, "aphids", color='#00ff00', fontsize=10, family='monospace', weight='bold')

    plt.title('Sample Pipeline Output', fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig7_Sample_Output.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created Fig 7")


if __name__ == "__main__":
    create_fig1()
    create_fig2()
    create_fig3()
    create_fig4()
    create_fig5()
    create_fig6()
    create_fig7()
    print(f"\nAll figures successfully generated in '{OUTPUT_DIR}' directory.")
