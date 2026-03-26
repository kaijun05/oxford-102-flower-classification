<h1>Oxford 102 Flower Classification: Deep Neural Networks 🌸</h1>

CDS6354 Machine Learning (Trimester November/December 2025 - Term 2530)

<h2>Overview</h2>
<div align='justify'>
  <p>
    This project explores <strong>Fine-Grained Visual Classification (FGVC)</strong> using the Oxford 102 Flower Dataset. By leveraging 
    Transfer Learning and Data Augmentation, we benchmarked multiple Deep Neural Network architectures to determine the 
    optimal balance between classification precision and computational efficiency.
  </p>
</div>

<h2>Dataset</h2>
<div align='justify'>
  <p>
    The Oxford 102 dataset consists of <strong>8,189</strong> images across <strong>102</strong> different flower categories. 
    Identifying these species is technically challenging due to:
    <ul>
      <li>
        <strong>High Inter-class Similarity:</strong> Different species (e.g., Sunflower vs. Marigold) share similar colors and 
        geometric patterns.
      </li>
      <li>
        <strong>High Intra-class Variation:</strong> Significant differences in lighting, scale, and petal orientation 
        within the same species.
      </li>
    </ul>
  </p>
</div>

<h2>Interactive Web App</h2>
<div align='justify'>
  <p>
    The repository includes a live <strong>Streamlit</strong> dashboard that allows users to: <strong>upload</strong> any flower image, 
    <strong>toggle</strong> between two fine-tuned architectures: ResNet50 and MobileNetV2, and <strong>visualize</strong> Top-5 predictions 
    with real-time confidence scores.
  </p>
  <br>
  <p align='center'>
    View the <strong>Web App</strong>
  </p>
  <p align='center'>
    <a href='https://oxford-102-flower-classification.streamlit.app/'>
      <img src='https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white'>
    </a>
  </p>
</div>

<h2>Model Architectures</h2>
<div align='justify'>
  <p>
    We implemented and evaluated three distinct approaches using <strong>TensorFlow/Keras</strong>:
  </p>
  <table>
    <tr>
      <th>Architecture</th>
      <th>Strategy</th>
      <th>Primary Strength</th>
    </tr>
    <tr>
      <td>ResNet50</td>
      <td>
        Transfer Learning (Baseline, Baseline + Fine-Tuned, Augmented, Augmented + Fine-Tuned). Fine-tuning performed by 
        unfreezing higher-level layers (conv5 block).
      </td>
      <td>
        <strong>Accuracy Champion:</strong> Achieved <strong>97%</strong> test accuracy for Augmented + Fine-Tuned variant through 
        deep residual learning.
      </td>
    </tr>
    <tr>
      <td>MobileNetV2</td>
      <td>
        Transfer Learning (Baseline, Baseline + Fine-Tuned, Augmented, Augmented + Fine-Tuned). Fine-tuning performed by 
        unfreezing higher-level layers.
      </td>
      <td>
        <strong>Efficiency Expert:</strong> Achieved <strong>91%</strong> accuracy for Baseline + Fine-Tuned with a 
        lightweight, mobile-optimized footprint.
      </td>
    </tr>
    <tr>
      <td>Custom CNN</td>
      <td>Built from scratch (Baseline, Baseline + Tuned, Augmented, Augmented + Tuned). </td>
      <td>
        <strong>Baseline:</strong> Served as a control group to measure the efficacy of Transfer Learning. Hyperparameters optimized using 
        Random Search. Retrained using the best parameter combination.
      </td>
    </tr>
  </table>
</div>

<h2>Experimental Results</h2>
<div align='justify'>
  <p>
    Our research objectives (RO) focused on the <strong>impact of data strategy on model robustness</strong>:
  </p>
  <ul>
    <li>
      <strong>RO1 (Performance):</strong> ResNet50 (Augmented) <strong>outperformed</strong> all models, proving that skip-connections effectively 
      preserve fine-grained features. In contrast, the Custom CNN variants consistently exhibited the <strong>lowest accuracies and highest test losses</strong> across all trials. This significant performance gap confirms that shallower architectures lack the depth required to resolve the complex patterns in the Oxford 102 Flower dataset.
    </li>
    <br>
    <li>
      <strong>RO2 (Augmentation):</strong> Data Augmentation (Rotation, Zoom, Flip) improved ResNet50's accuracy in general. ResNet-50 has a 
      strong advantage of basing its architecture on the pre-trained model and fine-tuning, which has enabled it to show a strong similarity in the trend of both training and validation metrics, suggesting <strong>successful avoidance of overfitting</strong>. Conversely, the Custom CNN suffered from <strong>extreme overfitting</strong>, characterized by high training accuracy but failing validation scores. While MobileNetV2 was robust due to transfer learning, its response to augmentation was more volatile.
    </li>
    <br>
    <li>
      <strong>RO3 (Computational Efficiency):</strong> MobileNetV2 offers a <strong>higher tradeoff between computational cost (in terms of training time) and accuracy</strong>. While <strong>ResNet50 achieved the absolute peak accuracy (97%)</strong>, MobileNetV2 remained highly competitive (91%) while requiring <strong>shorter training times</strong> and fewer parameters. This confirms that while deeper networks like ResNet50 are superior for raw precision, MobileNetV2 is the better choice for environments with limited resources where deployment speed and hardware efficiency are primary considerations.
    </li>
  </ul>
</div>

<h2>Installation and Requirements</h2>

1. In your terminal, clone the repository:
```bash
git clone https://github.com/kaijun05/oxford-102-flower-classification.git
cd oxford-102-flower-classification
```
2. Handle the large files (Git LFS)
The model weights exceed 100MB and are managed via Git LFS. Ensure LFS is installed to pull the full `.keras` files:
```bash
git lfs install
git lfs pull
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the app:
```bash
streamlit run app.py
```

<h2>Contributions</h2>

|   | Name                              |
|--:|:-----------------------------------------:|
| 1 | LOOI KAI JUN |
| 2 | CHEW EN QING |
| 3 | CHIN JIA WEN |
| 4 | PRITIVE NAIR A/L K SRITHARAN |
| 5 | SAYID ABDUR-RAHMAN AL-AIDARUS BIN SYED ABU BAKAR MASHOR AL-IDRUS |
