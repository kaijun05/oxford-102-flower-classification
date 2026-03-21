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
      <strong>RO1 (Performance):</strong> ResNet50 (Augmented) outperformed all models, proving that skip-connections effectively 
      preserve fine-grained features. The test accuracies of the Custom CNN models are much lower as compared to the ResNet-50 and MobileNetV2. 
      Also, the test losses of all Custom CNN variants scored the highest. This broad gap evidently points out that the Custom CNN cannot be 
      compared in terms of performance for this fine-grained classification problem.
    </li>
    <br>
    <li>
      <strong>RO2 (Augmentation):</strong> Data Augmentation (Rotation, Zoom, Flip) improved ResNet50's accuracy in general. ResNet-50 has a 
      strong advantage of basing its architecture on the pre-trained model and fine-tuning, which has enabled it to show a strong similarity in the 
      trend of both training and validation metrics, suggesting successful avoidance of overfitting. 
      This generalization is also augmented with data augmentation. On the contrary, Custom CNN has a terrible overfitting condition because it 
      has a low validation score. Although MobileNetV2 tends to be robust because of transfer learning, data augmentation behavior can be more 
      complicated during the prevention of overfitting, but early stopping can be used to address it.
    </li>
    <br>
    <li>
      <strong>RO3 (Generalizability):</strong> MobileNetV2 offers a higher tradeoff between computational cost (in terms of training time) and 
      accuracy than ResNet-50, particularly where the cost factor is of main concern. ResNet-50 can train very slightly higher peak accuracy, 
      although MobileNetV2 is competitive in terms of accuracy, but with much lower training time in some settings.
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
