<h1 align="center">Morphi</h1>

<p align="center">
  A computational framework for organoid instance segmentation,<br>
  PI-positive region segmentation, and organoid-level death-ratio quantification.
</p>

<p align="center">
  <strong>Official implementation of “AI-driven phenotypic profiling of organoids in droplet culture”</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.2.2-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Paper-Under%20Review-lightgrey" alt="Paper Status">
</p>

<hr>

<h2>Overview</h2>

<p>
  Morphi is a biomedical image-analysis framework developed for automated
  organoid segmentation and quantitative analysis of propidium iodide
  (PI)-positive regions.
</p>

<p>The pipeline supports:</p>

<ul>
  <li>organoid instance segmentation;</li>
  <li>PI-positive region segmentation;</li>
  <li>organoid-level death-ratio calculation;</li>
  <li>batch processing of microscopy images;</li>
  <li>visualization and export of quantitative results;</li>
  <li>ONNX-based model deployment;</li>
  <li>a graphical user interface for non-programmatic use.</li>
</ul>

<p>
  For each detected organoid, the death ratio is calculated as:
</p>

<p align="center">
  <code>
    R<sub>death</sub> =
    |M<sub>organoid</sub> ∩ M<sub>Dead</sub>| /
    |M<sub>organoid</sub>|
  </code>
</p>

<p>
  where <code>M<sub>organoid</sub></code> denotes the predicted organoid mask,
  and <code>M<sub>Dead</sub></code> denotes the predicted Dead-region mask.
</p>

<h2>Workflow</h2>

<p align="center">
  <img
    src="assets/workflow.png"
    alt="Morphi workflow"
    width="900"
  >
</p>

<p>
  The analysis pipeline first identifies individual organoid instances and
  segments PI-positive regions. The two prediction outputs are subsequently
  integrated to calculate organoid-level morphological and viability-related
  measurements.
</p>

<h2>Repository Structure</h2>

<pre><code>morphi/
├── assets/                  # Images used in this README
├── checkpoints/             # Pretrained model checkpoints
├── configs/                 # Training and inference configurations
├── datasets/                # Dataset loading and preprocessing
├── examples/                # Example input images
├── models/                  # Model architectures
├── outputs/                 # Default output directory
├── scripts/                 # Training, evaluation, and export scripts
├── app.py                   # Graphical user interface
├── evaluate.py              # Evaluation entry point
├── inference.py             # Inference entry point
├── train.py                 # Training entry point
├── requirements.txt         # Python dependencies
├── setup.py                 # Package configuration
├── LICENSE                  # Software license
└── README.md                # Project documentation
</code></pre>

<h2>Installation</h2>

<h3>1. Clone the repository</h3>

<pre><code class="language-bash">git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY
</code></pre>

<h3>2. Create a Conda environment</h3>

<pre><code class="language-bash">conda create -n morphi python=3.10
conda activate morphi
</code></pre>

<h3>3. Install dependencies</h3>

<pre><code class="language-bash">pip install -r requirements.txt
</code></pre>

<p>
  To install Morphi as an editable Python package:
</p>

<pre><code class="language-bash">pip install -e .
</code></pre>

<h2>System Requirements</h2>

<p>The code has been tested with the following environment:</p>

<table>
  <thead>
    <tr>
      <th>Component</th>
      <th>Version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Operating system</td>
      <td>Ubuntu XX.XX / macOS XX</td>
    </tr>
    <tr>
      <td>Python</td>
      <td>3.10</td>
    </tr>
    <tr>
      <td>PyTorch</td>
      <td>X.X.X</td>
    </tr>
    <tr>
      <td>CUDA</td>
      <td>XX.X</td>
    </tr>
    <tr>
      <td>cuDNN</td>
      <td>X.X</td>
    </tr>
    <tr>
      <td>ONNX Runtime</td>
      <td>X.X.X</td>
    </tr>
  </tbody>
</table>

<p>
  GPU acceleration is recommended. CPU inference may be supported but can be
  substantially slower, particularly for large microscopy images.
</p>

<h2>Data Preparation</h2>

<p>
  Organize the dataset using the following directory structure:
</p>

<pre><code>data/
├── train/
│   ├── images/
│   └── annotations.json
├── val/
│   ├── images/
│   └── annotations.json
└── test/
    ├── images/
    └── annotations.json
</code></pre>

<p>
  Instance annotations should follow the COCO instance-segmentation format.
  Each image record should contain the image dimensions, file name, image ID,
  category information, bounding boxes, and segmentation annotations.
</p>

<h3>Dataset Availability</h3>

<p>
  The dataset used in this study is
  <strong>[publicly available / available under controlled access / available upon reasonable request]</strong>.
</p>

<p>
  Dataset repository:
  <a href="DATASET_URL">DATASET NAME</a>
</p>

<p>
  Dataset DOI:
  <a href="https://doi.org/DATASET_DOI">DATASET_DOI</a>
</p>

<h2>Pretrained Models</h2>

<p>
  Download the pretrained model checkpoints and place them in the
  <code>checkpoints/</code> directory.
</p>

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Task</th>
      <th>Format</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Organoid instance model</td>
      <td>Organoid instance segmentation</td>
      <td>PyTorch</td>
      <td><a href="MODEL_URL">Download</a></td>
    </tr>
    <tr>
      <td>PI segmentation model</td>
      <td>PI-positive region segmentation</td>
      <td>PyTorch</td>
      <td><a href="MODEL_URL">Download</a></td>
    </tr>
    <tr>
      <td>Organoid instance model</td>
      <td>ONNX deployment</td>
      <td>ONNX</td>
      <td><a href="MODEL_URL">Download</a></td>
    </tr>
    <tr>
      <td>PI segmentation model</td>
      <td>ONNX deployment</td>
      <td>ONNX</td>
      <td><a href="MODEL_URL">Download</a></td>
    </tr>
  </tbody>
</table>

<p>Expected checkpoint structure:</p>

<pre><code>checkpoints/
├── organoid_instance_model.pth
├── pi_segmentation_model.pth
├── organoid_instance_model.onnx
└── pi_segmentation_model.onnx
</code></pre>

<h2>Quick Start</h2>

<h3>Run inference on a single image</h3>

<pre><code class="language-bash">python inference.py \
    --image examples/example.tif \
    --instance-checkpoint checkpoints/organoid_instance_model.pth \
    --pi-checkpoint checkpoints/pi_segmentation_model.pth \
    --output-dir outputs/example
</code></pre>

<h3>Run inference on an image folder</h3>

<pre><code class="language-bash">python inference.py \
    --input-dir path/to/images \
    --instance-checkpoint checkpoints/organoid_instance_model.pth \
    --pi-checkpoint checkpoints/pi_segmentation_model.pth \
    --output-dir outputs/batch_results
</code></pre>

<h3>Example output structure</h3>

<pre><code>outputs/
├── instance_masks/          # Organoid instance masks
├── pi_masks/                # PI-positive segmentation masks
├── visualizations/          # Segmentation overlays
├── measurements.csv         # Organoid-level measurements
└── run_information.json     # Inference parameters and metadata
</code></pre>

<h2>Output Measurements</h2>

<p>
  The output CSV file may contain the following organoid-level measurements:
</p>

<table>
  <thead>
    <tr>
      <th>Column</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>image_name</code></td>
      <td>Name of the source microscopy image</td>
    </tr>
    <tr>
      <td><code>organoid_id</code></td>
      <td>Identifier of the detected organoid instance</td>
    </tr>
    <tr>
      <td><code>organoid_area</code></td>
      <td>Area of the organoid mask in pixels</td>
    </tr>
    <tr>
      <td><code>pi_positive_area</code></td>
      <td>PI-positive area within the organoid mask</td>
    </tr>
    <tr>
      <td><code>death_ratio</code></td>
      <td>PI-positive area divided by total organoid area</td>
    </tr>
    <tr>
      <td><code>centroid_x</code></td>
      <td>Horizontal coordinate of the organoid centroid</td>
    </tr>
    <tr>
      <td><code>centroid_y</code></td>
      <td>Vertical coordinate of the organoid centroid</td>
    </tr>
  </tbody>
</table>

<h3>Example measurements</h3>

<table>
  <thead>
    <tr>
      <th>Image</th>
      <th>Organoid ID</th>
      <th>Organoid Area</th>
      <th>PI-positive Area</th>
      <th>Death Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>example_001.tif</td>
      <td>1</td>
      <td>12,450</td>
      <td>2,153</td>
      <td>0.173</td>
    </tr>
    <tr>
      <td>example_001.tif</td>
      <td>2</td>
      <td>9,284</td>
      <td>3,862</td>
      <td>0.416</td>
    </tr>
  </tbody>
</table>

<h2>Example Results</h2>

<table>
  <thead>
    <tr>
      <th align="center">Input image</th>
      <th align="center">Organoid instances</th>
      <th align="center">PI-positive regions</th>
      <th align="center">Combined output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">
        <img
          src="assets/example_input.png"
          alt="Example input"
          width="220"
        >
      </td>
      <td align="center">
        <img
          src="assets/example_instances.png"
          alt="Organoid instance segmentation"
          width="220"
        >
      </td>
      <td align="center">
        <img
          src="assets/example_pi.png"
          alt="PI-positive segmentation"
          width="220"
        >
      </td>
      <td align="center">
        <img
          src="assets/example_combined.png"
          alt="Combined segmentation output"
          width="220"
        >
      </td>
    </tr>
  </tbody>
</table>

<h2>Training</h2>

<h3>Train the organoid instance-segmentation model</h3>

<pre><code class="language-bash">python train.py \
    --config configs/instance_segmentation.yaml
</code></pre>

<h3>Train the PI-positive region-segmentation model</h3>

<pre><code class="language-bash">python train.py \
    --config configs/pi_segmentation.yaml
</code></pre>

<p>
  Dataset paths, model parameters, optimizer settings, learning-rate schedules,
  batch sizes, training epochs, and random seeds should be defined in the
  corresponding configuration files.
</p>

<h2>Evaluation</h2>

<p>
  Evaluate a trained model on the held-out test set:
</p>

<pre><code class="language-bash">python evaluate.py \
    --config configs/evaluation.yaml \
    --instance-checkpoint checkpoints/organoid_instance_model.pth \
    --pi-checkpoint checkpoints/pi_segmentation_model.pth \
    --output-dir outputs/evaluation
</code></pre>

<p>The evaluation pipeline reports metrics including:</p>

<ul>
  <li>Dice coefficient;</li>
  <li>intersection over union;</li>
  <li>precision;</li>
  <li>recall;</li>
  <li>average precision;</li>
  <li>instance-level detection metrics;</li>
  <li>Pearson correlation coefficient for organoid-level death ratios.</li>
</ul>

<p>
  The held-out test set was not used for model selection, threshold selection,
  or hyperparameter tuning.
</p>

<h2>Reproducing the Reported Results</h2>

<p>
  To reproduce the main quantitative results reported in the manuscript:
</p>

<pre><code class="language-bash">bash scripts/reproduce_results.sh
</code></pre>

<p>The reproduction script performs the following operations:</p>

<ol>
  <li>loads the model checkpoints used in the manuscript;</li>
  <li>runs inference on the held-out test set;</li>
  <li>calculates segmentation and organoid-level quantitative metrics;</li>
  <li>exports the numerical results;</li>
  <li>generates the corresponding tables and figures.</li>
</ol>

<h3>Random Seeds</h3>

<table>
  <thead>
    <tr>
      <th>Library</th>
      <th>Seed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Python</td>
      <td>42</td>
    </tr>
    <tr>
      <td>NumPy</td>
      <td>42</td>
    </tr>
    <tr>
      <td>PyTorch</td>
      <td>42</td>
    </tr>
  </tbody>
</table>

<p>
  Exact numerical reproduction can depend on the operating system, CUDA
  version, GPU architecture, library versions, and nondeterministic GPU
  operations.
</p>

<h2>ONNX Deployment</h2>

<h3>Export a PyTorch checkpoint to ONNX</h3>

<pre><code class="language-bash">python scripts/export_onnx.py \
    --checkpoint checkpoints/model.pth \
    --output checkpoints/model.onnx
</code></pre>

<h3>Run ONNX Runtime inference</h3>

<pre><code class="language-bash">python scripts/onnx_inference.py \
    --model checkpoints/model.onnx \
    --image examples/example.tif \
    --output-dir outputs/onnx_example
</code></pre>

<p>
  The trained models are exported in the Open Neural Network Exchange
  (<strong>ONNX</strong>) format and executed using
  <strong>ONNX Runtime</strong>.
</p>

<h2>Graphical User Interface</h2>

<p>Launch the graphical user interface using:</p>

<pre><code class="language-bash">python app.py
</code></pre>

<p>
  The graphical interface supports image import, model inference, result
  visualization, manual review, and export of organoid-level measurements.
</p>

<h3>Packaged macOS application</h3>

<p>
  macOS users can launch the packaged application by opening:
</p>

<pre><code>dist/Morphi.app
</code></pre>

<h2>Code Availability</h2>

<p>
  The source code used to generate the results reported in the manuscript is
  available in this repository.
</p>

<p>
  The exact version associated with the submitted or published manuscript is
  permanently archived on Zenodo:
</p>

<p>
  <a href="https://doi.org/ZENODO_DOI">
    https://doi.org/ZENODO_DOI
  </a>
</p>

<p>
  Updated development versions are maintained on GitHub. The Zenodo archive
  should be used when reproducing or citing the exact manuscript version.
</p>

<h2>Citation</h2>

<p>
  Citation information will be updated upon publication.
</p>

<p>
  When the article has been published, replace the placeholder below with the
  final citation:
</p>

<pre><code class="language-bibtex">@article{AUTHOR_YEAR_MORPHI,
  title   = {PAPER TITLE},
  author  = {AUTHOR LIST},
  journal = {JOURNAL},
  year    = {YEAR},
  volume  = {VOLUME},
  pages   = {PAGE RANGE OR ARTICLE NUMBER},
  doi     = {DOI}
}
</code></pre>

<h2>License</h2>

<p>
  This project is released under the
  <a href="LICENSE">MIT License</a>.
</p>

<p>
  Third-party source code, pretrained models, libraries, and datasets remain
  subject to their respective licenses and terms of use.
</p>

<h2>Acknowledgements</h2>

<p>
  This project uses or builds upon several open-source software packages,
  including:
</p>

<ul>
  <li><a href="https://pytorch.org/">PyTorch</a>;</li>
  <li><a href="https://pytorch.org/vision/">torchvision</a>;</li>
  <li><a href="https://monai.io/">MONAI</a>;</li>
  <li><a href="https://onnx.ai/">ONNX</a>;</li>
  <li><a href="https://onnxruntime.ai/">ONNX Runtime</a>;</li>
  <li><a href="https://opencv.org/">OpenCV</a>.</li>
</ul>

<p>
  Please consult the corresponding projects for their documentation,
  citations, and licensing requirements.
</p>

<h2>Contact</h2>

<p>
  For technical questions, bug reports, or feature requests, please open a
  GitHub issue.
</p>

<p>
  For questions concerning the manuscript or dataset, contact:
</p>

<ul>
  <li>
    <strong>Author Name</strong>:
    <a href="mailto:EMAIL@example.com">EMAIL@example.com</a>
  </li>
</ul>

<hr>

<p align="center">
  <strong>Morphi</strong><br>
  Automated organoid segmentation and quantitative image analysis
</p>
