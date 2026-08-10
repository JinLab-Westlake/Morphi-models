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


<h2>Overview</h2>

<ul>
  <li><strong>DoANet</strong>: semantic segmentation of live/dead regions.</li>
  <li><strong>Instance segmentation</strong>: organoid detection and segmentation using Detectron2.</li>
  <li><strong>Death-rate analysis</strong>: combines semantic and instance masks to calculate the death rate of each organoid.</li>
</ul>


<h2>Installation</h2>

<pre><code>git clone https://github.com/JinLab-Westlake/Morphi-models.git
cd Morphi

chmod +x install.sh
bash install.sh

conda activate morphi
</code></pre>



<h2>Model Checkpoints</h2>

<p>Download the pretrained checkpoints and place them under:</p>

<pre><code>checkpoints/
├── doanet/
│   └── &lt;DOANET_CHECKPOINT&gt;.pth
└── instance/
    └── &lt;INSTANCE_CHECKPOINT&gt;.pth
</code></pre>

<p>Download:</p>

<pre><code>&lt;MODEL_CHECKPOINT_DOWNLOAD_URL&gt;</code></pre>

<h2>Input</h2>

<p>Morphi accepts bright-field organoid images.</p>

<pre><code>example/
└── input/
    └── example.png
</code></pre>

<p>
DoANet processes full-resolution images using 224 × 224 patches.
Overlapping predictions are averaged before reconstruction of the final
semantic mask.
</p>

<h2>Run the Pipeline</h2>

<pre><code>bash scripts/run_full_pipeline.sh</code></pre>

<h2>Output</h2>

<pre><code>pipeline_results/
└── example/
    ├── example_semantic.png
    ├── instance_mask/
    │   └── example_instance_label.tif
    ├── overlay/
    │   └── example.png
    └── instance_death_rate.xlsx
</code></pre>


<p>
Each organoid receives a unique integer ID in the instance mask.
The death-rate table contains:
</p>

<pre><code>image_name
instance_id
instance_area
live_pixels
dead_pixels
overlap_pixels
death_rate
</code></pre>

<h2>Run DoANet Only</h2>

<pre><code>python DoANet/inference_full_image.py \
    --image_path &lt;INPUT_IMAGE&gt; \
    --checkpoint &lt;DOANET_CHECKPOINT&gt; \
    --output_dir &lt;MODEL_OUTPUT_DIRECTORY&gt; \
    --test_save_dir &lt;OUTPUT_DIRECTORY&gt; \
    --cfg &lt;DOANET_CONFIG&gt; \
    --img_size 224 \
    --stride 224 \
    --num_classes 3
</code></pre>

<h2>Reproducibility</h2>

<p>
The experimental environment is provided in
<code>env/environment.yaml</code>.
The Detectron2 source used for the experiments is bundled in
<code>env/detectron2/</code>.
</p>


<h2>Citation</h2>

<pre><code>&lt;INSERT CITATION AFTER PUBLICATION&gt;</code></pre>

<h2>License</h2>

<p>&lt;INSERT LICENSE&gt;</p>
