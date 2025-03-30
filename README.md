# Fast Texture Synthesis via Pseudo-Optimizer

## Introduction

This project proposes a **fast texture synthesis** method using a **pseudo-optimizer**. The goal is to generate high-quality textures efficiently by leveraging optimization techniques to accelerate the synthesis process. This approach is particularly useful in fields where the rapid generation of realistic textures is crucial, such as video games, virtual reality, and simulation.

## Prerequisites

Make sure you have the following dependencies installed:

- Python 3.8.10
- NumPy
- scikit-image
- Matplotlib
- Pillow

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Architecture Overview

The proposed method relies on using a **pseudo-optimizer** for texture synthesis. Unlike traditional approaches based on costly iterative optimizations, this method employs an efficient approximation that reduces computation time while maintaining high texture quality.

![Texture Synthesis Architecture](images/texture_synthesis_architecture.png)

### Key Features:

- **Efficiency**: Significant reduction in computation time compared to traditional methods.
- **High Quality**: Generation of realistic textures with high fidelity to input examples.
- **Flexibility**: Applicable to various types of textures and adaptable to different use cases.

### Why This Approach Works:

- **Efficient Approximation**: The use of a pseudo-optimizer allows approximation of optimal solutions without requiring intensive computations.
- **Contextual Information Utilization**: The method exploits contextual image information to generate coherent and realistic textures.

### Applications:

- **Video Games**: Rapid texture generation for complex environments.
- **Virtual Reality**: Creation of immersive textures for realistic simulations.
- **Graphic Design**: Production of patterns and textures for artistic and commercial projects.

## Usage

### Preparation

To prepare the data for texture synthesis, place your sample images in the `input_textures` folder. Ensure the images are in the appropriate format (e.g., PNG or JPEG) and have a uniform size for optimal results.

### Training

To train the model on your custom dataset, use the following script:

```bash
python train.py --data_dir input_textures --epochs 50 --batch_size 8 --device ("cpu" or "cuda")
```

#### Loss Function

The model is trained by minimizing a loss function that evaluates the difference between the generated texture and the input sample. The Adam optimizer is used to adjust the model weights with a learning rate tuned for stable convergence.

### Evaluation

To evaluate the model on a dataset, use the following script:

```bash
python evaluate.py --model_path path_to_your_model.pt --data_dir input_textures --device ("cpu" or "cuda") --batch_size 8
```

### Texture Synthesis

To generate new textures from a given example, run:

```bash
python synthesize.py --model_path path_to_your_model.pt --input_image path_to_input_image.png --output_image path_to_output_image.png --device ("cpu" or "cuda")
```

## Results

The performance of the pseudo-optimizer-based texture synthesis method was evaluated in terms of visual quality and computation time. The results show a significant improvement in efficiency without compromising the quality of the generated textures.

| Metric                | Value               |
| --------------------- | ------------------ |
| Synthesis Time       | 0.5 seconds        |
| Structural Similarity Index (SSIM) | 0.95              |
| Mean Squared Error (MSE) | 0.01              |

Here are some examples of generated textures:

![Generated Textures](images/generated_textures.png)

## Acknowledgements

This project was developed as part of the Deep Learning course at IMT Atlantique, supervised by Pierre-Henri Conze.

## Authors

- **Skander MAHJOUB**, email: skander.mahjoub@imt-atlantique.net
- **Maria FLORENZA**, email: maria.florenza-lamberti@imt-atlantique.net
