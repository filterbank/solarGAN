# Perception-Evaluation
We provide PyTorch implementations for [Perception Evaluation – A new solar image quality metric based on the multi-fractal property of texture features](https://arxiv.org/pdf/1905.09980.pdf)
## Require
- Python 3.5 
- Pytorch 0.4.0 
- numpy 
- CUDA 8.0 
## Test Data
  ```
  cd /PeMeasureData/
  ```
The image H_000000.fits is real observation data and ref_x1.fits is reference image from Speckle reconstruction.

## Test
  ```
  python3 test.py
  ```
