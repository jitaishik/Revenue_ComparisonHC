# Revenue_ComparisonHC
This is a python implementation of revenue function of comparison-based hierarchical clustering algorithms.

## Citing Revenue_ComparisonHC
If you use this software please cite the following publication:
```
@misc{https://doi.org/10.48550/arxiv.2211.16459,
  doi = {10.48550/ARXIV.2211.16459},
  
  url = {https://arxiv.org/abs/2211.16459},
  
  author = {Mandal, Aishik and Perrot, Michaël and Ghoshdastidar, Debarghya},
  
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {A Revenue Function for Comparison-Based Hierarchical Clustering},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Installation
You can install this package by cloning the repository and running the setup file:
```
git clone https://github.com/jitaishik/Revenue_ComparisonHC.git
cd Revenue_ComparisonHC
python setup.py install
```

## Examples
* [Revenue for AddS-3,t-STE,MulK-3 on synthetic data](https://github.com/jitaishik/Revenue_ComparisonHC/blob/main/triplets.py)
* [Revenue for AddS-4,4K-AL on synthetic data](https://github.com/jitaishik/Revenue_ComparisonHC/blob/main/quadruplets.py)
* [Revenue for AddS-3,t-STE,MulK-3 on glass](https://github.com/jitaishik/Revenue_ComparisonHC/blob/main/examples/triplets/Glass_triplet.ipynb)
* [Revenue for AddS-4,4K-AL on glass](https://github.com/jitaishik/Revenue_ComparisonHC/blob/main/examples/quadruplets/Glass_quad.ipynb)
* [Revenue for AddS-3,t-STE,MulK-3 on car](https://github.com/jitaishik/Revenue_ComparisonHC/blob/main/examples/triplets/Car_triplet.ipynb)
* [Revenue for AddS-4,4K-AL on car](https://github.com/jitaishik/Revenue_ComparisonHC/blob/main/examples/quadruplets/Car_quad.ipynb)

## References
* [AddS-Clustering](https://github.com/mperrot/AddS-Clustering)
* [ComparisonHC](https://github.com/mperrot/ComparisonHC)
* [cblearn](https://github.com/dekuenstle/cblearn)
