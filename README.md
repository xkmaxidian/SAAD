# SA2E

## SA2E: spatial-aware auto-encoder for cell type deconvolution of spatial transcriptomics data

### Yaxiong Ma, Zengfa Dou, Yuhong Zha and Xiaoke Ma

Spatial transcriptomics (ST) technologies  measure transcriptions of cells together with their spatial location, but each spot of ST usually contains a mixture of cell types, posing a great challenge for downstream analysis.   Cell type deconvolution deciphers mixtures of cell types of spots by integrating single-cell RNA-seq (scRNA-seq) and ST data. Current methods utilize the given bio-markers to construct signatures of cell types, failing to fully precisely decipher abundance of cell types with limited or no known bio-markers.  To address this limitation, we propose  a spatial-aware auto-encoder framework (called SA2E) for cell type deconvolution of ST data without requiring bio-markers of cell types, where signatures of cell types  are automatically learned from data. Specifically,  SA2E employs a spatial-aware auto-encoder to learn the latent features of spots, where the local topological structure of spatial graph of spots is preserved with graph regularization. Unlike available algorithms, SA2E automatically learns signatures of cell types by leveraging latent features of spots obtained by auto-encoder by enforcing signatures of cell types to reconstruct transcription of ST,  thereby implicitly associating cell types with ST data.   Extensive experiments demonstrate that SA2E outperforms current state-of-the-art methods for  cell type deconvolution on the simulated and real ST data, proving that signatures of cell types are learnable, which provides alternatives for deciphering cell types of ST data.

![SA2E workflow](docs/SA2E.png)

## System Requirements

pandas, numpy, scanpy, matplotlib, seaborn, anndata, leidenalg, POT

### Compared deconvolution algorithms

* [Cell2location](https://github.com/BayraktarLab/cell2location)
* [RCTD](https://github.com/vigneshshanmug/RCTD)
* [SPOTlight](https://rdrr.io/github/MarcElosua/SPOTlight)
* [Redeconve](https://github.com/ZxZhou4150/Redeconve)
* [CARD](https://yma-lab.github.io/CARD/documentation/04_CARD_Example.html)
* [DestVI](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/spatial/DestVI_tutorial.html)
* [SpatialDWLS](https://github.com/QuKunLab/SpatialBenchmarking/blob/main/Codes/Deconvolution/SpatialDWLS_pipeline.r)
* [DSTG](https://github.com/Su-informatics-lab/DSTG/tree/main)

## Compared spatial domain identification algorithms

Algorithms that are compared include: 

* [SpaGCN](https://github.com/jianhuupenn/SpaGCN)
* [GraphST](https://deepst-tutorials.readthedocs.io/en/latest/)
* [stLearn](https://github.com/BiomedicalMachineLearning/stLearn)

### Contact:

We are continuing adding new features. Bug reports or feature requests are welcome.

Please send any questions or found bugs to Xiaoke Ma [xkma@xidian.edu.cn](mailto:xkma@xidian.edu.cn).

