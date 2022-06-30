# TripletCell
TripletCell: a deep metric learning framework for accurate annotation of cell types at the single-cell level 

Abstract
Single-cell transcriptomics has significantly accelerated the experimental characterization of distinct cell lineages and types in complex tissues and organisms. However, experiment-based cell type identification heavily relies on the generation of high-quality of transcriptomic data and manual annotation, which can be laborious and time-consuming. Furthermore, the heterogeneity of scRNA-seq datasets poses another challenge for accurate cell type annotation, such as the batch effect induced by different scRNA-seq protocols and samples. To overcome these limitations, here we propose a novel pipeline, termed TripletCell, for cross-species, cross-protocol, and cross-sample cell type annotation. We develop a cell embedding and dimension-reduction module for the feature extraction (FE) in TripletCell, namely TripletCell-FE, to leverage the deep metric learning-based algorithm the relationships between the reference gene expression matrix and the query cells.
![image](overflow.png)

Requirement:

scanpy  1.7.2

scikit-learn  0.24.2

torch  1.10.0

python 3.6.13

#Datasets:

https://zenodo.org/record/3357167#.Yr0WvBVBwuU

#Usage:

python main.py

#Connect

If you have any questions, please contact yanliu@njust.edu.cn
