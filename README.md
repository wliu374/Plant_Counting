# Plant_Counting
## IntegrateNet: A Deep learning Network for maize Stand Counting from UAV Imagery by Integrating Density and Local Count Maps.
Wenxin Liu, Jing Zhou, Biwen Wang, Martin Costa, Shawn M. Kaeppler, Zhou Zhang, Member, IEEE <br>  <br>
This research was supported by USDA National Institute of Food and Agriculture, AFRI project 1028196. (Corresponding author: Zhou Zhang.)<br><br>
Wenxin Liu is with the Department of Electrical and Computer Engineering, University of Wisconsin-Madison, Madison, WI 53706, USA.<br><br>
Jing Zhou, Biwen Wang, and Zhou Zhang are with the Department of  Biological Systems Engineering, University of Wisconsin-Madison, Madison, WI 53706, USA (e-mail: zzhang347@wisc.edu).<br><br>
Martin Costa, and Shawn M. Kaeppler are with the Department of Agronomy, University of Wisconsin–Madison, Madison, WI 53706, USA. <br><br>

## Abstract
Crop stand count plays an important role in modern agriculture as a reference for precision management activities and in interpreting plant breeding data. The traditional counting method by visual ratings is extremely tedious, inefficient, and error-prone. Recent applications of unmanned aerial vehicles carrying cameras and sensors facilitate data collection efficiency in agricultural fields. With the development of computer vision and deep learning models, plant counting from high-resolution imagery serves as a promising alternative to visual ratings. Generating density maps out of crop canopy images and regressing redundant local counts from an image are two mainstream existing methods. However, the density map is easily biased by object size variations, while the local count suffers poor output visualization and examination. In this study, a new network - IntegrateNet was proposed to supervise the learning of density map and local count simultaneously and thus boost the model performance by balancing the trade-off between their errors. The IntegrateNet was trained and validated with an image set containing 124 maize aerial images. The model achieved an excellent result for 24 test images with the root mean square error of 2.28, and the Coefficient of determination (R2) of 0.9578 between the predicted and ground-truth maize stand counts. In conclusion, the proposed model provides an efficient solution for counting maize stands at early stages and could be used as a reference for similar studies.

## Structure
![](https://github.com/wliu374/Plant_Counting/blob/main/structure.png) 

## Dataset
* 124 maize plots with 4864 × 3648 resolution
* stand counts range 53 to 102
* acquired by Phantom 4 Pro V2.0 at 12 m above ground level(AGL) <br><br>
![](https://github.com/wliu374/Plant_Counting/blob/main/figures/Fig.3.PNG)<br>
![](https://github.com/wliu374/Plant_Counting/blob/main/figures/Fig.4.PNG)
## Comparison
IntegrateNet is created based on TasselNet versions. <br> <br>
H. Lu, L. Liu, Y.-N. Li, X.-M. Zhao, X.-Q. Wang, and Z.-G. Cao, “Tasselnetv3: Explainable plant counting with guided upsampling and background suppression,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–15, 2021.  <br><r>
![](https://github.com/wliu374/Plant_Counting/blob/main/scatterplot.png) 
  
| Model | MAE | RMSE | 	R<sup>2</sup> |
| :-------------: | :-------------: | :-------------: | :-------------: |
| IntegrateNet | 1.62 | 2.28 | 0.9578 |
| TasselNetV3-Lite | 2.39 | 3.69 | 0.8893 |
| TasselNetV3-Seg† | 3.06 | 4.64 | 0.8248 |
| TasselNetV3-Seg‡ | 3.56 | 4.93 | 0.8027 |
  
## Results
![](https://github.com/wliu374/Plant_Counting/blob/main/figures/figure3.PNG)

## How to use
The concrete introduction of the repository is in [Tutorial](https://github.com/wliu374/Plant_Counting/blob/main/Tutorial.pdf).

