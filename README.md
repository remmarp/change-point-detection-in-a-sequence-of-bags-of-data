# change-point-detection-in-a-sequence-of-bags-of-data
This repo contains implementation of '[change-point detection in a sequence of bags-of-data](https://ieeexplore.ieee.org/abstract/document/7095580)' with python.
Note that this is not an official repo for the paper. 

With a toy example, this repo tries to detect cpd. 

## With change-point score based on log likelihood ratio
### CPD Results with Data
![alt text][LL_CP_Data] 
### CP score
![alt text][LL_CP_Score] 
### Adaptive theresholding (Gamma)
![alt text][LL_CP_Gamma] 

## With change-point score based on symmetrized KL divergence
### CPD Results with Data
![alt text][KL_CP_Data] 
### CP score
![alt text][KL_CP_Score] 
### Adaptive theresholding (Gamma)
![alt text][KL_CP_Gamma] 

[LL_CP_Data]: https://github.com/remmarp/change-point-detection-in-a-sequence-of-bags-of-data/blob/master/assets/LL_CP_data.png "LL_CP_Data"
[LL_CP_Score]: https://github.com/remmarp/change-point-detection-in-a-sequence-of-bags-of-data/blob/master/assets/LL_CP_score.png "LL_CP_Score"
[LL_CP_Gamma]: https://github.com/remmarp/change-point-detection-in-a-sequence-of-bags-of-data/blob/master/assets/LL_CP_gamma.png "LL_CP_Gamma"

[KL_CP_Data]: https://github.com/remmarp/change-point-detection-in-a-sequence-of-bags-of-data/blob/master/assets/LL_CP_data.png "KL_CP_Data"
[KL_CP_Score]: https://github.com/remmarp/change-point-detection-in-a-sequence-of-bags-of-data/blob/master/assets/LL_CP_score.png "KL_CP_Score"
[KL_CP_Gamma]: https://github.com/remmarp/change-point-detection-in-a-sequence-of-bags-of-data/blob/master/assets/LL_CP_gamma.png "KL_CP_Gamma"
