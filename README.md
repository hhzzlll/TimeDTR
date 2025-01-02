# DUAL TRAJECTORY REVISED DIFFUSION MODEL FOR TIME SERIES FORECASTING
## TimeDTR
This repository contains code for the paper, DUAL TRAJECTORY REVISED DIFFUSION MODEL FOR TIME SERIES FORECASTING.
## Abstract
Diffusion models have exhibited state-of-the-art performance in generative tasks across various domains. A few recent works leveraged the powerful modeling ability of the diffusion model to time-series forecasting, leading to a significant breakthrough. However, all these works perform the forecasting through incorporating the historical time-series conditions into the backward denoising. This causes the diffusion model to lose the essential consistency between forward and backward processes, thereby limiting the precision of the inference. In this paper, we propose a novel Dual Trajectory Revised Diffusion Model (TimeDTR) for time-series forecasting, which leverages an unconventional conditioning strategy to incorporate the historical information into both forward and backward trajectories in the diffusion model. Experimental results on six real-world datasets demonstrate that TimeDTR takes a big step forward from the state-of-the-art in time-series forecasting, especially in the long-term forecasting tasks, in terms of forecasting accuracy. The codes of the experiments with datasets and our algorithms are available at https://github.com/hhzzlll/TimeDTR.
## Overview of TimeDTR
![image](https://github.com/user-attachments/assets/542c38fc-2e23-4296-9abd-114e79d90b86)
## Train and Inference
[python main_ddpm.py](url)
## Experiments
### 1.Comparison experiment
![image](https://github.com/user-attachments/assets/bf5bc4b4-8d8a-481c-808d-967c8eb32d39)
### 2.Visualizations of Forecasting Performance
![image](https://github.com/user-attachments/assets/862809af-d059-47dc-b0d3-8a62c90419b8)
### 3.Ablation Study of TimeDTR
![image](https://github.com/user-attachments/assets/cdbb4d54-0a98-42bb-b9a9-ee0878c0bf68)
## Acknowledgement
This research was supported by the National Natural Science Foundation of China (NSFC) under the grant No. 62372149, No. U23A20303 and No. 62472140
