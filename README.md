# Attentive LSTM with Adversarial Training for Stock Movement Prediction with Behavioral Features

## A project for my research thesis "Stock Movement Price Prediction with Behavioral Features"

* **TensorFlow to PyTorch Upgrade**: Upgraded the codebase from TensorFlow to PyTorch.
* **Adversarial Advantage**: Inspired by Goodfellow et al. (2014), incorporated adversarial training to make the model more robust against real-world market noise. (https://arxiv.org/abs/1412.6572)
* **Feature Engineering**: Engineered new features based on behavioral economics to capture investor psychology (regret & average return), enhancing prediction accuracy.
* **Time Series**: The model is specifically designed for time series data, crucial for predicting future stock movements.


In Partial Fulfillment of The Requirements for the Degree of Master of Science in Data Science
Submitted to the Senate of the Technion - Israel Institute of Technology. 

Advised by Dr. Ori Plonsky and Prof. Margarita Osadchy. 

the following are $examples$ of run commands:

* -p ../data/kdd17/preprocessed_rania_withdist_PiXi -l 15 -u 32 -l2 0.01 -v 1 -rl 0 -la 0.05 -le 0.001 -f 0 -r 1e-2 -seed 100 -hi 1 -shuffle 0
* -p ../data/stocknet-dataset/price/preprocessed_rania_withdist_PiXi -l 15 -u 32 -l2 0.01 -v 1 -rl 0 -la 0.05 -le 0.001 -r 1e-2 -seed 100 -shuffle 0 -hi 1
* -p ../data/synthetic_data_2/preprocessed/ -l 15 -u 16 -l2 0.001 -v 1 -rl 0 -la 0.05 -le 0.001 -r 0.01 -seed 100 -shuffle 0.

#### Note:
* **Thesis PDF**:  The research thesis for this project, titled "Stock Movement Price Prediction with Behavioral Features",  is available at: https://drive.google.com/file/d/1AJlAgYA-08ARaHSrxnsXrJxyaRd3Xnk9/view?usp=sharing 
