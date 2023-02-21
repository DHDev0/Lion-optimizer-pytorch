# Lion-optimizer-pytorch
Lion optimizer pytorch  
From: https://arxiv.org/pdf/2302.06675.pdf  
Base on implementation : https://github.com/lucidrains/lion-pytorch  
  
Example: 
~~~python
import lion
optimizer = Lion( model_params, lr = 1e-4, betas = (0.9, 0.99), weight_decay = 0.0) 
~~~ 
