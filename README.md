# Adaptive CBF-RRT* 
This repo  is for the implementation of the a sampling-based motion planning algorithm, in which we utilize control barrier function as a low level local planner for RRT*. Furthermore, we equip the algorithm with adaptive sampling procedure to enhance the sampling in the configuration space. For details refer to [the arxiv paper]. 

# Safety-critical local trajectory planning 
The following local planners are used with RRT* instead of traditional steering action with collusion check. 
## CLF-CBF-QP 
Exact steering is essential when rewiring the tree. 
## CBF-QP 
When adding a new sample to the tree, it is ok to accept some deviation from the exact desired sample. We accept such flexibility because the sample is exploratory anyway, this also implies efficiont sampling where the algorithm accept all samples (including the ones which are in the unsafe set). 

# Adaptive sampling 
We use the CE method
