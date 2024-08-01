#  Space-Time SwarmPRM

We add time dimesion for SwarmPRM. Number of agent on each node is constrained by the available area in each node.

Macroscopically, we have GMMs representing groups of agents travelling on the map. The GMM has a max capacity that does not exceed the current maps max flow. 

GMM does not constraint the number of agents in each distribution. We therefore constraint the 

## Discussion

The Gaussian map does not have a temporal dimension.

## Algorithm

The travese capacity corresponds to the max flow from the start distributions to goal distributions

During transportation, several agents need to wait at certain nodes for other agents to finish first. 

## Experiments

We create environment with spherical obstacles (?). 

### Gaussian PRM 

Consider the problem of constructing a Gaussian PRM. We consider the following 
strategies. We sample points similar to a regular PRM 

Gaussian Sampling Strategy

https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/PRM/prmsampling_02.pdf

#### Univariate Gaussian

We consider all the gaussian nodes has a spherical shape and the covariance matrix
has the form: 

[
    sigma^2     0
    0         sigma^2
]

The sigma value is computed by comparing the center of the point to the closest obstacle. Setting it as the isocontour of a certain probability density on the PDF, we compute the corresponding variance value. The capacity is computed based on the area of the circle.

#### Gaussian Region with iterative refinement

Refer to paper:
https://arxiv.org/pdf/2403.02977
https://groups.csail.mit.edu/robotics-center/public_papers/Deits14.pdf

IRIS distro: https://github.com/rdeits/iris-distro

Iteratively update an ellipsoid region until it collides with the environment.
Construct the Gaussian 


### Dev notes

Assumptions:

We assume there are multiple teams of agents on the map. At every timestep, each
team of agents follow a Gaussian Mixture distribution on the map, with the center
of the distributions matches a center of the Gaussian distribution on the map.
All the distributions are synchronized, meaning that the agents must reach a static
distribution before they transport to the next distribution.

Population per node = Size of Team * coefficient for the corresponding Gaussian distribution.

Optimization objective: makespan & traverse cost


How to balance the distribution of agents through the map?