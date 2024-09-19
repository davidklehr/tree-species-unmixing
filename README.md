# Tree Species Unmixing using dense time series
This repository will contain all information and scripts we used for my workflow of the tree species unmixing approach. 

## About
Forests are awesome ecosystems that play a crucial role in maintaining the health of our only planet.
Monitoring and mapping forest stands, along with accurate and current data on forest types, cover, and tree species composition, are vital for sustainable forest management.

This Method aims to map sub-pixel tree species mixtures of different tree speices in heavily mixed and strongly fragmented temperate forests like in Germany.

The workflow is seperated in the following steps:
* Step 1: Get yout time series data
** Step 1: Generation of a dense Sentienl-2 time series.\
          In our approach, we used an forest specific spline interpolation method. The corresponding R-code can be found here [<a href="https://github.com/davidfrantz/force-udf/tree/main/rstats/ts" >https://github.com/davidfrantz/force-udf/tree/main/rstats/ts</a>]. However in theory alternative interpolation methods can be used (RBF, linear), just ensure some serious denseness and observations in phenological important months (spring for budbust, fall for leaf decay).
** Step 2: (optional but recommended) Data cubing.\
As descibed in our publication we used FORCE (<a href="https://github.com/davidfrantz/force" >https://github.com/davidfrantz/force</a>) for data preprocessing. We recommend to do the same as it is a performant alternative for large data amount and developed for satelite processing of Sentinel-2 and Landsat. The data cubing is not entirely necessary, but will increase the processing speed by far using parallelization.
* Step 3 [ Link ]
* Step 4
* ...

Test

You will find the used python and R scripts in the according src folder. And some Documentation in the setup_guide.md.
I would recommend to start at the setup guide, as we give aou some information about used python libraries, folder structure, etc. here.
If something is unclear, don't be shy and ask. We give my best to make my workflow as clear as possible.

Feel free to use the code and adapt it to your own projects.
Just be fair when publishing it to refer to the oroginal work by us:

* Reference will follow

and Vu-Dong:

* Pham, V.-D., Thiel, F., Frantz, D., Okujeni, A., Schug, F., van der Linden, S., 2024. Learning the variations in annual spectral-temporal metrics to enhance the transferability of regression models for land cover fraction monitoring. Remote Sensing of Environment 308. https://doi.org/10.1016/j.rse.2024.114206.



