# Hackathon 5

**Beichen Zhang, 17 February 2021**

1. *Write a few sentences about the differences between the `ResBlock` and `Bottleneck` layers above. Why might the Bottleneck block be more suitable for deeper architectures with more layers?* 

   The major difference between a ResBlock and a Bottleneck-based block is that the bottleneck has a 1 by 1 convolution layer at the beginning and end of the architecture.  The first 1 by 1 convolution layer will reduce the depth of channels and computation cost. At the end of the architecture, the other 1 by 1 convolution layer will restore the depth back to the original. Therefore, as compared to the regular ResBlock, a bottleneck-based architechture is expected to perform better in a deeper network because it will simply the parameters and reduce the training time.

2. *Write some python code which builds a network using the general structure described above and either ResBlock or Bottleneck blocks. It doesn't have to be a full set of code that runs, just a function or class that builds a network from these blocks. You might find this architecture useful for homework 1.*

   See the python file: hackathon5_beichen.py