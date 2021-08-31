# DIG Code
Code for "Discretized Integrated Gradients for Explaining Language Models"

<h2 align="center">
  Overview of variants of DIG
  <img align="center"  src="./overview.png" alt="...">
</h2>
**Overview of paths used in DIG and IG**. w is the word being attributed. The gray region is the neighborhood of w. Green line depicts the straight-line path from w to w' used by IG and the green squares are the corresponding interpolation points. **Left**: In DIG-Greedy, we first  monotonize each word in the neighborhood (red arrow). Then the word closest to its corresponding monotonic point is selected as the anchor (blue line to w_5 since the red arrow of w_5 has the shortest magnitude). **Right**: In DIG-MaxCount we first count the number of monotonic dimensions for each word in the neighborhood (shown in [.] above). Then, the word with the highest number of monotonic dimensions is selected as the anchor word (blue line to w_4), followed by changing the non-monotonic dimensions of w_4 (red line to c). Repeating this step gives the zigzag blue path. Finally, the red stars are the interpolated points used by our method. Please refer to the paper for more details.

### Dependencies

- Dependencies can be installed using `requirements.txt`.

### Evaluating DIG:

- Install all the requirements from `requirements.txt.`

- Execute `./setup.sh` for setting up the folder hierarchy for experiments.

- Commands for reproducing the reported results on DistilBERT fine-tuned on SST2:

  ```shell
  # Generate the KNN graph
  python knn.py -dataset sst2 -nn distilbert
  
  # DIG (strategy: Greedy)
  python main.py -dataset sst2 -nn distilbert -strategy greedy

  # DIG (strategy: MaxCount)
  python main.py -dataset sst2 -nn distilbert -strategy maxcount
  ```
  Similarly, commands can be changed for other settings.
