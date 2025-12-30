# 3lcvrp-alns-dblf

Algorithm implementation of [Advanced loading constraints for 3D vehicle routing problems](https://doi.org/10.1007/s00291-021-00645-w)

## How to run
```python 3l_cvrp_alns/run_experiments.py --instances [dataset folder] --out [output folder] --time [time limit per run] --iter [max interations] --repeats [repeats per instance] --start-instance [resume from this instance] --start-repeat [resume the instance from this iteration (indexed from 0)]```

Example:

```python 3l_cvrp_alns/run_experiments.py --instances 3l_cvrp_alns/instances/Gendreau_et_al_2006 --out 3l_cvrp_alns/output/Gendreau_et_al_2006 --time 3600 --iter 100 --repeats 10 --start-instance 3l_cvrp13.txt --start-repeat 4```

## References
- Krebs, C., Ehmke, J.F. & Koch, H. Advanced loading constraints for 3D vehicle routing problems. OR Spectrum 43, 835–875 (2021). https://doi.org/10.1007/s00291-021-00645-w