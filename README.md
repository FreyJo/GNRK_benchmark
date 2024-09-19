# About this repository

## Original benchmark
The file `pendulum_benchmark.py` can be used to reproduce the benchmark results presented in the paper:

"Gauss-Newton Runge-Kutta Integration for Efficient Discretization of Optimal Control Problems with Long Horizons and Least-Squares Costs"
by Jonathan Frey, Katrin Baumgärtner and Moritz Diehl

which can be cited as:


```
@Article{Frey2024b,
  Title                    = {{G}auss-{N}ewton {R}unge-{K}utta integration for efficient discretization of optimal control problems with long horizons and least-squares costs},
  Author                   = {Frey, Jonathan and Baumg{\"a}rtner, Katrin and Diehl, Moritz},
  Journal                  = {European Journal of Control},
  Year                     = {2024},

  File                     = {Frey2024b.pdf:pdf/Frey2024b.pdf:PDF},
  Publisher                = {Elsevier}
}
```

The results in this paper were obtained with acados `v0.3.2`
https://github.com/acados/acados/releases/tag/v0.3.2


## Extension to NMPC controllers based on multi-phase OCP formulations
NMPC controllers based on multi-phase OCP formulations have been compared on the same problem formulation.
This can be found in the file `multi_phase_pendulum_benchmark.py`
The work is presented in the paper:

"Multi-Phase Optimal Control Problems for Efficient Nonlinear Model Predictive Control with acados"

by Jonathan Frey, Katrin Baumgärtner, Gianluca Frison and Moritz Diehl

The results in this paper were obtained with acados `v0.4.0`
https://github.com/acados/acados/releases/tag/v0.4.0
