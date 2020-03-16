# Sparse Graphical Memory (SGM)

[Sparse Graphical Memory (SGM)](https://mishalaskin.github.io/sgm/) is a data structure for reinforcement-learning agents to solve long-horizon, sparse-reward navigation tasks.
This codebase is a TensorFlow implementation of SGM accompanying the paper [Sparse Graphical Memory for Robust Planning](https://arxiv.org/abs/2003.06417).

## To install:
1. Create a new conda environment: `conda create -n sgm python=3.6`
2. Activate the conda environment: `conda activate sgm`
3. Install the requirements: `pip install -r requirements.txt`
4. Install the package: `pip install -e .`

## To launch the experiments:
1. To launch SGM experiments, run all the cells in `notebooks/SGM in Thinned FourRooms.ipynb`
2. To launch SoRB experiments, run all the cells in `notebooks/SoRB in Thinned FourRooms.ipynb`

## To visualize the results:
1. Call `sgm/plot.py` with the directories of the logs produced by the experiments followed by the `--returns_v_cleanup_steps` flag, e.g., `python sgm/plot.py logs/thinned_fourrooms_sgm logs/thinned_fourrooms_sorb --returns_v_cleanup_steps`
2. Find the visualized results in the `plots` directory, e.g., in `plots/returns_v_cleanup_steps`

## Commands you may find useful
Enable rendering of environments when there is no display, e.g., on a server: `xvfb-run -s "-screen 0 1400x900x24" jupyter notebook` [https://stackoverflow.com/questions/40195740/how-to-run-openai-gym-render-over-a-server](source)

Create tunnel from local machine to remote machine: `ssh -N -f -L localhost:8888:localhost:8888 user@remote.hostname.edu`

Run Jupyter notebook with no time limit on cell execution and replace the notebook's contents with the new output: `jupyter nbconvert --ExecutePreprocessor.timeout=-1 --execute --to notebook --inplace <notebook.ipynb>` [https://stackoverflow.com/questions/35545402/how-to-run-an-ipynb-jupyter-notebook-from-terminal](source)

## Credits

This code was built upon the [code released by Eysenbach et al.](http://bit.ly/rl_search) under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Reference

```
@unpublished{laskin2020sparse,
  title={Sparse Graphical Memory for Robust Planning},
  author={Laskin, Michael and Emmons, Scott and Jain, Ajay and Kurutach, Thanard and Abbeel, Pieter and Pathak, Deepak},
  note={arXiv:2003.06417}
}
```