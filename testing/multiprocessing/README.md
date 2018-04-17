# Multiprocessing Sandbox

This sandbox stemmed from my frustrations of the cpu/gpu only being at ~20% (or 150% on linux with an 8 core cpu) when training with only a single process.  To try and make training faster I wanted to see if I could leverage pythons multiprocessing library to load up the cpu and gpu.

## Issue

I am having difficulty getting tflearn/tensorflow to run in multiple processes.  Up to this point I have tried passing the tf computational graph into the processes as a copy (regular input), through a queue and as a global, all to no avail.  The current issue is the processes hang when trying to [predict the action](run_episodes#L24).  I have tried seemingly everything, including adding locks, and namespaces.

If you know how to use a tensorflow model in several processes to utilize all computing power at hand let me know.  I would be extremely interested in seeing how you did it.

## Error Reproduction

1.  Run single_train.py to demonstrate that there are no dimension errors with [run_episodes](run_episodes.py).
```bash
python3 single_train.py
```
2.  Run [open_parallel](open_parallel.py) to show what I expect from multiprocessing.
```bash
python3 open_parallel.py
```
3.  Run [parallel_train](parallel_train.py) to see the issue.  The processes will output 'e' right before calling `model.predict()` and then never get to outputting 'f'.
```bash
python3 parallel_train.py
```
NOTE: The reason it loops though once before hanging is that the model has no weights so it picks random actions instead of using the model.  [Line 45 of NeuralNetwork](NeuralNetwork.py#L45) shows the if statement that determines whether or not to use the model or pick random moves.
