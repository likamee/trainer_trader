# Oanda Trainer and Trader

## Description

This is a Python project that uses Oanda to retrieve candle information. It has three modes: Training, Backtesting, and Realtime Trading. The training procedure performs a Particle Swarm Optimization (PSO) train on a window of past candles to achieve the best parameters for the strategy. These parameters are then used in the next window for a real-time trading process. This code is ready for live trading.

## Getting Started

### Dependencies

* Docker
* Poetry

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets

## Files

The main folder contains three files representing the three methods:

1. `backtest.py`: This file processes the backtesting.
2. `realtime.py`: This file handles the realtime trading.
3. `training.py`: This file manages the training.

In addition, the `app->optimizer` folder contains the PSO optimization.

## Help

If you have any question or issues, feel free to contact me.

## Authors

Contributors names and contact info

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the MIT License
