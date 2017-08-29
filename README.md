Building Trading Models Using Reinforcement Learning
==================


This repository contains **the framework built to my dissertation** of the quantitative finance mastership program, from [FGV](http://portal.fgv.br/en/news/fgv-among-worlds-10-best-think-tanks) University. I proposed the use of a learning algorithm and tile coding to develop an interest rate trading strategy directly from historical high-frequency order book data.

<p align="center"><img src="book_sim.gif" alt="Example simulator" width="58%" style="middle"></p>

No assumption about market dynamics was made, but it has required the creation of this simulator wherewith the learning agent could interact to gain experience. You can check my master thesis <a href="http://hdl.handle.net/10438/18707" target="_blank">here </a> and the presentation <a href="https://ucaiado.github.io/AdaptativeTrading_Model/" target="_blank">here</a>. Both are in Portuguese. The code structure is heavily inspired by Udacity's [smartcab](https://github.com/udacity/machine-learning/tree/master/projects/smartcab) project and in OpenAi's [Gym](https://github.com/openai/gym).


### Install
This project requires **Python 2.7** and the following Python libraries installed:

- [Bintrees](https://pypi.python.org/pypi/bintrees/2.0.2)
- [Matplotlib](http://matplotlib.org/)
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [Seaborn](https://web.stanford.edu/~mwaskom/software/seaborn/)
- [BeautifulSoup](https://pypi.python.org/pypi/beautifulsoup4)


### Run
In a terminal or command window, navigate to the top-level project directory `rl_trading/` (that contains this README) and run the following command:

```shell
$ python -m market_sim.agent [-h] [-t] [-d] [-s] [-m] <OPTION>
```

Where *OPTION* is the kind of agent to be run. The flag *[-t]* is the number of trials to perform using the same file, *[-d]* is the date of the file to use in the simulation, *[-m]* is the month of the *date* flag and *[-s]* is the number of sessions on each trial. Use the flag *[-h]* to get information about what kind of agent is currently available, as well as other flags to use. The simulation will generate log files to be analyzed later on. Be aware that any of those simulations options might take several minutes to complete.


### Data
An example of the datasets used in this project can be found [here](https://www.dropbox.com/s/xo5ul1h3hmtfw1k/201702.zip?dl=0). Unzip it and include in the folder `data/preprocessed`.


### Main References
1. GOULD, M. D. et al. *Limit order books*. Quantitative Finance, 2013.
2. CHAN, N. T.; SHELTON, C. *An electronic market-maker*. 2001.
3. BUSONIU, L. et al. *Reinforcement learning and dynamic programming using function approximators*. CRC press, 2010.
4. SUTTON, R. S.; BARTO, A. G. *Reinforcement Learning: An Introduction*, draft, in progress. 2st. MIT Press, 2017.

### License
The contents of this repository are covered under the [Apache 2.0 License](LICENSE.md).
