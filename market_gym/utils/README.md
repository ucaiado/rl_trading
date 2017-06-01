# Utilities

Implement several functions and libraries used by other components, like:

- di_utilities.py: It is specific for Brazilian interest rate market. Calculate working days, recover settlements prices from the BVMF website and so on.
- eda.py: Several functions to help visualize the results. May have functions that do not work anymore.
- microstruct_stats.py: Implement a structure to hold information about microstructure data. Use classes from `utils.py`.


## Examples related to `di_utilities.py`

Navigate to the top-level project directory `rl_trading/` and access ipython console and type

```python
    from market_gym.utils import di_utilities

    for s_cmm in ['DI1N18', 'DI1N19', 'DI1F20']:
        print s_cmm, di_utilities.calculateDI1expirationDays(s_cmm, s_today='01/05/2017')
```

It should return:

```shell
    DI1N18 (270.0, datetime.date(2018, 7, 1), datetime.date(2017, 6, 1))
    DI1N19 (519.0, datetime.date(2019, 7, 1), datetime.date(2017, 6, 1))
    DI1F20 (649.0, datetime.date(2020, 1, 1), datetime.date(2017, 6, 1))
```
