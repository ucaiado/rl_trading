#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Library to execute a exploratory data analysis (EDA). It is an approach to
analyzing data sets to summarize their main characteristics, often with visual
methods. Primarily EDA is for seeing what the data can tell us beyond the
formal modeling or hypothesis testing task.


@author: ucaiado

Created on 10/20/2016
"""

###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
try:
    import warnings
    from IPython import get_ipython
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import defaultdict
    import json
    import numpy as np
    import pandas as pd
    import StringIO
    warnings.filterwarnings('ignore', category=UserWarning,
                            module='matplotlib')
    # Display inline matplotlib plots with IPython
    get_ipython().run_line_magic('matplotlib', 'inline')
    # aesthetics
    sns.set_palette('deep', desat=.6)
    sns.set_context(rc={'figure.figsize': (8, 4)})
    sns.set_style('whitegrid')
    sns.set_palette(sns.color_palette('Set2', 10))
    # loading style sheet
    get_ipython().run_cell('from IPython.core.display import HTML')
    get_ipython().run_cell('HTML(open("ipython_style.css").read())')
except:
    pass
###########################################

'''
Begin help functions
'''


def func_estimator(x):
    '''
    pseudo estimator to be used by poinplot
    '''
    return x[0]

'''
End help functions
'''


def read_logs(i_desired_trial, s_fname):
    '''
    Return a dictionary with information for the passed log file and trial and
    the number of trades in the main instrument of the strategy

    :param i_desired_trial: integer. the trial ID to colect data
    :param s_fname: string. the name of the log file analized
    '''
    with open(s_fname) as fr:
        # initiate the returned dictionary ans other control variables
        d_rtn = {'pnl': defaultdict(dict),
                 'position': defaultdict(dict),
                 'duration': defaultdict(dict),
                 'mid': defaultdict(dict)}
        f_reward = 0.
        f_count_step = 0
        last_reward = 0.
        i_trial = 0
        i_trades = 0

        for idx, row in enumerate(fr):
            if row == '\n':
                continue
            # extract desired information
            # count the number of trials
            if ' New Trial will start!' in row:
                i_trial += 1
                f_count_step = 0
                f_reward = 0
            elif '.update():' in row and i_trial == i_desired_trial:
                s_aux = row.strip().split(';')[1]
                s_x = row.split('time = ')[1].split(',')[0]
                s_date_all = s_x
                s_x = s_date_all[:-7]
                s_date = s_x
                ts_date_all = pd.to_datetime(s_date_all)
                ts_date = pd.to_datetime(s_date)
                last_reward = float(s_aux.split('reward = ')[1].split(',')[0])
                f_duration = float(s_aux.split('duration = ')[1].split(',')[0])
                f_reward += last_reward
                f_count_step += 1.
                # extract some data
                d_rtn['duration'][i_trial][ts_date_all] = f_duration
                if ', position = ' in s_aux:
                    s_pos = s_aux.split(', position = ')[1].split('}')[0][1:]
                    s_pos = s_pos.replace("'", "")
                    l_pos = [(a.strip(), float(b)) for a, b in
                             [s.split(': ')for s in s_pos.split(',')]]
                    d_rtn['position'][i_trial][ts_date_all] = dict(l_pos)
                if ', pnl = ' in s_aux:
                    s_action = s_aux.split(', pnl = ')[1].split(',')[0]
                    f_aux = float(s_action)
                    d_rtn['pnl'][i_trial][ts_date_all] = f_aux
                if 'crossed_prices' in s_aux or 'correction_by_trade' in s_aux:
                    i_trades += 1
                if ', inputs = ' in s_aux:
                    s_mid = s_aux.split(', inputs = ')[1].split("{'midPrice':")
                    s_mid = s_mid[1][1:].split('}}')[0]
                    s_mid = s_mid.replace("'", "")[1:]
                    l_pos = [(a.strip(), float(b)) for a, b in
                             [s.split(': ')for s in s_mid.split(',')]]
                    d_rtn['mid'][i_trial][ts_date_all] = dict(l_pos)
            # finish the loop as soon as the trial is analyzed
            if i_trial > i_desired_trial:
                break
    return d_rtn, i_trades


def read_logs2(i_desired_trial, s_fname):
    '''
    Return a dictionary with information for the passed log file and trial and
    the number of trades in the main instrument of the strategy

    :param i_desired_trial: integer. the trial ID to colect data
    :param s_fname: string. the name of the log file analized
    '''
    with open(s_fname) as fr:
        # initiate the returned dictionary ans other control variables
        d_rtn = {'pnl': defaultdict(dict),
                 'position': defaultdict(dict),
                 'duration': defaultdict(dict),
                 'mid': defaultdict(dict)}
        f_reward = 0.
        f_count_step = 0
        last_reward = 0.
        i_trial = 0
        i_trades = 0

        for idx, row in enumerate(fr):
            if row == '\n':
                continue
            # extract desired information
            # count the number of trials
            if ' New Trial will start!' in row:
                i_trial += 1
                f_count_step = 0
                f_reward = 0
            elif '.update():' in row and i_trial == i_desired_trial:
                s_aux = row.strip().split(';')[1]
                s_x = row.split('time = ')[1].split(',')[0]
                s_date_all = s_x
                s_x = s_date_all[:-7]
                s_date = s_x
                ts_date_all = pd.to_datetime(s_date_all)
                ts_date = pd.to_datetime(s_date)
                last_reward = float(s_aux.split('reward = ')[1].split(',')[0])
                f_duration = float(s_aux.split('duration = ')[1].split(',')[0])
                f_reward += last_reward
                f_count_step += 1.
                # extract some data
                d_rtn['duration'][i_trial][ts_date_all] = f_duration
                if ', position = ' in s_aux:
                    s_pos = s_aux.split(', position = ')[1].split('}')[0][1:]
                    s_pos = s_pos.replace("'", "")
                    l_pos = [(a.strip(), float(b)) for a, b in
                             [s.split(': ')for s in s_pos.split(',')]]
                    d_rtn['position'][i_trial][ts_date_all] = dict(l_pos)
                if ', pnl = ' in s_aux:
                    s_action = s_aux.split(', pnl = ')[1].split(',')[0]
                    f_aux = float(s_action)
                    d_rtn['pnl'][i_trial][ts_date_all] = f_aux
                if 'crossed_prices' in s_aux or 'correction_by_trade' in s_aux:
                    i_trades += 1
                if ', inputs = ' in s_aux:
                    s_mid = s_aux.split(', inputs = ')[1].split("{'midPrice':")
                    s_mid = s_mid[0].split(', position =')[0]
                    s_mid = s_mid.replace("'", '"').replace('None', '0')
                    l_mid = json.loads(s_mid)
                    s_mid = s_mid.replace("'", "")[1:]
                    l_mid = [(s_key, (float(x))) for s_key, x
                             in l_mid['midPrice'].iteritems()]
                    d_rtn['mid'][i_trial][ts_date_all] = dict(l_mid)
            # finish the loop as soon as the trial is analyzed
            if i_trial > i_desired_trial:
                break
    return d_rtn, i_trades


def read_logs_to_form_spread(i_desired_trial, s_fname):
    '''
    Return a dictionary with information for the passed log file and trial and
    the number of trades in the main instrument of the strategy (just F21 and
    F19)

    :param i_desired_trial: integer. the trial ID to colect data
    :param s_fname: string. the name of the log file analized
    '''
    with open(s_fname) as fr:
        # initiate the returned dictionary ans other control variables
        d_rtn = {'pnl': defaultdict(dict),
                 'position': defaultdict(dict),
                 'mid': defaultdict(dict),
                 'duration': defaultdict(dict),
                 'TOB_F21': defaultdict(dict),
                 'TOB_F19': defaultdict(dict),
                 'MY_PRICES': defaultdict(dict),
                 'EXEC': defaultdict(dict),
                 'LAST_SPREAD': defaultdict(dict)}  # where I am most aggres
        f_reward = 0.
        f_count_step = 0
        last_reward = 0.
        i_trial = 0
        i_trades = 0
        l_trade_actions = ['TAKE', 'crossed_prices', 'correction_by_trade',
                           'HIT']

        for idx, row in enumerate(fr):
            if row == '\n':
                continue
            # extract desired information
            # count the number of trials
            if ' New Trial will start!' in row:
                i_trial += 1
                f_count_step = 0
                f_reward = 0
            elif '.update():' in row and i_trial == i_desired_trial:
                s_aux = row.strip().split(';')[1]
                s_x = row.split('time = ')[1].split(',')[0]
                s_date_all = s_x
                s_x = s_date_all[:-7]
                s_date = s_x
                ts_date_all = pd.to_datetime(s_date_all)
                ts_date = pd.to_datetime(s_date)
                last_reward = float(s_aux.split('reward = ')[1].split(',')[0])
                f_duration = float(s_aux.split('duration = ')[1].split(',')[0])
                f_reward += last_reward
                f_count_step += 1.
                # extract some data
                d_rtn['duration'][i_trial][ts_date_all] = f_duration
                if ', position = ' in s_aux:
                    s_pos = s_aux.split(', position = ')[1].split('}')[0][1:]
                    s_pos = s_pos.replace("'", "")
                    l_pos = [(a.strip(), float(b)) for a, b in
                             [s.split(': ')for s in s_pos.split(',')]]
                    d_rtn['position'][i_trial][ts_date_all] = dict(l_pos)
                if 'action = ' in s_aux:
                    s_action = row.split('action = ')[1].split(',')[0].strip()
                    d_aux2 = {'DI1F21': 0, 'DI1F19': 0}
                    if ts_date_all not in d_rtn['EXEC'][i_trial]:
                        d_rtn['EXEC'][i_trial][ts_date_all] = d_aux2.copy()
                    d_aux2 = d_rtn['EXEC'][i_trial][ts_date_all]
                    if s_action in l_trade_actions:
                        s_msgs = s_aux.split('msgs_to_env = ')[1]
                        for d_aux in json.loads(s_msgs.replace("'", '"')):
                            i_mult = 1 if d_aux['S'] == 'Buy' else -1
                            d_aux2[d_aux['C']] += float(d_aux['P']) * i_mult

                if ', pnl = ' in s_aux:
                    s_action = s_aux.split(', pnl = ')[1].split(',')[0]
                    f_aux = float(s_action)
                    d_rtn['pnl'][i_trial][ts_date_all] = f_aux
                if 'crossed_prices' in s_aux or 'correction_by_trade' in s_aux:
                    i_trades += 1
                if ', inputs = ' in s_aux:
                    s_mid = s_aux.split(', inputs = ')[1].split("{'midPrice':")
                    s_mid = s_mid[0].split(', position =')[0]
                    s_mid = s_mid.replace("'", '"').replace('None', '0')
                    l_mid = json.loads(s_mid)
                    s_mid = s_mid.replace("'", "")[1:]
                    l_mid = [(s_key, (float(x))) for s_key, x
                             in l_mid['midPrice'].iteritems()]
                    d_rtn['mid'][i_trial][ts_date_all] = dict(l_mid)
                    if s_mid[0] != '{':
                        s_mid = '{' + s_mid
                    d_input = json.loads(s_mid)
                    d_aux = d_input['TOB']['DI1F19']
                    d_rtn['TOB_F19'][i_trial][ts_date_all] = d_aux
                    d_aux = d_input['TOB']['DI1F21']
                    d_rtn['TOB_F21'][i_trial][ts_date_all] = d_aux
                    d_aux = dict(zip(['BID', 'ASK'], d_input['last_spread']))
                    d_rtn['LAST_SPREAD'][i_trial][ts_date_all] = d_aux
                    d_aux = dict(zip(['BID', 'ASK'],
                                     [d_input['agentOrders']['agentBid'],
                                     d_input['agentOrders']['agentAsk']]))
                    d_rtn['MY_PRICES'][i_trial][ts_date_all] = d_aux
            # finish the loop as soon as the trial is analyzed
            if i_trial > i_desired_trial:
                break
    return d_rtn, i_trades


def plot_trial(d_data, i_trades):
    '''
    Plots the data from logged metrics during a specific trial of a simulation.
    It is designed to plot trades using D1F21, F19 and F23.

    :param d_data: dict. data with the metrics used
    :param i_trades: integer. number of trades in the simulation
    '''
    fig = plt.figure(figsize=(12, 10))
    s_key = d_data['mid'].keys()[0]
    majorFormatter = mpl.dates.DateFormatter('%H:%M')

    ###############
    # Spread plot
    ###############
    df_spread = pd.DataFrame(d_data['mid'][s_key]).T
    df_spread = df_spread.resample('1min').last()

    ax = plt.subplot2grid((6, 6), (4, 0), colspan=2, rowspan=2)
    ((df_spread['DI1F23'] - df_spread['DI1F21'])*10**2).plot(ax=ax)
    ax.set_title('F23 - F21')
    ax.set_ylabel('Spread')
    # ax.xaxis.set_major_formatter(majorFormatter)

    ax = plt.subplot2grid((6, 6), (4, 2), colspan=2, rowspan=2)
    ((df_spread['DI1F21'] - df_spread['DI1F19'])*10**2).plot(ax=ax)
    ax.set_title('F21 - F19')
    # ax.xaxis.set_major_formatter(majorFormatter)

    ###############
    # PnL plot
    ###############
    ax = plt.subplot2grid((6, 6), (0, 0), colspan=3, rowspan=4)
    df_pnl = pd.Series(d_data['pnl'][s_key])
    df_pnl = df_pnl.resample('1min').last()
    df_pnl.plot(ax=ax)
    ax.axhline(xmin=0, xmax=1, y=0, color='black', linestyle='dashed')
    ax.set_title('PnL Curve')
    ax.set_ylabel('Value')
    # ax.xaxis.set_major_formatter(majorFormatter)

    ###############
    # Position plot
    ###############

    ax1 = plt.subplot2grid((6, 6), (0, 3), colspan=3, rowspan=2)
    df_pos = pd.DataFrame(d_data['position'][s_key]).T
    df_pos = df_pos.resample('1min').last()
    df_pos.plot(ax=ax1)
    ax1.set_title('Position')
    ax1.set_ylabel('Qty')
    # ax1.xaxis.set_major_formatter(majorFormatter)

    ###############
    # Duration plot
    ###############
    ax2 = plt.subplot2grid((6, 6), (2, 3), colspan=3, rowspan=2)  # sharex=ax1
    ax2.set_title('Duration Exposure')
    ax2.set_ylabel('Duration')
    df_duration = pd.Series(d_data['duration'][s_key])
    df_duration = df_duration.resample('1min').last()
    df_duration.plot(ax=ax2)
    # ax2.xaxis.set_major_formatter(majorFormatter)
    # set the last acis as visible
    # plt.setp(ax1.get_xticklabels(), visible=False)
    # plt.setp(ax2.get_xticklabels(), visible=True)

    ###############
    # Write success rate
    ###############

    s_duration_color = '#43a97f'
    if abs(df_duration.iloc[-1]) > 1.2:
        s_duration_color = '#d24445'

    s_pnl_color = '#43a97f'
    if df_pnl.iloc[-1] < 0.:
        s_pnl_color = '#d24445'

    l_cmm = ['DI1F19', u'DI1F21', 'DI1F23']
    l_aux = [int(df_pos.iloc[-1][s_cmm]) for s_cmm in l_cmm]

    ax = plt.subplot2grid((6, 6), (4, 4), colspan=2, rowspan=2)
    ax.axis('off')
    ax.text(0.40, .9, '{:0.0f} trades simulated.'.format(i_trades),
            fontsize=14, ha='center')
    ax.text(0.40, 0.75, 'Final PnL:', fontsize=16, ha='center')
    ax.text(0.40, 0.61, 'R$ {:0.1f}'.format(df_pnl.iloc[-1]), fontsize=20,
            ha='center', color=s_pnl_color)
    ax.text(0.40, 0.48, 'Final Duration:', fontsize=16, ha='center')
    ax.text(0.40, 0.34, '{:0.2f} years'.format(df_duration.iloc[-1]),
            fontsize=20, ha='center', color=s_duration_color)
    ax.text(0.40, 0.21, 'Final Position:', fontsize=16, ha='center')
    ax.text(0.40, 0, 'F19 {}, F21 {},\nF23 {}'.format(*l_aux),
            fontsize=15, ha='center', color='gray')

    plt.tight_layout()

    return fig


def plot_simulation(s_fname, s_log_to_filter):
    '''
    Plots the data from all trial of a simulation to help navigate between them

    :param s_fname: string. the name of the log of results
    :param s_log_to_filter: string. The log file name used by the simulation
    '''
    # prepare the file to be loaded as a json
    output = StringIO.StringIO()
    output.write('[')
    last_row = None
    for row in open(s_fname):
        if last_row:
            output.write(last_row[:-1] + ',')
        last_row = row
    output.write(last_row[:-1] + ']')
    json_data = json.loads(output.getvalue())
    # load and parse the file
    d_pnl = {}
    d_duration = {}
    d_ntrades = {}

    for row in json_data:
        if row['log_file'] == s_log_to_filter:
            d_pnl[row['trial']] = row['PnL']
            d_duration[row['trial']] = row['Duration']
            d_ntrades[row['trial']] = row['Total_Traded']['total']

    # plot the data
    fig, l_ax = plt.subplots(2, 2, figsize=(8, 6))
    l_ax = list(np.array(l_ax).ravel())
    l_color = sns.color_palette("Set2", 10)

    ###############
    # pnl plot
    ###############
    df_aux = pd.DataFrame(d_pnl)
    df_pnl = df_aux.loc[['last', 'max', 'min'], :]
    df_mdd = df_aux.loc[['MDD'], :]
    df_plot = df_pnl.unstack().reset_index()
    df_plot.columns = ['trial', 'measure', 'pnl']
    df_plot2 = df_plot.copy()
    sns.pointplot(x='trial', y='pnl', data=df_plot, aspect=.75, conf_lw=1,
                  capsize=0.3, estimator=func_estimator, ax=l_ax[0])
    f_avg = df_pnl.loc['last', :].mean()
    f_rows = df_plot[df_plot.measure == 'last'].shape[0] * 1.
    i_hits = ((df_plot[df_plot.measure == 'last'].pnl > 0).sum()/f_rows)*100
    s_txt = 'avg. pnl = {:0.2f} | hits: {:0.1f} %'
    l_ax[0].axhline(xmin=0, xmax=1, y=f_avg, color='black', linestyle='dashed',
                    label=s_txt.format(f_avg, i_hits))
    l_ax[0].set_title('Last/Min/Max PnL', fontsize=12)
    l_ax[0].set_ylabel('PnL')
    l_ax[0].legend(fontsize=8)

    ###############
    # Duration plot
    ###############
    df_aux = pd.DataFrame(d_duration)
    df_aux = df_aux.loc[['avg', 'max', 'min'], :]
    df_plot = df_aux.unstack().reset_index()
    df_plot.columns = ['trial', 'measure', 'duration']
    f_avg = df_aux.loc['avg', :].mean()
    sns.pointplot(x='trial', y='duration', data=df_plot, aspect=.75, conf_lw=1,
                  capsize=0.3, estimator=func_estimator, color=l_color[1],
                  ax=l_ax[3])
    l_ax[3].axhline(xmin=0, xmax=1, y=f_avg, color='black', linestyle='dashed',
                    label='avg. pnl = {:0.2f}'.format(f_avg))
    l_ax[3].set_title('Avg/Min/Max Duration', fontsize=12)
    l_ax[3].set_ylabel('years')
    l_ax[3].legend(fontsize=8)

    ###############
    # Number of operations plot
    ###############
    df_plot = df_mdd.unstack().reset_index()
    df_plot.columns = ['trial', 'measure', 'mdd']
    sns.barplot('trial', y='mdd', color=l_color[7], data=df_plot, ax=l_ax[2])
    l_ax[2].set_title('Maximum Drawdown', fontsize=12)
    l_ax[2].set_ylabel('Loss')

    ###############
    # Maximum Drawdown plot
    ###############
    df_plot = pd.DataFrame(pd.Series(d_ntrades))
    df_plot /= 5
    df_plot = df_plot.astype(int)
    df_plot = df_plot.unstack().reset_index()
    df_plot.columns = ['aux', 'trial', 'qty']
    sns.barplot('trial', y='qty', color=l_color[7], data=df_plot, ax=l_ax[1])
    l_ax[1].set_title('Number of Operations', fontsize=12)
    l_ax[1].set_ylabel('Qty')

    s_title = u'Agent: {}    |    File: {}'.format(json_data[0]['Agent'],
                                                   json_data[0]['log_file'])
    fig.suptitle(s_title, y=1.03, fontsize=15)
    fig.tight_layout()

    return df_plot2


def plot_learning_sim2(s_fname, s_log_to_filter, i_win=10, no_plot=False):
    '''
    Plots the data from logged metrics during a specific trial of a simulation.
    It is designed to plot trades using D1F21, F19 and F23. Return the data
    used in the plot

    :param s_fname: string. the name of the log of results
    :param s_log_to_filter: string. The log file name used by the simulation
    :param i_win*: integer. number of steps to average down
    '''
    # prepare the file to be loaded as a json
    output = StringIO.StringIO()
    output.write('[')
    last_row = None
    for row in open(s_fname):
        if last_row:
            output.write(last_row[:-1] + ',')
        last_row = row
    output.write(last_row[:-1] + ']')
    json_data = json.loads(output.getvalue())
    # load and parse the data from json
    d_rtn = {}
    d_count = {'Train': 0, 'Test': 0}

    for row in json_data:
        b_learning = 'parameters' in row
        if row['log_file'] == s_log_to_filter:
            d_count[row['parameters']['sim']] += 1

            i_id = row['trial']
            d_rtn[i_id] = {}

            d_rtn[i_id]['tot_traded'] = row['Trades']['DI1F21']['qBid']
            d_rtn[i_id]['tot_traded'] += row['Trades']['DI1F21']['qAsk']
            if 'DI1F19' in row['Trades']:
                d_rtn[i_id]['tot_traded'] += row['Trades']['DI1F19']['qBid']
                d_rtn[i_id]['tot_traded'] += row['Trades']['DI1F19']['qAsk']
            if 'DI1F23' in row['Trades']:
                d_rtn[i_id]['tot_traded'] += row['Trades']['DI1F23']['qBid']
                d_rtn[i_id]['tot_traded'] += row['Trades']['DI1F23']['qAsk']
            d_rtn[i_id]['avg_pnl'] = row['PnL']['last']
            # d_rtn[i_id]['avg_pnl'] /= d_rtn[i_id]['tot_traded']
            d_rtn[i_id]['avg_reward'] = row['Reward']
            d_rtn[i_id]['avg_reward'] /= row['parameters']['nsteps']
            d_rtn[i_id]['exploration_factor'] = row['parameters']['episilon']
            d_rtn[i_id]['learning_factor'] = row['parameters']['alpha']
            d_rtn[i_id]['simulation'] = row['parameters']['sim']
            d_rtn[i_id]['date'] = row['File_Date']
            d_rtn[i_id]['log_file'] = row['log_file']

            f_total = max(1., sum(row['parameters']['actions'].values()) * 1.)
            for s_key in ['null', 'BEST_BID', 'BEST_BOTH', 'BEST_OFFER']:
                s_key2 = s_key
                if s_key == 'null':
                    s_key2 = 'None'
                d_rtn[i_id][s_key2] = row['parameters']['actions'][s_key]
                d_rtn[i_id][s_key2] /= f_total

    df_rtn = pd.DataFrame(d_rtn).T
    if no_plot:
        return df_rtn
    df_train = df_rtn[df_rtn.simulation == 'Train']
    # df_train = df_rtn.copy()
    df_test = df_rtn[df_rtn.simulation == 'Test']

    # plot data
    plt.figure(figsize=(12, 8))

    ###############
    # ## Average step reward plot
    ###############

    ax = plt.subplot2grid((6, 6), (0, 3), colspan=3, rowspan=2)
    ax.set_title('{}-Trial Rolling Average Reward per Action'.format(i_win))
    ax.set_ylabel('Reward per Action')
    ax.set_xlabel('Trial Number')
    ax.set_xlim((i_win, df_train.shape[0]))

    # Create plot-specific data
    # ax.axhline(xmin=0, xmax=1, y=0, color='black', linestyle='dashed')
    ax.plot(df_train.avg_reward.rolling(window=i_win, center=False).mean(),
            label=False)

    ###############
    # ## Parameters Plot
    ###############

    ax = plt.subplot2grid((6, 6), (2, 3), colspan=3, rowspan=2)

    # Check whether the agent was expected to learn
    if b_learning:
        ax.set_ylabel('Parameter Value')
        ax.set_xlabel('Trial Number')
        ax.set_xlim((1, df_train.shape[0]))
        ax.set_ylim((0, 1.05))
        ax.plot(df_train[['exploration_factor']], label='Exploration factor')
        ax.plot(df_train[['learning_factor']], label='Learning factor')

        ax.legend(bbox_to_anchor=(0.5, 1.19), fancybox=True, ncol=2,
                  loc='upper center', fontsize=10)

    else:
        ax.axis('off')
        ax.text(0.52, 0.30, 'Simulation completed\nwith learning disabled.',
                fontsize=24, ha='center', style='italic')

    ###############
    # ## Actoins ratio Plot
    ###############

    ax = plt.subplot2grid((6, 6), (0, 0), colspan=3, rowspan=4)
    ax.set_title('Actions Ratio'.format(i_win))
    ax.set_ylabel('Ratio')
    ax.set_xlabel('Trial Number')

    ax.set_xlim((i_win, df_train.shape[0]))

    ax.plot(df_train[['None']].rolling(window=i_win, center=False).mean(),
            label='None', linestyle='dotted', linewidth=3)
    ax.plot(df_train[['BEST_BOTH']].rolling(window=i_win, center=False).mean(),
            label='Best Both', linestyle='dotted', linewidth=3)
    ax.plot(df_train[['BEST_BID']].rolling(window=i_win, center=False).mean(),
            label='Best Bid', linestyle='dotted', linewidth=2)
    s_key = 'BEST_OFFER'
    ax.plot(df_train[[s_key]].rolling(window=i_win, center=False).mean(),
            label='Best Offer', linestyle='dotted', linewidth=2)

    ax.legend(loc='upper right', fancybox=True, fontsize=10)

    ###############
    # ## Rolling PnL per Contract plot
    ###############

    ax = plt.subplot2grid((6, 6), (4, 0), colspan=4, rowspan=2)
    ax.set_title('{}-Trial Rolling PnL'.format(i_win))
    ax.set_ylabel('Average PnL')
    ax.set_xlabel('Trial Number')
    ax.set_xlim((i_win, df_train.shape[0]))

    # Rolling avg pnl
    ax.plot(df_train.avg_pnl.rolling(window=i_win, center=False).mean(),
            label=False)

    ###############
    # ## Test results
    ###############

    ax = plt.subplot2grid((6, 6), (4, 4), colspan=2, rowspan=2)
    ax.axis('off')

    if d_count['Test'] > 0:
        f_avg_pnl = df_test.avg_pnl[df_test.simulation == 'Test'].mean()
        f_avg_reward = df_test.avg_reward[df_test.simulation == 'Test'].mean()
        # Write success rate
        s_msg = '{} testing / {} training trials simulated.'
        ax.text(0.40, .9, s_msg.format(d_count['Test'], d_count['Train']),
                fontsize=14, ha='center')
        ax.text(0.40, 0.7, 'Average PnL in Test:', fontsize=16, ha='center')
        ax.text(0.40, 0.42, '{:0.2f}'.format(f_avg_pnl), fontsize=40,
                ha='center', color='black')
        ax.text(0.40, 0.27, 'Average Reward in Test:', fontsize=16,
                ha='center')
        ax.text(0.40, 0, '{:0.2f}'.format(f_avg_reward), fontsize=40,
                ha='center', color='black')

    else:
        ax.text(0.36, 0.30, 'Simulation completed\nwithout tests.',
                fontsize=20, ha='center', style='italic')

    plt.tight_layout()
    plt.show()

    return df_rtn


def plot_learning_sim(s_fname, s_log_to_filter, i_win=10):
    '''
    Plots the data from logged metrics during a specific trial of a simulation.
    It is designed to plot trades using D1F21, F19 and F23. Return the data
    used in the plot

    :param s_fname: string. the name of the log of results
    :param s_log_to_filter: string. The log file name used by the simulation
    :param i_win*: integer. number of steps to average down
    '''
    # prepare the file to be loaded as a json
    output = StringIO.StringIO()
    output.write('[')
    last_row = None
    for row in open(s_fname):
        if last_row:
            output.write(last_row[:-1] + ',')
        last_row = row
    output.write(last_row[:-1] + ']')
    json_data = json.loads(output.getvalue())
    # load and parse the data from json
    d_rtn = {}
    d_count = {'Train': 0, 'Test': 0}

    for row in json_data:
        b_learning = 'parameters' in row
        if row['log_file'] == s_log_to_filter:
            d_count[row['parameters']['sim']] += 1

            i_id = row['trial']
            d_rtn[i_id] = {}

            d_rtn[i_id]['tot_traded'] = row['Trades']['DI1F19']['qBid']
            d_rtn[i_id]['tot_traded'] += row['Trades']['DI1F19']['qAsk']
            d_rtn[i_id]['tot_traded'] += row['Trades']['DI1F21']['qBid']
            d_rtn[i_id]['tot_traded'] += row['Trades']['DI1F19']['qAsk']
            d_rtn[i_id]['avg_pnl'] = row['PnL']['last']
            # d_rtn[i_id]['avg_pnl'] /= d_rtn[i_id]['tot_traded']
            d_rtn[i_id]['avg_reward'] = row['Reward']
            d_rtn[i_id]['avg_reward'] /= row['parameters']['nsteps']
            d_rtn[i_id]['exploration_factor'] = row['parameters']['episilon']
            d_rtn[i_id]['learning_factor'] = row['parameters']['alpha']
            d_rtn[i_id]['simulation'] = row['parameters']['sim']

            for s_key in ['0', '1', '2']:
                # f_precision = 0.
                # if row['parameters']['PR'][s_key]['a'] != 0.:
                #     f_precision = row['parameters']['PR'][s_key]['a_m']
                #     f_precision /= (row['parameters']['PR'][s_key]['a'])
                # f_recall = 0.
                # if row['parameters']['PR'][s_key]['m'] != 0.:
                #     f_recall = row['parameters']['PR'][s_key]['a_m']
                #     f_recall /= (row['parameters']['PR'][s_key]['m'])
                # f_f1 = 0.
                # if (f_precision + f_recall) != 0:
                #     f_f1 = (2 * f_precision * f_recall)
                #     f_f1 /= (f_precision + f_recall)
                # d_rtn[i_id]['f1_{}'.format(s_key)] = f_f1
                d_rtn[i_id]['f1_{}'.format(s_key)] = 0.

    df_rtn = pd.DataFrame(d_rtn).T
    df_train = df_rtn[df_rtn.simulation == 'Train']
    # df_train = df_rtn.copy()
    df_test = df_rtn[df_rtn.simulation == 'Test']

    # plot data
    plt.figure(figsize=(12, 8))

    ###############
    # ## Average step reward plot
    ###############

    ax = plt.subplot2grid((6, 6), (0, 3), colspan=3, rowspan=2)
    ax.set_title('{}-Trial Rolling Average Reward per Action'.format(i_win))
    ax.set_ylabel('Reward per Action')
    ax.set_xlabel('Trial Number')
    ax.set_xlim((i_win, df_train.shape[0]))

    # Create plot-specific data
    # ax.axhline(xmin=0, xmax=1, y=0, color='black', linestyle='dashed')
    ax.plot(df_train.avg_reward.rolling(window=i_win, center=False).mean(),
            label=False)

    ###############
    # ## Parameters Plot
    ###############

    ax = plt.subplot2grid((6, 6), (2, 3), colspan=3, rowspan=2)

    # Check whether the agent was expected to learn
    if b_learning:
        ax.set_ylabel('Parameter Value')
        ax.set_xlabel('Trial Number')
        ax.set_xlim((1, df_train.shape[0]))
        ax.set_ylim((0, 1.05))
        ax.plot(df_train[['exploration_factor']], label='Exploration factor')
        ax.plot(df_train[['learning_factor']], label='Learning factor')

        ax.legend(bbox_to_anchor=(0.5, 1.19), fancybox=True, ncol=2,
                  loc='upper center', fontsize=10)

    else:
        ax.axis('off')
        ax.text(0.52, 0.30, 'Simulation completed\nwith learning disabled.',
                fontsize=24, ha='center', style='italic')

    ###############
    # ## F1 Score Plot
    ###############

    ax = plt.subplot2grid((6, 6), (0, 0), colspan=3, rowspan=4)
    ax.set_title('{}-Trial Rolling F1 Score'.format(i_win))
    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Trial Number')

    ax.set_xlim((i_win, df_train.shape[0]))

    ax.plot(df_train[['f1_0']].rolling(window=i_win, center=False).mean(),
            label='Caiu', linestyle='dotted', linewidth=3)
    ax.plot(df_train[['f1_1']].rolling(window=i_win, center=False).mean(),
            label='Ficou Parado', linestyle='dotted', linewidth=3)
    ax.plot(df_train[['f1_2']].rolling(window=i_win, center=False).mean(),
            label='Subiu', linestyle='dotted', linewidth=2)

    ax.legend(loc='upper right', fancybox=True, fontsize=10)

    ###############
    # ## Rolling PnL per Contract plot
    ###############

    ax = plt.subplot2grid((6, 6), (4, 0), colspan=4, rowspan=2)
    ax.set_title('{}-Trial Rolling PnL'.format(i_win))
    ax.set_ylabel('Average PnL')
    ax.set_xlabel('Trial Number')
    ax.set_xlim((i_win, df_train.shape[0]))

    # Rolling avg pnl
    ax.plot(df_train.avg_pnl.rolling(window=i_win, center=False).mean(),
            label=False)

    ###############
    # ## Test results
    ###############

    ax = plt.subplot2grid((6, 6), (4, 4), colspan=2, rowspan=2)
    ax.axis('off')

    if d_count['Test'] > 0:
        f_avg_pnl = df_test.avg_pnl[df_test.simulation == 'Test'].mean()
        f_avg_reward = df_test.avg_reward[df_test.simulation == 'Test'].mean()
        # Write success rate
        s_msg = '{} testing / {} training trials simulated.'
        ax.text(0.40, .9, s_msg.format(d_count['Test'], d_count['Train']),
                fontsize=14, ha='center')
        ax.text(0.40, 0.7, 'Average PnL in Test:', fontsize=16, ha='center')
        ax.text(0.40, 0.42, '{:0.2f}'.format(f_avg_pnl), fontsize=40,
                ha='center', color='black')
        ax.text(0.40, 0.27, 'Average Reward in Test:', fontsize=16,
                ha='center')
        ax.text(0.40, 0, '{:0.2f}'.format(f_avg_reward), fontsize=40,
                ha='center', color='black')

    else:
        ax.text(0.36, 0.30, 'Simulation completed\nwithout tests.',
                fontsize=20, ha='center', style='italic')

    plt.tight_layout()
    plt.show()

    return df_rtn
