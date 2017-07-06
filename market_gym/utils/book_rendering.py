#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
test library to render

@author: ucaiado

Created on 07/04/2017
"""

import matplotlib.pyplot as plt
import numpy as np

text_style = dict(horizontalalignment='right', verticalalignment='center',
                  fontsize=14, fontdict={'family': 'sans-serif'})

header_style = dict(horizontalalignment='right', verticalalignment='center',
                    fontsize=15, fontdict={'family': 'sans-serif'},
                    color='black', weight='bold')

time_style = dict(horizontalalignment='center', verticalalignment='center',
                  fontsize=17, fontdict={'family': 'sans-serif'},
                  color='black', weight='bold')
info_style = dict(horizontalalignment='left', verticalalignment='center',
                  fontsize=13, fontdict={'family': 'sans-serif'})


def img_init(fig, e):
    '''
    '''
    # initialize the subplots in the figure
    i_mcol = min(6, len(e.l_instrument) * 2)
    i_mrow = int(np.ceil(len(e.l_instrument) * 2 / 6.))

    ax_time = plt.subplot2grid((i_mrow*2+1, i_mcol), (0, 0), colspan=i_mcol-2,
                               rowspan=1)
    ax_time.axis('off')
    t0 = ax_time.text(.5, 0.5, e.order_matching.s_time, **time_style)

    ax_summary = plt.subplot2grid((i_mrow*2+1, i_mcol), (0, i_mcol-2),
                                  colspan=2, rowspan=1)
    f_pnl = 0.
    f_pos = 0.
    f_delta = 0.
    f_tot = 0
    ax_summary.axis('off')
    t1 = ax_summary.text(0.2, 0.96, 'PnL: R$ {:.01f}'.format(f_pnl),
                         **info_style)
    t2 = ax_summary.text(0.2, 0.64, 'Pos on main: {:.0f}'.format(f_pos),
                         **info_style)
    t3 = ax_summary.text(0.2, 0.32, 'PnL on Pos: {:.01f}'.format(f_delta),
                         **info_style)
    t4 = ax_summary.text(0.2, 0., 'Total traded: {:.0f}'.format(f_tot),
                         **info_style)

    l_ax = []

    i_charts = len(e.l_instrument)
    i_count = 0
    for i_row in xrange(1, i_mrow*2+1, 2):
        for i_col in xrange(0, i_mcol, 2):
            i_count += 1
            l_ax.append(plt.subplot2grid((i_mrow*2+1, i_mcol), (i_row, i_col),
                                         colspan=2, rowspan=2))
            if i_count >= i_charts:
                break
    d_ax = dict(zip(e.l_instrument, l_ax))
    # fill each subplot with a book
    d_obj = {}
    d_obj['time'] = {'ax': ax_time, 'txt': t0}
    d_obj['summary'] = {'ax': ax_summary, 'pnl': t1, 'pos': t2, 'pos_pnl': t3,
                        'tot': t4}
    for s_cmm in e.l_instrument:
        d_obj[s_cmm] = {'ax': d_ax[s_cmm]}
        ax = d_ax[s_cmm]
        df_book = e.get_order_book(s_cmm, True)

        ax.axis('off')
        ax.set_title(s_cmm + '\n', fontsize=16)
        i_row, i_col = df_book.shape
        l_txt_format = ['{:0.0f}', '{:0.02f}', '{:0.02f}', '{:0.0f}']

        ax.text(0.1, 1.00, 'qBid', **header_style)
        ax.text(0.35, 1.00, 'Bid', **header_style)
        ax.text(0.6, 1.00, 'Ask', **header_style)
        ax.text(0.85, 1.00, 'qAsk', **header_style)

        d_obj[s_cmm] = {'ax': d_ax[s_cmm],
                        'qBid': {1: None, 2: None, 3: None, 4: None, 5: None},
                        'Bid': {1: None, 2: None, 3: None, 4: None, 5: None},
                        'Ask': {1: None, 2: None, 3: None, 4: None, 5: None},
                        'qAsk': {1: None, 2: None, 3: None, 4: None, 5: None}}
        for i in xrange(i_row):

            # qBid
            s_txt = '{:0,.0f}'.format(df_book.iloc[i, 0].i_qty)
            d_obj[s_cmm]['qBid'][i] = ax.text(0.1, 0.80 - i * 0.22, s_txt,
                                              **text_style)

            # Bid / Ask
            s_txt1 = '{:0.02f}'.format(df_book.iloc[i, 1])
            s_txt2 = '{:0.02f}'.format(df_book.iloc[i, 2])
            d_obj[s_cmm]['Bid'][i] = ax.text(0.35, 0.80 - i * 0.22, s_txt1,
                                             **text_style)
            d_obj[s_cmm]['Ask'][i] = ax.text(0.6, 0.80 - i * 0.22, s_txt2,
                                             **text_style)

            # qAsk
            s_txt = '{:0,.0f}'.format(df_book.iloc[i, 3].i_qty)
            d_obj[s_cmm]['qAsk'][i] = ax.text(0.85, 0.80 - i * 0.22, s_txt,
                                              **text_style)

    # fig.tight_layout()
    fig.set_tight_layout(True)
    fig.subplots_adjust(bottom=0.01)
    return d_obj


def img_update(d_obj, e, a):
    '''
    '''
    s_main_intr = e.s_main_intrument
    try:
        f_pnl = e.agent_states[a]['Pnl']
        f_pos = e.agent_states[a][s_main_intr]['Position']
        f_tot = e.agent_states[a][s_main_intr]['qAsk']
        f_tot += e.agent_states[a][s_main_intr]['qBid']
        f_delta = a.f_delta_pnl
        l_agent_prices = [x for x in
                          a.d_order_tree[s_main_intr]['ASK'].keys()]
        l_agent_prices += [x for x in
                           a.d_order_tree[s_main_intr]['BID'].keys()]
    except (KeyError, AttributeError) as e:
        f_pnl = 0.
        f_pos = 0.
        f_tot = 0.
        f_tot += 0.
        f_delta = 0.
        l_agent_prices = []
    d_obj['time']['txt'].set_text(e.order_matching.s_time)
    d_obj['summary']['pnl'].set_text('PnL: R$ {:.01f}'.format(f_pnl))
    d_obj['summary']['pos'].set_text('Pos on main: {:.0f}'.format(f_pos))
    d_obj['summary']['pos_pnl'].set_text('PnL on Pos: {:.01f}'.format(f_delta))
    d_obj['summary']['tot'].set_text('Total traded: {:.0f}'.format(f_tot))

    for s_cmm in e.l_instrument:
        df_book = e.get_order_book(s_cmm, True)
        for i in xrange(5):
            # qBid
            s_txt = '{:0,.0f}'.format(df_book.iloc[i, 0].i_qty)
            d_obj[s_cmm]['qBid'][i].set_text(s_txt)
            # Bid
            s_txt1 = '{:0.02f}'.format(df_book.iloc[i, 1])
            d_obj[s_cmm]['Bid'][i].set_text(s_txt1)
            if s_cmm == s_main_intr and df_book.iloc[i, 1] in l_agent_prices:
                d_obj[s_cmm]['Bid'][i].set_bbox(dict(facecolor='gray',
                                                     alpha=0.5,
                                                     edgecolor='none'))
            else:
                d_obj[s_cmm]['Bid'][i].set_bbox(None)
            # Ask
            s_txt2 = '{:0.02f}'.format(df_book.iloc[i, 2])
            d_obj[s_cmm]['Ask'][i].set_text(s_txt2)
            if s_cmm == s_main_intr and df_book.iloc[i, 2] in l_agent_prices:
                d_obj[s_cmm]['Ask'][i].set_bbox(dict(facecolor='gray',
                                                     alpha=0.5,
                                                     edgecolor='none'))
            else:
                d_obj[s_cmm]['Ask'][i].set_bbox(None)

            # qAsk
            s_txt = '{:0,.0f}'.format(df_book.iloc[i, 3].i_qty)
            d_obj[s_cmm]['qAsk'][i].set_text(s_txt)
