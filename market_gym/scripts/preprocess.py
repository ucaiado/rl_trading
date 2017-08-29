#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Preprocess the dataset to make it easy to use in simulation

@author: ucaiado

Created on 10/24/2016
"""

import argparse
import textwrap
from collections import defaultdict
import sys
import numpy as np
import os
import pandas as pd
import time
import zipfile

from market_gym.lob import translator, matching_engine, book, parser_data
import book_cleaner as book_cleaner

'''
Begin help functions
'''


def renamefiles(s_date):
    '''
    rename old files to be processed

    :param s_date: string. date of the files
    '''
    s_root = 'data/temp/{}'.format(s_date)
    for s_fname in os.listdir(s_root):
        s_filename = s_root + '/' + s_fname
        if '_new2' in s_filename:
            os.rename(s_filename, s_filename.replace('_new2', '_new'))


'''
End help functions
'''


def process_files(s_date, l_instruments, last_check=True, s_filetype='BMF'):
    '''
    It is a ugly function to clean files fom BvmfFTP. Should be refactor

    :param s_date: string. date of the file usually in YYYYMMDD format
    :param l_instruments: list. Instruments to extract data
    '''
    sys.path.append('../../')
    if s_filetype == 'BMF':
        s_fname_bid = 'data/original/OFER_CPA_BMF_{}.zip'.format(s_date)
        s_fname_ask = 'data/original/OFER_VDA_BMF_{}.zip'.format(s_date)
    elif s_filetype == 'VISTA':
        s_fname_bid = 'data/original/OFER_CPA_{}.zip'.format(s_date)
        s_fname_ask = 'data/original/OFER_VDA_{}.zip'.format(s_date)
    elif s_filetype == 'OPCOES':
        s_fname_bid = 'data/original/OFER_CPA_OPCOES_{}.zip'.format(s_date)
        s_fname_ask = 'data/original/OFER_VDA_OPCOES_{}.zip'.format(s_date)
    else:
        raise NotImplemented('file type {} is not valid'.format(s_filetype))

    # create the folder to hold the data
    s_ddir = 'data/temp/{}'.format(s_date)
    if not os.path.isdir(s_ddir):
        os.makedirs(s_ddir)

    # analyse the files inside each zip file
    f_tot_all = 0
    l_original_files = [s_fname_ask, s_fname_bid]  # , s_fname_trades]
    for s_fname in l_original_files:
        archive = zipfile.ZipFile(s_fname, 'r')
        l_files = archive.infolist()
        print 'There are {} files inside the {}'.format(len(l_files), s_fname)
        f_tot = sum([x.file_size/(1024.**2) for x in l_files])
        f_tot_all += f_tot
        print 'The total size of the files is {:0.1f} MB\n'.format(f_tot)
        archive.close()
    s_err = '=======================\nTOTAL SIZE: {:0.1f} MB'
    print s_err.format(f_tot_all)

    # extract a list of instruments
    # l_instruments = ['DI1F19', 'DI1F21', 'DI1F23', 'DI1F25']
    print '\nExtract data fom Ask side:'
    parser_data.extract_data(s_date, s_fname_ask, 'ASK', l_instr=l_instruments)
    print '\n\nExtract data fom Bid side:'
    parser_data.extract_data(s_date, s_fname_bid, 'BID', l_instr=l_instruments)

    # iterate files
    cleaning_files(s_date, l_instruments, last_check=last_check)


def cleaning_files(s_date, l_instruments, s_type='old', last_check=True):
    '''
    '''
    # crindo horarios de checagem
    l_hours = [8, 10, 11, 12, 13, 14, 15, 16]
    my_stop_time_gen = matching_engine.get_next_stoptime(l_hours, i_sec=10)

    # collect current scenarious
    f_tot_time = 0
    f_start_time = time.time()

    # open bid
    l_order_books = []

    s_fbid = 'data/temp/{}/{}_BID_{}.zip'
    s_fask = 'data/temp/{}/{}_ASK_{}.zip'
    s_last_fname = 'data/temp/{}/{}_{}_{}.zip'
    s_append_to_fname = '_new.zip'
    if s_type == 'new':
        s_fbid = 'data/temp/{}/{}_BID_{}_new.zip'
        s_fask = 'data/temp/{}/{}_ASK_{}_new.zip'
        s_last_fname = 'data/temp/{}/{}_{}_{}_new.zip'
        s_append_to_fname = '_new2.zip'

    print '\nfirst check of the books:'
    for s_name in l_instruments:
        s_fbid_aux = s_fbid.format(s_date, s_date, s_name)
        s_fask_aux = s_fask.format(s_date, s_date, s_name)
        l_order_books.append(book_cleaner.LimitOrderBook(s_fbid_aux,
                                                         s_fask_aux,
                                                         'a',
                                                         b_mount_blklist=True))

    d_scenario = defaultdict(lambda: defaultdict(pd.DataFrame))
    for s_stoptime in my_stop_time_gen:
        for s_name, book_obj in zip(l_instruments, l_order_books):
            # set the stop time
            book_obj.set_stop_time(s_stoptime)
            for b_test in book_obj:
                pass
            # acumulate data
            if float(s_stoptime.split(":")[-1]) % 10 == 0:
                # filter data form this book
                df_rtn = (book_obj.get_n_best_prices(10)).iloc[:10, :].copy()
                # freeze qtyties
                df_rtn.qBid = [str(x) for x in df_rtn.qBid]
                df_rtn.qAsk = [str(x) for x in df_rtn.qAsk]
                d_scenario[s_name][s_stoptime] = df_rtn

    # time
    f_time = time.time() - f_start_time
    f_tot_time += f_time
    s_err = '\n======================\TIME TO READ BOOKS (1): {:0.2f}'
    print s_err.format(f_tot_time)

    print '\n----> CHECKING BY TRADE\n'
    for book_obj in l_order_books:
        print book_obj.d_ask['instrumento_symbol']
        for s_side, d_aux in zip(['ASK', 'BID'],
                                 [book_obj.d_wrongask,
                                 book_obj.d_wrongbid]):
            print s_side
            df_status = pd.DataFrame(d_aux).T.fillna(0)
            s_err = 'numero de ids supostamente invalidas: {}'
            print s_err.format(df_status.shape[0])
            print df_status.head()
            print

    print '\n----> CHECKING WHEN PRICES CROSSED EACH OTHER\n'
    for book_obj in l_order_books:
        print book_obj.d_ask['instrumento_symbol']
        for s_side, d_aux in zip(['ASK', 'BID'],
                                 [book_obj.d_warningask,
                                 book_obj.d_warningbid]):
            print s_side
            df_status = pd.DataFrame(d_aux).T.fillna(0)
            s_err = 'numero de ids supostamente invalidas: {}'
            print s_err.format(df_status.shape[0])
            print df_status.head()
            print

        print

    # read the books processed
    i_test = 0
    i_total = 0
    f_start_time = time.time()

    d_files = {}
    for s_name, book_obj in zip(l_instruments, l_order_books):
        for s_side, d_invalid, d_warning in zip(['ASK', 'BID'],
                                                [book_obj.d_wrongask,
                                                book_obj.d_wrongbid],
                                                [book_obj.d_warningask,
                                                book_obj.d_warningbid]):
            s_file = 'data/temp/{}/{}.txt'
            d_files[s_name] = open(s_file.format(s_date, s_name), 'w')
            myParser = parser_data.LineParser(s_side)
            s_fname = s_last_fname.format(s_date, s_date, s_side, s_name)
            with zipfile.ZipFile(s_fname, 'r') as archive:
                fr = archive.open(archive.infolist()[0])
                for idx, row in enumerate(fr):
                    i_total += 1
                    # parse the line
                    d_aux = myParser(row)
                    b_print = True
                    if idx > 1:
                        if d_aux['seq_order_number'] in d_invalid.keys():
                            b_print = False
                            i_test += 1
                        elif d_aux['seq_order_number'] in d_warning.keys():
                            b_print = False
                            i_test += 1
                    # print valid rows
                    if b_print:
                        d_files[s_name].write(row)
            d_files[s_name].close()
            # create new zip file
            s_fname2 = s_fname.split('.zip')[0].replace('_new', '')
            s_fname2 += s_append_to_fname
            zf = zipfile.ZipFile(s_fname2,
                                 mode='w',
                                 compression=zipfile.ZIP_DEFLATED,
                                 )
            zf.write(s_file.format(s_date, s_name))
            # close the files and clean the memory
            zf.close()
            os.remove(s_file.format(s_date, s_name))
            os.remove(s_fname)

    s_err = 'Took {:0.2f} seconds to loop {:0,.0f} lines in the file'
    print s_err.format(time.time() - f_start_time, i_total)
    s_err = 'We are going to not use {:0,.0f} rows'
    print s_err.format(i_test)
    if i_test > 2000:
        if not last_check:
            return
    # recrindo horarios de checagem
    l_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16]
    my_stop_time_gen = matching_engine.get_next_stoptime(l_hours, i_sec=10)

    f_tot_time = 0
    f_start_time = time.time()

    # open bid
    l_order_books = []

    for s_name in l_instruments:
        s_fbid = 'data/temp/{}/{}_BID_{}{}'.format(s_date,
                                                   s_date,
                                                   s_name,
                                                   s_append_to_fname)
        s_fask = 'data/temp/{}/{}_ASK_{}{}'.format(s_date,
                                                   s_date,
                                                   s_name,
                                                   s_append_to_fname)
        l_order_books.append(book.LimitOrderBook(s_fbid, s_fask, 'a'))

    d_scenario = defaultdict(lambda: defaultdict(pd.DataFrame))
    for s_stoptime in my_stop_time_gen:
        for s_name, book_obj in zip(l_instruments, l_order_books):
            # set the stop time
            book_obj.set_stop_time(s_stoptime)
            for b_test in book_obj:
                pass
            # filter data form this book

            f_err = 0.08
            f_rows = 50
            # esta rotina é lenta
            # df_rtn = (book_obj.get_n_best_prices(10)).iloc[:10, :].copy()
            df_rtn = book_obj.get_n_top_prices(10, True).copy()
            # freeze qtyties
            df_rtn.qBid = [str(x) for x in df_rtn.qBid]
            df_rtn.qAsk = [str(x) for x in df_rtn.qAsk]
            # acumulate data
            d_scenario[s_name][s_stoptime] = df_rtn

    # time
    f_time = time.time() - f_start_time
    f_tot_time += f_time
    print '\n======================\nTOTAL: {:0.2f} seconds'.format(f_tot_time)

    s_err = 'Foram coletados {} books do arquivo por ativo\n'
    print s_err.format(len(d_scenario['DI1F21']))
    l_hours = []
    for s_key in l_instruments:
        print s_key
        i_crossed = 0
        i_tot = 0
        for s_key2 in d_scenario[s_key]:
            try:
                df_test = d_scenario[s_key][s_key2].iloc[0, :]
                i_crossed += (df_test['Bid'] > df_test['Ask'])*1
                if df_test['Bid'] > df_test['Ask']:
                    l_hours.append(s_key2)
            except:
                pass
        s_err = u'!! Em {} dos books coletados os preços cruzaram'
        print s_err.format(i_crossed)
        s_to_print = np.random.choice(d_scenario[s_key].keys())
        print d_scenario[s_key][s_to_print]
        print
    print u'Horários com problemas'
    print np.sort(l_hours)
    print u'\nExemplo:'
    if len(l_hours) == 0:
        print u'Não há exemplos para serem plotados'
    else:
        print d_scenario[s_key][np.random.choice(l_hours)]

if __name__ == '__main__':
    s_txt = '''\
            File Preprocessing
            --------------------------------
            Clean files from FTP. The original files should be put in the
            "data/original/" folder, located in the top-level project directory
            '''
    obj_formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=obj_formatter,
                                     description=textwrap.dedent(s_txt))
    parser.add_argument('date', default=None, type=str, metavar='',
                        help='date of the file to be used (formart AAAAMMDD)')
    s_help = 'If it is a additional cleaning run of the initial file'
    parser.add_argument('-sr', '--secrun', action='store_false',
                        help=s_help)
    s_help = 'type of the file to be processed. Just used if it is the first'
    s_help += ' run. It can be "BMF", "VISTA" or "OPCOES".'
    parser.add_argument('-ft', '--filetype', default=None, type=str,
                        help=s_help)
    # s_help = 'If should not check for crossing prices'
    # parser.add_argument('-nc', '--nocheck', action='store_false',
    #                     help=s_help)
    # recover arguments
    args = parser.parse_args()
    s_date = args.date
    b_old = args.secrun
    s_ftype = args.filetype
    if not s_ftype:
        s_ftype = 'BMF'
    if s_ftype == 'BMF':
        l_instruments = ['DI1F18', 'DI1N18', 'DI1F19', 'DI1N19', 'DI1F20',
                         'DI1N20', 'DI1F21', 'DI1F23', 'DI1F25', 'DI1F27']
    elif s_ftype == 'VISTA':
        l_instruments = ['PETR4', 'VALE5']
    elif s_ftype == 'OPCOES':
        l_instruments = ['PETRQ34', 'PETRE34', 'PETRE10', 'PETRE18', 'PETRE17',
                         'PETRE46', 'PETRE16', 'PETRQ15', 'PETRE15', 'PETRE45',
                         'PETRE44', 'PETRE43', 'PETRQ43', 'PETRE47', 'PETRQ13',
                         'PETRE13', 'PETRE12', 'PETRE63']
    # b_lcheck = args.nocheck
    # run the script
    if b_old:
        process_files(s_date, l_instruments, last_check=False,
                      s_filetype=s_ftype)
    else:
        renamefiles(s_date)
        cleaning_files(s_date,  l_instruments, s_type='new', last_check=False)
        renamefiles(s_date)
