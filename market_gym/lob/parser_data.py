#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Library to parse the market data file form BmfBovespa

@author: ucaiado

Created on 06/10/2016
"""
# import libraries
import os
import StringIO
import zipfile
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import subprocess

'''
Begin help functions
'''


class NoMappedException(Exception):
    """
    NoMappedException is raised by the _call_() method in the LineParser class
    to indicate that the row length does not match any layout used
    """
    pass


class NoValidTypeException(Exception):
    """
    NoValidTypeException is raised by the _init_() method in the LineParser
    class to indicate that the s_type passed to the constructor is not acepted
    """
    pass

'''
End help functions
'''


def extract_data_old(s_fname, s_type, l_instr=[]):
    '''
    Extract the data from the file related to the instruments desired and
    create a zip file to each one. Also create a TXT with the instruments
    presented in the s_fname file. It is not expected that a given instrument
    has too much data to fit in memory

    :param s_fname: string. Path to the file to parse
    :param s_type: string. type of the file {BID, ASK, NEG}
    :param l_instr*: list. List of strings with the instruments to be extracted
    '''

    # loop all the file
    set_instruments = set()
    # create a list of files to save the information
    d_files = {}
    for s_instr in l_instr:
        d_files[s_instr] = StringIO.StringIO()

    myParser = LineParser(s_type)
    with zipfile.ZipFile(s_fname, 'r') as archive:
        fr = archive.open(archive.infolist()[0])
        for idx, row in enumerate(fr):
            d_aux = myParser(row)
            if myParser.last_identification == 'MSG':
                s_instr = d_aux['instrumento_symbol']
                set_instruments.add(s_instr)
                if d_aux['instrumento_symbol'] in l_instr:
                    d_files[s_instr].write(row)
            elif myParser.last_identification == 'RH':
                for s_intr in l_instr:
                    d_files[s_intr].write(row)
            elif myParser.last_identification == 'RT':
                for s_intr in l_instr:
                    d_files[s_intr].write(row)
        fr.close()

    # save a list of the instruments in the file
    s_file_instr = 'data/temp/{:%Y%m%d}_{}_INSTR.txt'
    s_file_instr = s_file_instr.format(d_aux['end_date'], myParser.s_file_type)
    df_instr = pd.Series(list(set_instruments))
    df_instr.sort_values(inplace=True)
    df_instr.to_csv(s_file_instr, sep='\t', index=False)

    # criate zipfiles
    i_count = 1
    for s_instr in l_instr:
        s_file = 'data/temp/{:%Y%m%d}_{}_{}.zip'
        s_file = s_file.format(d_aux['end_date'],
                               myParser.s_file_type,
                               s_instr)
        zf = zipfile.ZipFile(s_file,
                             mode='w',
                             compression=zipfile.ZIP_DEFLATED,
                             )
        zf.writestr('data.txt', d_files[s_instr].getvalue())
        # close the files and clean the memory
        zf.close()
        d_files[s_instr].close()
        i_count += 1
    # print succcess message
    print '{} files created'.format(i_count)


def extract_data(s_date, s_fname, s_type, l_instr=[], b_sort=False):
    '''
    Extract the data from the file related to the instruments desired and
    create a zip file to each one. Also create a TXT with the instruments
    presented in the s_fname file. It is not expected that a given instrument
    has too much data to fit in memory

    :param s_date: string. Date
    :param s_fname: string. Path to the file to parse
    :param s_type: string. type of the file {BID, ASK, NEG}
    :param l_instr*: list. List of strings with the instruments to be extracted
    :param b_sort*: boolean. sort the files created using sort toll from Linux
    '''

    # loop all the file
    set_instruments = set()
    # create a list of files to save the information
    d_files = {}
    s_path = os.path.dirname(os.path.abspath(s_fname))
    s_path = s_path.split('/original')[0]
    for s_instr in l_instr:
        s_file = s_path + '/temp/{}/{}_raw.txt'
        d_files[s_instr] = open(s_file.format(s_date, s_instr), 'w')

    myParser = LineParser(s_type)
    with zipfile.ZipFile(s_fname, 'r') as archive:
        fr = archive.open(archive.infolist()[0])
        for idx, row in enumerate(fr):
            d_aux = myParser(row)
            if myParser.last_identification == 'MSG':
                s_instr = d_aux['instrumento_symbol']
                set_instruments.add(s_instr)
                # include rows just in desired instruments
                if d_aux['instrumento_symbol'] in l_instr:
                    d_files[s_instr].write(row)
            elif myParser.last_identification == 'RH':
                # include the header in all files
                for s_intr in l_instr:
                    d_files[s_intr].write(row)
        fr.close()

    if myParser.last_identification == 'RT':
        # last row of all files
        for s_intr in l_instr:
            d_files[s_intr].write(row)
            d_files[s_intr].close()

    # save a list of the instruments in the file
    s_file_instr = s_path + '/temp/{:%Y%m%d}/{:%Y%m%d}_{}_INSTR.txt'
    ts_date = d_aux['end_date']
    s_file_instr = s_file_instr.format(ts_date, ts_date, myParser.s_file_type)
    df_instr = pd.Series(list(set_instruments))
    df_instr.sort_values(inplace=True)
    df_instr.to_csv(s_file_instr, sep='\t', index=False)

    # criate zipfiles
    i_count = 1
    for s_instr in l_instr:
        s_file = s_path + '/temp/{:%Y%m%d}/{:%Y%m%d}_{}_{}.zip'
        s_file = s_file.format(d_aux['end_date'],
                               d_aux['end_date'],
                               myParser.s_file_type,
                               s_instr)
        zf = zipfile.ZipFile(s_file,
                             mode='w',
                             compression=zipfile.ZIP_DEFLATED,
                             )
        # s_command = 'sort -t $";" -k 5 -o {} {}'
        s_command = 'sort -t $";" -k 12,12 -k 7,7 -k 5,5 -o {} {}'
        s_file2 = s_path + '/temp/{:%Y%m%d}/{}.txt'
        s_file3 = s_path + '/temp/{:%Y%m%d}/{}_raw.txt'
        s_command = s_command.format(s_file2.format(ts_date, s_instr),
                                     s_file3.format(ts_date, s_instr))
        subprocess.check_output(['bash', '-c', s_command])
        zf.write(s_file2.format(ts_date, s_instr))
        # close the files and clean the memory
        zf.close()
        os.remove(s_file3.format(ts_date, s_instr))
        os.remove(s_file2.format(ts_date, s_instr))
        i_count += 1
    # print succcess message
    print '{} files created'.format(i_count)


class LineParser(object):
    '''
    Parser the a row from the market data files provided by BmfBovespa. Apply
    the related layout to parse de string
    '''
    def __init__(self, s_file_type):
        '''
        Initialize a LineParser object

        :param s_file_type: string. bid, ask or trade
        '''
        self.s_file_type = s_file_type
        # define dictionaries to classification
        self.d_valid_values = {'001': 'New', '002': 'Update',
                               '003': 'Cancel', '004': 'Trade',
                               '005': 'Reentry', '006': 'New Stop Price',
                               '007': 'Rejected', '008': 'Removed',
                               '009': 'Stop Price Triggered',
                               '011': 'Expired', '012': 'Eliminated'}
        self.d_order_status = {'0': 'New', '1': 'Partially Filled',
                               '2': 'Filled', '4': 'Canceled', '5': 'Replaced',
                               '8': 'Rejected', 'C': 'Expired'}

        self.d_aggressor = {'0': 'Neutral',
                            '1': 'Agressor',
                            '2': 'Passive',
                            '': 'None'}
        self.d_side = {'1': 'Buy Order', '2': 'Sell Order'}
        self.d_trade_indicator = {' ': 'None',
                                  '1': 'Trade',
                                  '2': 'Trade cancelled'}
        self.d_cross_trade_indicator = {' ': 'None',
                                        '1': 'Intentional',
                                        '0': 'Not Intentional'}
        # attribute the correct parse function
        if s_file_type in ['BID', 'ASK']:
            self.func_parse = self.parse_detail_book
        elif s_file_type == 'TRADE':
            self.func_parse = self.parse_detail_trade
        else:
            raise NoValidTypeException()

    def split_string(self, s_row):
        '''
        Split the data accord to its format

        :param s_row: string. a row in the file being parsed
        '''
        # split the data
        self.last_row = s_row
        l_fields = s_row.strip().split(';')
        if len(l_fields) == 1:
            l_fields = s_row.strip().split(' ')
            l_fields = [x for x in l_fields if x != '']
        self.i_fileds_count = len(l_fields)
        # store list as attribute
        self.l_fields = l_fields

    def parse_header(self, l_data):
        '''
        Format the list of the header passe and return a dictionary

        :param l_data: list. list with the elements of the parsed row
        '''
        d_rtn = {}
        # Name of file
        d_rtn['name_of_file'] = l_data[1]
        # Initial date of file
        d_rtn['initial_date'] = pd.to_datetime(l_data[2], format='%Y-%m-%d')
        d_rtn['initial_date'] = d_rtn['initial_date'].date()
        # End date of file
        d_rtn['end_date'] = pd.to_datetime(l_data[3], format='%Y-%m-%d')
        d_rtn['end_date'] = d_rtn['end_date'].date()
        # Contain the total of lines when the file Trailer record
        d_rtn['total_of_lines'] = int(l_data[4])

        return d_rtn

    def parse_detail_trade(self, l_data):
        '''
        Format the list of details passed and return a dictionary. It is
        related to the NEG files.

        :param l_data: list. list with the elements of the parsed row
        '''
        d_rtn = {}
        # Session date
        d_rtn['session_date'] = l_data[0]
        # Instrument identifier
        d_rtn['instrumento_symbol'] = l_data[1].replace(' ', '')
        # Trade number
        d_rtn['session_date'] = l_data[2]
        # Trade price
        d_rtn['trade_price'] = float(l_data[3])
        # Traded quantity
        d_rtn['trade_qty'] = int(l_data[4])
        # Trade time (format HH:MM:SS.NNNNNN)
        d_rtn['trade_time'] = l_data[5]
        # Trade indicador: 1 - Trade  / 2 - Trade cancelled
        d_rtn['trade_indicator'] = self.d_trade_indicator[l_data[6]]
        # Buy order date
        d_rtn['buy_order_date'] = l_data[7]
        # Sequential buy order number
        d_rtn['seq_buy_order_id'] = l_data[8]
        # Secondary Order ID -  Buy Order.
        d_rtn['secondary_buy_order_id'] = l_data[9]
        # Aggressor Buy Order Indicator
        d_rtn['agressor_buy_order_indicator'] = self.d_aggressor[l_data[10]]
        # Sell order sell date
        d_rtn['sell_order_date'] = l_data[11]
        # Sequential sell order number
        d_rtn['seq_sell_order_id'] = l_data[12]
        # Secondary Order ID - Sell Order
        d_rtn['secondary_sell_order_id'] = l_data[13]
        # Aggressor Sell Order Indicator
        d_rtn['agressor_sell_order_indicator'] = self.d_aggressor[l_data[14]]
        # Cross Trade Indicator
        s_aux = self.d_cross_trade_indicator[l_data[15]]
        d_rtn['cross_trade_indicator'] = s_aux
        # Buy Member
        d_rtn['buy_member'] = int(l_data[16])
        # Sell Member
        d_rtn['sell_member'] = int(l_data[17])

        return d_rtn

    def parse_detail_book(self, l_data):
        '''
        Format the list of details passed and return a dictionary. It is
        related to the OFER_CPA and OFER_VDA files.

        :param l_data: list. list with the elements of the parsed row
        '''
        d_rtn = {}
        # Session date
        d_rtn['session_date'] = l_data[0]
        # Instrument identifier
        d_rtn['instrumento_symbol'] = l_data[1].replace(' ', '')
        # "1" Buy Order /  "2" Sell Order
        d_rtn['order_side'] = self.d_side[l_data[2]]
        # Sequential order number
        d_rtn['seq_order_number'] = l_data[3]
        # Secondary Order ID
        d_rtn['secondary_order_id'] = l_data[4]
        # Valid values
        d_rtn['execution_type'] = self.d_valid_values[l_data[5]]
        # Order time entry in system (HH:MM:SS.NNN) used as priority indicator
        s_aux = l_data[6]
        f_aux = int(s_aux[0:2])*60*60 + int(s_aux[3:5])*60 + int(s_aux[6:8])
        f_aux += float(s_aux[9:])/10**6
        d_rtn['priority_time'] = s_aux
        d_rtn['priority_seconds'] = f_aux
        # Priority indicator
        # !! can include 100 messagesn from a external agent
        d_rtn['priority_indicator'] = int(l_data[7]) * 100
        # Order price
        d_rtn['order_price'] = float(l_data[8].replace(' ', ''))
        # Total quantity of order
        d_rtn['total_qty_order'] = int(l_data[9])
        # Traded quantity of order
        d_rtn['traded_qty_order'] = int(l_data[10])
        # Order date
        d_rtn['order_date'] = l_data[11]
        # Order datetime entry (AAAA-MM-DD HH:MM:SS)
        d_rtn['order_datetime_entry'] = l_data[12]

        # redefine priority time
        # NOTE: in Bovespa, orders from prior days mess up the flow
        if d_rtn['session_date'] != d_rtn['order_date']:
            f_aux = int(8)*60*60.
            d_rtn['priority_seconds'] = f_aux

        # Order status
        d_rtn['order_status'] = self.d_order_status[l_data[13]]
        # Aggressor Indicator
        d_rtn['agressor_indicator'] = self.d_aggressor[l_data[14]]
        # Entering Firm - Available from March/2014
        d_rtn['member'] = int(l_data[15])
        # field usess to control the loop
        s_idx = d_rtn['order_date'].replace('-', '')
        d_rtn['idx'] = s_idx[2:] + l_data[4][6:]
        d_rtn['is_today'] = d_rtn['order_date'] == d_rtn['session_date']
        # include new fields
        d_rtn['action'] = 'history'
        d_rtn['agent_id'] = 10
        d_rtn['order_id'] = d_rtn['priority_indicator']
        d_rtn['org_total_qty_order'] = d_rtn['total_qty_order']
        d_rtn['order_qty'] = None
        # correct the total_qty_order
        i_org_qty = d_rtn['org_total_qty_order']
        i_traded_qty = d_rtn['traded_qty_order']
        d_rtn['total_qty_order'] = i_org_qty - i_traded_qty
        return d_rtn

    def __call__(self, s_row):
        '''
        Parse the string passed

        :param s_row: string. a row in the file being parsed
        '''
        self.split_string(s_row)
        l_fields = self.l_fields
        # define the type of the row and parse it
        self.last_identification = 'MSG'
        if self.i_fileds_count == 5:
            self.last_identification = l_fields[0]
            d_rtn = self.parse_header(l_fields)
        elif self.i_fileds_count >= 15:
            d_rtn = self.func_parse(l_fields)
        else:
            raise NoMappedException()
        return d_rtn
