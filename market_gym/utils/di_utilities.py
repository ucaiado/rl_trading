#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement functions to help recover some informations from contracts that may
be used in tests

@author: ucaiado

Created on 04/03/2017
"""
# load libraries
import numpy as np
import os
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup as bsoup
import requests


# set up global variables
S_ROOT = os.getcwd()
FR_DU = S_ROOT + '/data/misc/du.txt'
FR_CONFIG = S_ROOT + '/data/misc/config.txt'
DF_DU = pd.read_table(FR_DU, parse_dates=['DATE'])
DF_DU = pd.DataFrame(list(DF_DU.DATE.apply(lambda x: x.date())))
DF_DU.columns = ['DATE']
DF_CONFIG = pd.read_table(FR_CONFIG)

'''
Begin help functions
'''


def WorkingDays(s_start, s_end, df_du=DF_DU, b_returnDF=False):
    '''
    Return the total Working days between two dates. All parameter Can be
    passed as timestamp object
    :param s_start: string. The start date. Formated as dd/mm/yyyy
    :param s_end: string. The end date. Format as dd/mm/yyyy
    :param b_returnDF: boolean. Return the dataframe with all valid dates
    '''
    # checa se eh necessario converter string
    ts_start = s_start
    if isinstance(s_start, str):
        ts_start = datetime(int(s_start[6:]), int(s_start[3:5]),
                            int(s_start[0:2]))
        ts_start = ts_start.date()
    ts_end = s_end

    if isinstance(s_end, str):
        ts_end = datetime(int(s_end[6:]), int(s_end[3:5]),
                          int(s_end[0:2]))
        ts_end = ts_end.date()
    # checa se datas estao na ordem certa e inverte se necessario
    i_mult = 1
    if ts_end < ts_start:
        ts_aux = ts_end
        ts_end = ts_start
        ts_start = ts_aux
        i_mult = -1
    # calcula quantidade de dias uteis
    df_aux = df_du.ix[(df_du >= ts_end).DATE, :].reset_index(drop=True)
    ts_end = pd.to_datetime(df_aux.ix[0, :].values)[0].date()
    i_du = (((df_du > ts_start) & (df_du <= ts_end)).sum()*1.).values[0]
    if b_returnDF:
        df_rtn = df_du[((df_du >= ts_start) & (df_du <= ts_end)).T.values[0]]
        return df_rtn.reset_index(drop=True)

    return i_du * i_mult


def truncate(f, n):
    '''
    Truncates/pads a float f to n decimal places
    '''
    s = '%.12f' % f
    i, p, d = s.partition('.')
    return np.ceil((float('.'.join([i, (d+'0'*n)[:n]]))-0.005)*100.)/100.

'''
End help functions
'''


def calulateDI1expirationDays(s_instrument, s_today=None, df_du=DF_DU,
                              df_config=DF_CONFIG):
    '''
    Calculate the number of days to expiration of the DI1 contract

    :param s_instrument: string. The name of the DI1 instrument
    :param s_today: string. The today date in the format dd/mm/yyyy
    :return i_du: integer. numeber of working days to expiration
    :return ts_venc: timestamp. Date of expiration
    :return ts_today: timestamp. Today date
    '''
    if isinstance(s_today, type(None)):
        ts_today = datetime.today().date()
    else:
        ts_today = datetime(int(s_today[6:]), int(s_today[3:5]),
                            int(s_today[0:2]))
        ts_today = ts_today.date()

    # separa dados do nome do ativo
    i_yVenc = int(s_instrument[4:7])+2000
    # checa mes de vencimento
    i_mVenc = df_config.ix[(df_config.TIPO == 'DI_VENC') &
                           (df_config.ITEM == s_instrument[3:4]),
                           'VALOR'].values[0]
    ts_venc = datetime(i_yVenc, int(i_mVenc), 1).date()
    i_du = WorkingDays(s_start=ts_today, s_end=ts_venc, df_du=df_du)

    return i_du, ts_venc, ts_today


def get_param_to_env(s_date, l_instrument):
    '''
    Return lists of days to maturity, settlement prices and interest rate
    to be used in the YieldCurve environment

    :poaram s_date: string. date in the format AAAAMMDD
    :param l_instrument: list. name of the instrument desired
    '''
    # TODO: generalize to more than one date
    s_date1 = '{}/{}/{}'.format(s_date[-2:], s_date[-4:-2], s_date[:4])
    o_aux = Settlements()
    func_du = calulateDI1expirationDays
    l_du = []
    l_pu = []
    l_price_adj = []
    for s_cmm in l_instrument:
        f_du, dt1, dt2 = func_du(s_cmm, s_today=s_date1)
        l_du.append(f_du)
        df = o_aux.getData(s_cmm, s_date1, s_date1, b_notInclude=False)

        l_pu.append(df['PU_Atual'].values[0])
        f_pu_anterior = df['PU_Anterior'].values[0]
        f_price = ((10.**5/f_pu_anterior)**(252./f_du)-1)*100
        l_price_adj.append(f_price)
    l_du = [l_du]
    l_pu = [l_pu]
    l_price_adj = [l_price_adj]

    return l_du, l_pu, l_price_adj


class Settlements(object):
    '''
    Represent the settlements of DI1 contracts recover from BVMF website
    '''

    def __init__(self):
        '''
        Initialize a Settlements object
        '''
        # define variaveis
        self.s_fname = S_ROOT + '/data/misc/ajustes.tsv'
        self.somethingChanged = False
        self.url = 'http://www2.bmf.com.br/pages/portal/bmfbovespa/boletim1'
        self.url += '/Ajustes1.asp?txtData='
        # recupera informacao anterior e inserwe em d_settlements
        df = pd.read_csv(self.s_fname,
                         sep="\t",
                         dayfirst=True,
                         parse_dates=["DATA"])
        d_rtn = {}
        d_aux = df.T.to_dict()
        for x in d_aux:
            if d_aux[x]['CMM'] not in d_rtn:
                d_rtn[d_aux[x]['CMM']] = {}
            d_data = {'PU_Anterior': d_aux[x]['PU_ANTERIOR'],
                      'PU_Atual': d_aux[x]['PU_ATUAL']}
            s_date = '{:%d/%m/%Y}'.format(d_aux[x]['DATA'])
            d_rtn[d_aux[x]['CMM']][s_date] = d_data
        self.d_settlements = d_rtn

    def getData(self, s_cmm, s_start, s_end, b_notInclude=True):
        '''
        Get all data related to a specific maturity to a date interval

        :param s_cmm: string. Commodity name
        :param s_start: string. The start date. Formated as dd/mm/yyyy
        :param s_end: string. The start date. Formated as dd/mm/yyyy
        :param b_notInclude: boolean. Not include today date
        '''
        d_settle = self.d_settlements
        s_today = '{:%d/%m/%Y}'.format(datetime.today().date())
        d_rtn = {}
        if pd.to_datetime(s_end, format='%d/%m/%Y') > datetime.today():
            s_end = '{:%d/%m/%Y}'.format(datetime.today())
        # inicio chave de cmm se nao tiver registro
        if s_cmm not in d_settle:
            d_settle[s_cmm] = {}
        # itero datas
        df_aux = WorkingDays(s_start, s_end, b_returnDF=True)
        for x in df_aux.iterrows():
            s_date = '{:%d/%m/%Y}'.format(x[1]['DATE'])
            if s_date in d_settle[s_cmm]:
                # se jah tiver informacao para este ativo, recupero
                d_rtn[s_date] = d_settle[s_cmm][s_date]
            else:
                # se nao tiver, puxo da web de todos DI1 e guardo
                if (s_today == s_date) & (b_notInclude):
                    continue
                d_aux = self._getFromWeb(s_date)
                if d_aux:
                    for y in d_aux:
                        # para todas as cmm, checa se jah tem registro
                        if y not in d_settle:
                            d_settle[y] = {}
                        s_key = d_aux[y].keys()[0]
                        # se dados para essa data e ativo nao existirem,
                        # preenche
                        if s_key not in d_settle[y]:
                            self.somethingChanged = True
                            d_settle[y][s_key] = d_aux[y][s_key]
                            if y == s_cmm:
                                d_rtn[s_key] = d_aux[y][s_key]
        # se tiver puxado novos dados, guardo o que puxei
        if self.somethingChanged:
            # guardo no objeto
            self.d_settlements = d_settle
            # escrevo no CSV
            l_data = []
            for j in d_settle:
                for i in d_settle[j]:
                    d_to_df = {}
                    d_to_df['PU_ANTERIOR'] = d_settle[j][i]['PU_Anterior']
                    d_to_df['PU_ATUAL'] = d_settle[j][i]['PU_Atual']
                    d_to_df['CMM'] = j
                    d_to_df['DATA'] = i
                    l_data.append(d_to_df)
            df = pd.DataFrame(l_data)
            if len(l_data) > 0:
                df = df.sort_values(by=['CMM', 'DATA'])
                df.to_csv(self.s_fname, index=False, sep='\t')
                self.d_settlements = d_settle
            self.somethingChanged = False
        # arruma ordem dos dados
        df = pd.DataFrame(d_rtn).T
        df.index = pd.to_datetime(df.index, format='%d/%m/%Y')
        df = df.sort_index(ascending=True)

        return df

    def getSettlements(self, l_cmm, s_start, s_end):
        '''
        Return two dataframes. One with the PU_Anterior and other with the
        PU_Atual

        :param l_cmm: list of string. Commodities names
        :param s_start: string. The start date. Formated as dd/mm/yyyy
        :param s_end: string. The start date. Formated as dd/mm/yyyy
        '''
        df_anterior = None
        df_atual = None
        for s_cmm in l_cmm:
            df = self.getData(s_cmm, s_start, s_end)
            df_aux1 = df['PU_Anterior']
            df_aux1.name = s_cmm
            df_aux2 = df['PU_Atual']
            df_aux2.name = s_cmm
            if isinstance(df_anterior, type(None)):
                df_anterior = df_aux1.copy()
                df_atual = df_aux2.copy()
            else:
                df_anterior = pd.concat([df_anterior, df_aux1], axis=1)
                df_atual = pd.concat([df_atual, df_aux2], axis=1)

        # organiza dados
        df_anterior = pd.DataFrame(df_anterior).sort_index(ascending=True)
        df_atual = pd.DataFrame(df_atual).sort_index(ascending=True)
        return df_anterior, df_atual

    def _getFromWeb(self, s_date, s_filter='DI1'):
        '''
        Return a dataframe of the data of specific data
        '''
        # recupera dados de website
        url = self.url + s_date
        resp = requests.get(url)
        ofile = resp.text
        soup = bsoup(ofile, 'lxml')
        soup.prettify()
        tables = soup.find_all('table')
        storeValueRows = tables[6].find_all('tr')
        # checa re retrornou valores
        if len(storeValueRows) == 2:
            return None
        # separa informacoes de interesse
        storeMatrix = []
        s_ass = ''
        for row in storeValueRows:
            storeMatrixRow = []
            for cell in row.find_all('td'):
                s = cell.get_text().strip()
                if s != '':
                    storeMatrixRow.append(s)
            if len(storeMatrixRow) == 6:
                s_ass = storeMatrixRow[0].split()[0]
                storeMatrixRow = [s_ass] + storeMatrixRow[1:]
            elif len(storeMatrixRow) == 5:
                storeMatrixRow = [s_ass] + storeMatrixRow
            storeMatrix.append(storeMatrixRow)
        # monta dataframe com dados filtrados
        df = pd.DataFrame(storeMatrix[1:], columns=storeMatrix[0])
        if s_filter:
            df = df[df.Mercadoria == s_filter].reset_index(drop=True)
        df = df.ix[:, :-2]
        df.index = [list(df.Mercadoria + df.Vct), [s_date]*df.shape[0]]
        df.drop([u'Mercadoria', u'Vct'], axis=1, inplace=True)
        # transforma dados em dicionario
        d_rtn = {}
        d_aux = df.T.to_dict()
        for x in d_aux:
            if x[0] not in d_rtn:
                d_rtn[x[0]] = {}
            s_atual = d_aux[x][u'Pre\xe7o de Ajuste Atual']
            s_anterior = d_aux[x][u'Pre\xe7o de Ajuste Anterior']
            s_atual = s_atual.replace('.', '').replace(',', '.')
            s_anterior = s_anterior.replace('.', '').replace(',', '.')
            d_rtn[x[0]][x[1]] = {'PU_Anterior': float(s_anterior),
                                 'PU_Atual': float(s_atual)}
        return d_rtn
