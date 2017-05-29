from subprocess import call
import sys
import argparse
import textwrap

if __name__ == '__main__':
    s_txt = '''\
            Run simulation
            --------------------------------
            Run one of the agents implemented in agents
            to trade Brazilian interest rate future
            contracts
            '''
    s_txt2 = 'the type of the agent to run the simulation. '
    obj_formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=obj_formatter,
                                     description=textwrap.dedent(s_txt))
    parser.add_argument('agent', default='debug', help=s_txt2,
                        metavar='<OPTION>')
    parser.add_argument('-t', '--trials', default=None, type=int, metavar='',
                        help='number of trials to perform in the same file')
    parser.add_argument('-d', '--date', default='', type=str, metavar='',
                        help='date of the file to be used (formart AAAAMMDD)')
    parser.add_argument('-v', '--version', default=None, type=int, metavar='',
                        help='version of the learner to be tested')
    s_help = 'If should use the value function from the previous day'
    parser.add_argument('-vf', '--valfunc', action='store_true',
                        help=s_help)
    s_help = 'If should keep learning (when using a previous value function)'
    parser.add_argument('-kl', '--keeplrn', action='store_true',
                        help=s_help)
    parser.add_argument('-i', '--instr', default=None, type=str, metavar='',
                        help='Intrument to trade. Default is DI1F21')
    s_help = 'number of different sessions to iterate on each trial'
    parser.add_argument('-s', '--sessions', default=None, type=int, metavar='',
                        help=s_help)
    args = parser.parse_args()
    # run simulation
    i_trials = '{}'.format(args.trials)
    s_date = args.date
    l_command_to_run = ['python', '-m', 'market_sim.agent', args.agent,
                        '-t', i_trials, '-d', s_date]
    if args.version:
        i_version = '{}'.format(args.version)
        l_command_to_run += ['-v', i_version]
    if args.valfunc:
        l_command_to_run += ['-vf']
    if args.keeplrn:
        l_command_to_run += ['-kl']
    if args.instr:
        l_command_to_run += ['-i', args.instr]
    if args.sessions:
        l_command_to_run += ['-s', '{}'.format(args.sessions)]

    call(l_command_to_run)
