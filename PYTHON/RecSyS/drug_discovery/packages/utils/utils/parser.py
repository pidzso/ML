import argparse


def args_parser():
    # parser for federated_multilearning
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed',      type=int,   default=0,              help='Random seed')
    parser.add_argument('--instance',  type=str,   default='',             help='Identification of the run')

    parser.add_argument('--epochs',    type=int,   default=20,             help="Number of rounds of training")
    parser.add_argument('--num_users', type=int,   default=10,             help="Number of users")
    parser.add_argument('--ansgar',                default=False,          help="Use Ansgar Split")
    parser.add_argument('--out_path',  type=str,   required=True,          help='Directory to save results')
    parser.add_argument('--hls',       type=int,   default=40,             help="Hidden layer sizes")

    parser.add_argument('--comp_m',    type=str,   default=None,          help='Compression methods')
    parser.add_argument('--comp_p',    type=str,   default=None,          help='Compression params')

    parser.add_argument('--noise',     type=str,   default=None,           help="Noising participants")
    parser.add_argument('--level',     type=str,   default=None,           help="Level of Noise")

    parser.add_argument('--weight',    type=str,   default='no',           help='Weight participants')
    parser.add_argument('--methods',   nargs='+',  default=['l1o', 'i1i'], help='Weight participants')
    parser.add_argument('--eval',      type=str,   default='self',         help='Evaluator of the updates')

    parser.add_argument('--opt',       type=str,   default='SGD',          help="Type of optimizer")
    parser.add_argument('--lr',        type=float, default=1.0,            help='Learning rate')
    parser.add_argument('--do',        type=float, default=0.2,            help='Drop out')
    parser.add_argument('--mom',       type=float, default=0.0,            help='SGD momentum')
    parser.add_argument('--wd',        type=float, default=1e-5,           help='Weight Decay')
    parser.add_argument('--act',       type=str,   default='relu',         help='Activation function')
    parser.add_argument('--gpu',                   default=None,           help="To use cuda, set to a specific GPU ID.")

    args = parser.parse_args()

    print(f'Seed     : {args.seed}')
    print(f'Ansgar   : {args.ansgar}')
    print(f'Rounds   : {args.epochs}')

    print(f'\nCompression Methods : {args.comp_m}')
    print(f'Compression Param   : {args.comp_p}')

    print(f'Contribution Methods : {args.methods}')
    print(f'Contribution Eval    : {args.eval}')

    print(f'Weights : {args.weight}')
    print(f'Noise   : {args.noise}')
    print(f'Level   : {args.level}\n')

    return args
