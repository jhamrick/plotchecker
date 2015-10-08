import argparse
import sys
import subprocess as sp
import warnings


def install_flit():
    sp.check_call([sys.executable, '-m', 'pip', 'install', 'flit'])


def install_nbgrader(symlink):
    command = ['flit', 'install']
    args = ['--deps', 'all']

    if symlink:
        args += ['--symlink']

    sp.check_call([sys.executable, '-m'] + command + args)


def install():
    install_flit()
    install_nbgrader(symlink=False)


def develop():
    install_flit()
    install_nbgrader(symlink=True)


def egg_info():
    print("This setup.py does not support egg_info. Please re-run with:")
    print("    python setup.py develop")
    sys.exit(1)


if __name__ == "__main__":
    warnings.warn(
        'Warning: this setup.py uses flit, not setuptools. '
        'Behavior may not be exactly what you expect!'
    )

    parser = argparse.ArgumentParser('install_dev')
    subparsers = parser.add_subparsers()

    install_parser = subparsers.add_parser('install')
    install_parser.set_defaults(func=install)
    install_parser.add_argument(
        '--force', action='store_true',
        help="this flag doesn't actually do anything, but is needed for readthedocs")

    develop_parser = subparsers.add_parser('develop')
    develop_parser.set_defaults(func=develop)

    egg_info_parser = subparsers.add_parser('egg_info')
    egg_info_parser.set_defaults(func=egg_info)

    args = parser.parse_args()
    args.func()
