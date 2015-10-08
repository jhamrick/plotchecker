import argparse
import sys
import subprocess as sp
import warnings


def install_flit():
    sp.check_call([sys.executable, '-m', 'pip', 'install', 'flit', 'mock'])


def install_plotchecker(symlink):
    from pathlib import Path
    from flit.install import Installer
    from flit.log import enable_colourful_output
    import mock

    # Hack to make docs build on RTD
    MOCK_MODULES = ['numpy', 'matplotlib', 'matplotlib.pyplot', 'matplotlib.colors', 'matplotlib.markers']
    for mod_name in MOCK_MODULES:
        sys.modules[mod_name] = mock.Mock()

    enable_colourful_output()
    p = Path('flit.ini')
    Installer(p, symlink=symlink, deps='none').install()


def install():
    install_flit()
    install_plotchecker(symlink=False)


def develop():
    install_flit()
    install_plotchecker(symlink=True)


def egg_info():
    print("This setup.py does not support egg_info. Please re-run with:")
    print("    python setup.py develop")
    sys.exit(1)


if __name__ == "__main__":
    warnings.warn(
        'Warning: this setup.py uses flit, not setuptools. '
        'Behavior may not be exactly what you expect! '
        'In particular, this DOES NOT install ANY dependencies! '
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
