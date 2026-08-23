import ctypes
import ctypes.util
import importlib.machinery
import os

_lib = None


def _find_lib():
    env_path = os.environ.get('GAM_LIB_PATH')
    if env_path and os.path.isfile(env_path):
        return env_path

    this_dir = os.path.dirname(os.path.abspath(__file__))
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = os.path.join(this_dir, '_native' + suffix)
        if os.path.isfile(candidate):
            return candidate

    dev_build = os.path.normpath(os.path.join(this_dir, '..', '..', 'build', 'libgam.so'))
    if os.path.isfile(dev_build):
        return dev_build

    return ctypes.util.find_library('gam')


def get_lib():
    global _lib
    if _lib is None:
        path = _find_lib()
        if path is None:
            raise RuntimeError(
                'Cannot find libgam. Set GAM_LIB_PATH, install wlearn-gam, '
                'or build with: cd gam && cmake -S . -B build && cmake --build build'
            )
        _lib = ctypes.CDLL(path)
    return _lib
