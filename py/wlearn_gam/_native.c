#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyModuleDef native_module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "wlearn-gam native C core",
    -1,
    NULL
};

PyMODINIT_FUNC PyInit__native(void) {
    return PyModule_Create(&native_module);
}
