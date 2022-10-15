# coding=utf-8
# Automatically generated by Pynguin.
import numpy as module_1

import mridc.collections.reconstruction.metrics.evaluate as module_0


def test_case_0():
    try:
        ndarray_0 = None
        bytes_0 = b""
        int_0 = -1621
        set_0 = {int_0, bytes_0}
        metrics_0 = module_0.Metrics(bytes_0, int_0, set_0)
        assert metrics_0.metrics_scores == {}
        assert metrics_0.output_path == -1621
        assert metrics_0.method == {b"", -1621}
        var_0 = metrics_0.means()
        float_0 = module_0.nmse(ndarray_0, ndarray_0)
    except BaseException:
        pass


def test_case_1():
    try:
        set_0 = None
        str_0 = ""
        list_0 = [str_0, set_0, str_0, str_0]
        dict_0 = {}
        metrics_0 = module_0.Metrics(str_0, list_0, dict_0)
        assert metrics_0.metrics_scores == {}
        assert metrics_0.output_path == ["", None, "", ""]
        assert metrics_0.method == {}
        int_0 = 178
        list_1 = [int_0]
        ndarray_0 = module_1.ndarray(*list_1)
        float_0 = module_0.ssim(ndarray_0, ndarray_0)
    except BaseException:
        pass


def test_case_2():
    try:
        ndarray_0 = None
        set_0 = {ndarray_0}
        list_0 = []
        str_0 = "[{\tRJ\tjWBpyT\\"
        bool_0 = False
        metrics_0 = module_0.Metrics(str_0, bool_0, set_0)
        assert len(metrics_0.metrics_scores) == 12
        assert metrics_0.output_path is False
        assert metrics_0.method == {None}
        metrics_1 = module_0.Metrics(set_0, list_0, metrics_0)
        assert len(metrics_1.metrics_scores) == 1
        assert metrics_1.output_path == []
        var_0 = metrics_1.stddevs()
        float_0 = module_0.psnr(ndarray_0, ndarray_0)
    except BaseException:
        pass


def test_case_3():
    try:
        ndarray_0 = None
        float_0 = module_0.psnr(ndarray_0, ndarray_0)
    except BaseException:
        pass


def test_case_4():
    try:
        str_0 = "u!XL{]:(U\\HaD9\\H\n"
        bytes_0 = b"\x883\x02\x04\xdc{\x86\xa3\x1d\x01"
        list_0 = [str_0, str_0, bytes_0]
        ndarray_0 = module_1.ndarray(*list_0)
    except BaseException:
        pass


def test_case_5():
    try:
        str_0 = "ADjC|5DA{G)oG}Wh!"
        bytes_0 = b"\x80\x11\xd0/\x9b\x04\xe4\xe6Z#\n\xa9y\x0b\xc6"
        bytes_1 = b"\xb3\x04\xb95%\xb1\xfd\x15\xe5\xce\x13\xd1"
        metrics_0 = module_0.Metrics(str_0, bytes_0, bytes_1)
        assert len(metrics_0.metrics_scores) == 14
        assert metrics_0.output_path == b"\x80\x11\xd0/\x9b\x04\xe4\xe6Z#\n\xa9y\x0b\xc6"
        assert metrics_0.method == b"\xb3\x04\xb95%\xb1\xfd\x15\xe5\xce\x13\xd1"
        var_0 = metrics_0.__repr__()
    except BaseException:
        pass


def test_case_6():
    try:
        ndarray_0 = module_1.ndarray()
    except BaseException:
        pass


def test_case_7():
    try:
        str_0 = "gbE:p\tp"
        set_0 = {str_0}
        dict_0 = {}
        str_1 = "vdH|&zT\x0bCdinBYn\x0b\r'"
        int_0 = -3166
        str_2 = "AE<*|_m\\n"
        metrics_0 = module_0.Metrics(str_1, int_0, str_2)
        assert len(metrics_0.metrics_scores) == 15
        assert metrics_0.output_path == -3166
        assert metrics_0.method == "AE<*|_m\\n"
        var_0 = metrics_0.push(set_0, dict_0)
    except BaseException:
        pass


def test_case_8():
    try:
        ndarray_0 = module_1.ndarray()
    except BaseException:
        pass


def test_case_9():
    try:
        ndarray_0 = None
        float_0 = module_0.mse(ndarray_0, ndarray_0)
    except BaseException:
        pass
