import os
import pkg_resources
import importlib

from janggo.utils import get_genome_size
from janggo.utils import get_parse_tree

from extra import _fnn_model1
from extra import _cnn_model2


def test_genome_size():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')
    gsize = get_genome_size('sacCer3', data_path)
    print(gsize)
    assert gsize['chrXV'] == 1091291


def test_modelzoo_parser1():
    parsetree = get_parse_tree(_fnn_model1)
    assert '_fnn_model1' in parsetree
    assert '_cnn_model2' not in parsetree
    assert '_model3' not in parsetree
    assert len(parsetree) == 1

def test_modelzoo_parser2():
    parsetree = get_parse_tree(_cnn_model2)
    assert '_fnn_model1' not in parsetree
    assert '_cnn_model2' in parsetree
    assert '_model3' not in parsetree
    assert len(parsetree) == 1


def _model4():
    inputs = 3
    outputs = inputs
    return inputs, outputs


def test_modelzoo_parser3():
    parsetree = get_parse_tree(_model4)
    assert '_model4' in parsetree
    assert len(parsetree) == 1
