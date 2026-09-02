# -*- coding: utf-8 -*-

'''Tests for cell2cell.external.gseapy -- offline paths only'''

import io
import sys

import pytest

from cell2cell.external import gseapy as gseapy_module


def test_load_gmt_from_a_local_file(tiny_gmt):
    pathway_per_gene = gseapy_module.load_gmt(tiny_gmt, backup_url=None)
    assert pathway_per_gene['Protein-A'] == {'TOYDB_PATHWAY_ONE'}
    assert pathway_per_gene['Protein-B'] == {'TOYDB_PATHWAY_ONE', 'TOYDB_PATHWAY_TWO'}
    assert pathway_per_gene['Protein-F'] == {'TOYDB_PATHWAY_THREE'}


def test_load_gmt_covers_every_listed_gene(tiny_gmt):
    pathway_per_gene = gseapy_module.load_gmt(tiny_gmt, backup_url=None)
    assert set(pathway_per_gene.keys()) == {'Protein-A', 'Protein-B', 'Protein-C',
                                            'Protein-E', 'Protein-F'}


def test_load_gmt_readable_names(tiny_gmt):
    pathway_per_gene = gseapy_module.load_gmt(tiny_gmt, backup_url=None,
                                              readable_name=True)
    names = set().union(*pathway_per_gene.values())
    # The DB prefix is dropped and underscores become spaces
    assert all('_' not in name for name in names)


def test_generate_lr_geneset_with_an_injected_annotation(tiny_gmt):
    pathway_per_gene = gseapy_module.load_gmt(tiny_gmt, backup_url=None)
    lr_list = ['Protein-A^Protein-B', 'Protein-B^Protein-C', 'Protein-E^Protein-F']
    geneset = gseapy_module.generate_lr_geneset(lr_list, lr_sep='^',
                                                pathway_per_gene=pathway_per_gene,
                                                min_pathways=0, max_pathways=10000)
    assert isinstance(geneset, dict)
    assert len(geneset) > 0
    for pathway, pairs in geneset.items():
        for pair in pairs:
            assert pair in lr_list


def test_generate_lr_geneset_respects_min_pathways(tiny_gmt):
    pathway_per_gene = gseapy_module.load_gmt(tiny_gmt, backup_url=None)
    lr_list = ['Protein-A^Protein-B', 'Protein-B^Protein-C']
    geneset = gseapy_module.generate_lr_geneset(lr_list, lr_sep='^',
                                                pathway_per_gene=pathway_per_gene,
                                                min_pathways=100, max_pathways=10000)
    assert geneset == {}


def test_generate_lr_geneset_with_complexes(tiny_gmt):
    pathway_per_gene = gseapy_module.load_gmt(tiny_gmt, backup_url=None)
    lr_list = ['Protein-A&Protein-B^Protein-C']
    geneset = gseapy_module.generate_lr_geneset(lr_list, complex_sep='&', lr_sep='^',
                                                pathway_per_gene=pathway_per_gene,
                                                min_pathways=0, max_pathways=10000)
    assert isinstance(geneset, dict)


def test_generate_lr_geneset_with_a_receptor_complex(tiny_gmt):
    '''Every member of a complex must be annotated for the pathway to count, so the
    receptor here keeps only the pathway both of its members share.'''
    pathway_per_gene = gseapy_module.load_gmt(tiny_gmt, backup_url=None)
    lr_list = ['Protein-A^Protein-B&Protein-C']
    geneset = gseapy_module.generate_lr_geneset(lr_list, complex_sep='&', lr_sep='^',
                                                pathway_per_gene=pathway_per_gene,
                                                min_pathways=0, max_pathways=10000)
    # Protein-B is in pathways ONE and TWO, Protein-C only in ONE, and the ligand
    # Protein-A only in ONE, so ONE is the single shared pathway.
    assert geneset == {'TOYDB_PATHWAY_ONE': {'Protein-A^Protein-B&Protein-C'}}


def test_generate_lr_geneset_loads_the_annotation_when_none_is_given(tiny_gmt, monkeypatch):
    '''Without `pathway_per_gene` the annotation is loaded from the pathway database.'''
    seen = {}
    real_load_gmt = gseapy_module.load_gmt

    def fake_load_gmt(filename, backup_url=None, readable_name=False):
        seen['filename'] = filename
        seen['backup_url'] = backup_url
        return real_load_gmt(tiny_gmt, backup_url=None)

    monkeypatch.setattr(gseapy_module, 'load_gmt', fake_load_gmt)
    geneset = gseapy_module.generate_lr_geneset(['Protein-A^Protein-B'], lr_sep='^',
                                                organism='human', pathwaydb='KEGG',
                                                min_pathways=0, max_pathways=10000)
    expected = gseapy_module.PATHWAY_DATA['human']['KEGG']
    assert seen['filename'] == expected['filename']
    assert seen['backup_url'] == expected['backup_url']
    assert geneset


def test_generate_lr_geneset_looks_for_the_gmt_in_the_output_folder(tiny_gmt, monkeypatch,
                                                                    tmp_path):
    seen = {}
    real_load_gmt = gseapy_module.load_gmt

    def fake_load_gmt(filename, backup_url=None, readable_name=False):
        seen['filename'] = filename
        return real_load_gmt(tiny_gmt, backup_url=None)

    monkeypatch.setattr(gseapy_module, 'load_gmt', fake_load_gmt)
    gseapy_module.generate_lr_geneset(['Protein-A^Protein-B'], lr_sep='^',
                                      output_folder=str(tmp_path),
                                      min_pathways=0, max_pathways=10000)
    assert seen['filename'].startswith(str(tmp_path))


def test_generate_lr_geneset_respects_max_pathways(tiny_gmt):
    pathway_per_gene = gseapy_module.load_gmt(tiny_gmt, backup_url=None)
    geneset = gseapy_module.generate_lr_geneset(['Protein-A^Protein-B'], lr_sep='^',
                                                pathway_per_gene=pathway_per_gene,
                                                min_pathways=0, max_pathways=0)
    assert geneset == {}


# ---------------------------------------------------------------------------------
# load_gmt when the file is not there
#
# The fallback used to print a message and then carry on to `with f:` with `f` never
# assigned, so the caller saw an UnboundLocalError instead of a usable error.
# ---------------------------------------------------------------------------------

def test_load_gmt_without_a_file_or_a_url(tmp_path):
    missing = tmp_path / 'not-here.gmt'
    with pytest.raises(FileNotFoundError):
        gseapy_module.load_gmt(str(missing), backup_url=None)


def test_load_gmt_downloads_the_file_when_it_is_missing(tmp_path, tiny_gmt, monkeypatch):
    missing = tmp_path / 'downloaded.gmt'
    content = open(tiny_gmt, 'rb').read()

    def fake_download(url, path):
        path.write_bytes(content)

    monkeypatch.setattr(gseapy_module, '_download', fake_download)
    pathway_per_gene = gseapy_module.load_gmt(str(missing),
                                              backup_url='http://toydb.org/toy.gmt')
    assert pathway_per_gene['Protein-A'] == {'TOYDB_PATHWAY_ONE'}


def test_load_gmt_reports_an_invalid_url(tmp_path, monkeypatch):
    def fake_download(url, path):
        raise ValueError('invalid URL')

    monkeypatch.setattr(gseapy_module, '_download', fake_download)
    with pytest.raises(FileNotFoundError, match='not a valid URL'):
        gseapy_module.load_gmt(str(tmp_path / 'missing.gmt'), backup_url='nonsense')


def test_load_gmt_reports_a_download_that_produced_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(gseapy_module, '_download', lambda url, path: None)
    with pytest.raises(FileNotFoundError, match='did not produce'):
        gseapy_module.load_gmt(str(tmp_path / 'missing.gmt'),
                               backup_url='http://toydb.org/toy.gmt')


# ---------------------------------------------------------------------------------
# The database and dependency checks
# ---------------------------------------------------------------------------------

def test_check_pathwaydb_accepts_every_documented_combination():
    for organism, databases in gseapy_module.PATHWAY_DATA.items():
        for pathwaydb in databases:
            gseapy_module._check_pathwaydb(organism, pathwaydb)


def test_check_pathwaydb_rejects_an_unknown_organism():
    with pytest.raises(ValueError, match='organism'):
        gseapy_module._check_pathwaydb('martian', 'GOBP')


def test_check_pathwaydb_rejects_an_unknown_database():
    with pytest.raises(ValueError, match='pathwaydb'):
        gseapy_module._check_pathwaydb('human', 'NotADatabase')


def test_check_if_gseapy_returns_the_module():
    assert gseapy_module._check_if_gseapy().__name__ == 'gseapy'


def test_check_if_gseapy_explains_how_to_install_it(monkeypatch):
    '''gseapy is a hard dependency, so the missing-dependency branch is unreachable
    unless the import is made to fail.'''
    monkeypatch.setitem(sys.modules, 'gseapy', None)
    with pytest.raises(ImportError, match='pip install gseapy'):
        gseapy_module._check_if_gseapy()


# ---------------------------------------------------------------------------------
# _download
# ---------------------------------------------------------------------------------

class _FakeResponse:
    '''Stands in for the object `urlopen` returns: a context manager over bytes.'''

    def __init__(self, payload, fail_after=None):
        self._stream = io.BytesIO(payload)
        self._fail_after = fail_after
        self._reads = 0

    def info(self):
        return {'content-length': str(len(self._stream.getvalue()))}

    def read(self, size):
        self._reads += 1
        if self._fail_after is not None and self._reads > self._fail_after:
            raise IOError('connection dropped')
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_download_writes_the_whole_response(tmp_path, monkeypatch):
    payload = b'PATHWAY\thttp://toydb.org/one\tProtein-A\n' * 500
    monkeypatch.setattr('urllib.request.urlopen',
                        lambda request, **kwargs: _FakeResponse(payload))
    target = tmp_path / 'downloaded.gmt'
    gseapy_module._download('http://toydb.org/toy.gmt', target)
    assert target.read_bytes() == payload


def test_download_removes_a_half_written_file(tmp_path, monkeypatch):
    '''A partial download must not be left behind to be mistaken for a good file.'''
    payload = b'x' * (1024 * 64)
    monkeypatch.setattr('urllib.request.urlopen',
                        lambda request, **kwargs: _FakeResponse(payload, fail_after=1))
    target = tmp_path / 'downloaded.gmt'
    with pytest.raises(IOError):
        gseapy_module._download('http://toydb.org/toy.gmt', target)
    assert not target.exists()
