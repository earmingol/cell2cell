# -*- coding: utf-8 -*-

'''Tests for cell2cell.external.goenrich (vendored ontology parser)'''

import cell2cell as c2c


def test_ontology_parses_a_minimal_obo(tmp_path):
    obo = tmp_path / 'toy.obo'
    obo.write_text('\n'.join([
        'format-version: 1.2',
        '',
        '[Term]',
        'id: GO:0000001',
        'name: toy root',
        'namespace: biological_process',
        '',
        '[Term]',
        'id: GO:0000002',
        'name: toy child',
        'namespace: biological_process',
        'is_a: GO:0000001',
        '',
        '']))
    graph = c2c.external.ontology(str(obo))
    assert 'GO:0000001' in graph.nodes()
    assert 'GO:0000002' in graph.nodes()
    assert graph.has_edge('GO:0000001', 'GO:0000002') or \
           graph.has_edge('GO:0000002', 'GO:0000001')
