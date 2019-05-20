# -*- coding: utf-8 -*-

from __future__ import absolute_import

from cell2cell.preprocessing import ppi, geneontology

### Integrate all datasets
def get_all_ppis(ppi_data, contact_proteins, mediator_proteins=None, interaction_columns=['A', 'B']):

    all_ppis = dict()
    all_ppis['contacts'] = ppi.filter_ppi_network(ppi_data,
                                              contact_proteins,
                                              mediator_proteins=mediator_proteins,
                                              interaction_type='contacts',
                                              interaction_columns=interaction_columns)
    if mediator_proteins is not None:
        all_ppis['mediated'] = ppi.filter_ppi_network(ppi_data,
                                                  contact_proteins,
                                                  mediator_proteins=mediator_proteins,
                                                  interaction_type='mediated',
                                                  interaction_columns=interaction_columns)

        all_ppis['combined'] = ppi.filter_ppi_network(ppi_data,
                                                  contact_proteins,
                                                  mediator_proteins=mediator_proteins,
                                                  interaction_type='combined',
                                                  interaction_columns=interaction_columns)
    # else:
    #     columns = interaction_columns + ['score']
    #     all_ppis['mediated'] = pd.DataFrame(columns=columns)
    #     all_ppis['combined'] = pd.DataFrame(columns=columns)
    # This is omitted because other functions use all_ppis.values() instead of checking all types of interactions.
    return all_ppis


def ppis_from_goterms(ppi_data, go_annotations, go_terms, contact_go_terms, mediator_go_terms=None, use_children=True,
                      go_header='GO', gene_header='Gene', interaction_columns=['A', 'B'], verbose=True):




    if use_children == True:
        contact_proteins = geneontology.get_genes_from_parent_go_terms(go_annotations,
                                                                       go_terms,
                                                                       contact_go_terms,
                                                                       go_header=go_header,
                                                                       gene_header=gene_header,
                                                                       verbose=verbose)

        mediator_proteins = geneontology.get_genes_from_parent_go_terms(go_annotations,
                                                                        go_terms,
                                                                        mediator_go_terms,
                                                                        go_header=go_header,
                                                                        gene_header=gene_header,
                                                                        verbose=verbose)
    else:
        contact_proteins = geneontology.get_genes_from_go_terms(go_annotations,
                                                                contact_go_terms,
                                                                go_header=go_header,
                                                                gene_header=gene_header)

        mediator_proteins = geneontology.get_genes_from_go_terms(go_annotations,
                                                                 mediator_go_terms,
                                                                 go_header=go_header,
                                                                 gene_header=gene_header)

    ppi_dict = get_all_ppis(ppi_data,
                            contact_proteins,
                            mediator_proteins,
                            interaction_columns=interaction_columns)

    return ppi_dict

