import scipy.stats as st


def hypergeom_representation(sample_size, class_in_sample, population_size, class_in_population):
    '''
    Performs an analysis of enrichment/depletion based on observation
    in a sample. It computes a p-value given a hypergeometric
    distribution.

    Parameters
    ----------
    sample_size : int
        Size of the sample obtained or number of elements
        obtained from the analysis.

    class_in_sample : int
        Number of elements of a given class that are
        contained in the sample. This is the class to be tested.

    population_size : int
        Size of the sampling space. That is, the total number
        of possible elements to be chosen when sampling.

    class_in_population : int
        Number of elements of a given class that are contained
        in the population. This is the class to be tested.

    Returns
    -------
    p_vals : tuple
        A tuple containing the p-values for depletion and
        enrichment analysis, respectively.
    '''
    # Computing the number of elements that are not in the same class
    nonclass_in_sample = sample_size - class_in_sample
    nonclass_in_population = population_size - class_in_population

    # Remaining elements in population after sampling
    rem_class = class_in_population - class_in_sample
    rem_nonclass = nonclass_in_population - nonclass_in_sample

    # Depletion Analysis
    depletion_hyp_p_val = st.hypergeom.cdf(class_in_sample, population_size, class_in_population, sample_size)

    # Enrichment Analysis
    enrichment_hyp_p_val = 1.0 - st.hypergeom.cdf(class_in_sample - 1.0, population_size, class_in_population,
                                                  sample_size)

    p_vals = (depletion_hyp_p_val, enrichment_hyp_p_val)
    return p_vals


def fisher_representation(sample_size, class_in_sample, population_size, class_in_population):
    '''
    Performs an analysis of enrichment/depletion based on observation
    in a sample. It computes a p-value given a fisher exact test.

    Parameters
    ----------
    sample_size : int
        Size of the sample obtained or number of elements
        obtained from the analysis.

    class_in_sample : int
        Number of elements of a given class that are
        contained in the sample. This is the class to be tested.

    population_size : int
        Size of the sampling space. That is, the total number
        of possible elements to be chosen when sampling.

    class_in_population : int
        Number of elements of a given class that are contained
        in the population. This is the class to be tested.

    Returns
    -------
    results : dict
        A dictionary containing the odd ratios and p-values for
        depletion and enrichment analysis.
    '''
    # Computing the number of elements that are not in the same class
    nonclass_in_sample = sample_size - class_in_sample
    nonclass_in_population = population_size - class_in_population

    # Remaining elements in population after sampling
    rem_class = class_in_population - class_in_sample
    rem_nonclass = nonclass_in_population - nonclass_in_sample

    # Depletion Analysis
    depletion_odds, depletion_fisher_p_val = st.fisher_exact([[class_in_sample, rem_class],
                                                              [nonclass_in_sample, rem_nonclass]],
                                                             alternative='less')

    # Enrichment Analysis
    enrichment_odds, enrichment_fisher_p_val = st.fisher_exact([[class_in_sample, rem_class],
                                                                [nonclass_in_sample, rem_nonclass]],
                                                               alternative='greater')

    p_vals = (depletion_fisher_p_val, enrichment_fisher_p_val)
    odds = (depletion_odds, enrichment_odds)
    results = {'pval' : p_vals,
               'odds' : odds,
              }
    return results