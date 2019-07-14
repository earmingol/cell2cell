import numpy as np
import pandas as pd


def generate_toy_ppi():
    data = np.asarray([['Protein-A', 'Protein-B'],
                       ['Protein-B', 'Protein-C'],
                       ['Protein-C', 'Protein-A'],
                       ['Protein-B', 'Protein-B'],
                       ['Protein-B', 'Protein-A']
                       ])
    ppi = pd.DataFrame(data, columns=['A', 'B'])
    ppi = ppi.assign(score=1.0)
    return ppi