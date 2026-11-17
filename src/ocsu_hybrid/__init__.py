"""OCSU-Hybrid: modular optical chemical structure understanding.

The package implements the OCSR-based route of the OCSU paper:

    image -> MolScribe -> SMILES
                         |-> ConvNeXt-T functional groups
                         |-> PubChem IUPAC name
                         `-> BioT5+ description
"""

__version__ = "0.1.0"
