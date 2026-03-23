import os
import pathlib
import typing as t

import click
import numpy as np
import numpy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
import rdkit
import rdkit.Chem.AllChem
from rdkit import Chem
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import visual_graph_datasets.typing as tc

# processing
from visual_graph_datasets.processing import *
from visual_graph_datasets.processing.base import *
from visual_graph_datasets.processing.colors import *
from visual_graph_datasets.processing.molecules import *

# visualization
from visual_graph_datasets.visualization import *
from visual_graph_datasets.visualization.base import *
from visual_graph_datasets.visualization.colors import *
from visual_graph_datasets.visualization.molecules import *

PATH = pathlib.Path(__file__).parent.absolute()

# -- custom imports --
"""
One way in which this class will be used is by copying its entire source code into a separate
python module, which will then be shipped with each visual graph dataset as a standalone input
processing functionality.

All the code of a class can easily be extracted and copied into a template using the "inspect"
module, but it may need to use external imports which are not present in the template by default.
This is the reason for this method.

Within this method all necessary imports for the class to work properly should be defined. The code
in this method will then be extracted and added to the top of the templated module in the imports
section.
"""
pass
# --


# -- The following class was dynamically inserted --
class MoleculeProcessing(ProcessingBase):
    """
    This class is used for the processing of the special "Molecule Graph" graph type. A molecule graph consists of 
    mutliple atoms (nodes) which are connected by different types of bonds (edges). The class provides methods to process 
    the domain string representation of molecule graphs (SMILES strings) into the graph dict representation. The class also 
    provides methods to create a new visual graph dataset element based on the given molecule graph.
    
    **Domain Representation**
    
    The domain specific representation of a molecule graph is called SMILES. It is a string representation of the
    molecular graph. SMILES is an established format for representing molecules and is used by many cheminformatics
    software packages. Characters in this string represent the different atom types while special characters such as 
    brackets and numbers represent the connections between the atoms. The SMILES string can be processed into a graph
    dict representation by using the ``process`` method.
    
    Examples:
    
    The string "C1=CC=C(N)C=C1" is a benzene ring to which a amine group is attached.
    
    **Graph Dict Representation**
    
    A domain-specific SMILES representation of a color graph can be processed into a graph dict representation 
    by using the ``process`` method. The graph dict representation is a dictionary which contains the full graph 
    data of the molecular graph.
    
    .. code-block:: python
    
        smiles = "C1=CC=C(N)C=C1"
        processing = MoleculeProcessing()
        graph = processing.process(smiles)
        print(graph)
    
    **Visualization**
    
    A molecular graph can be visualized using the ``visualize`` method (numpy array) or alternatively the 
    ``visualize_as_figure`` method (matplotlib figure). The visualization will result in an image with the given 
    width and height in pixels.
    
    .. code-block:: python
    
        smiles = "C1=CC=C(N)C=C1"
        processing = MoleculeProcessing()
        fig = processing.visualize_as_figure(smiles, width=1000, height=1000)
        plt.show()
        
    All the visualizations are created with the RDKit cheminformatics library - specificall the RDKit MolDraw2DSVG 
    functionality. The visualizations are created as SVG strings which are then converted into a numpy array or a
    matplotlib figure.
    
    **Customization**
    
    Generally, the most basic information that is encoded in a molecular graph is a one-hot encoding of the atom type 
    as part of the node attributes and an encoding of the bond type as part of the bond attributes. However, there are 
    many more possible bits of information that can be included for both the nodes and the edges. One might want to 
    customize the exact information that is encoded for a specific application. This can be easily done by subclassing
    the MoleculeProcessing class and overwriting the ``node_attribute_map`` and ``edge_attribute_map`` class variables.
    
    These class variables are dictionaries which map the names of the node and edge attributes to a dictionary which
    contains the callback function which is used to extract the attribute value from the molecule graph. The callback
    function is a function which takes the molecule graph and the atom or bond object as input and returns the value
    of the attribute. The callback function can be any function which returns a list of floats. The list of floats
    will be the attribute vector of the node or edge in the graph dict representation.    
    """

    # This is the descriptive string which will be used for the --help option if the command line interface 
    # for this class is invoked.
    description = (
        'This module exposes commands, which can be used to process domain-specific input data into valid '
        'elements of a visual graph dataset\n\n'
        'In this case, a SMILES string is used as the domain-specific representation of a molecular graph. '
        'Using the commands provided by this module, this smiles string can be converted into a JSON '
        'graph representation or be visualized as a PNG.\n\n'
        'Use the --help options for the various individual commands for more information.'
    )

    node_attribute_map = {
        'symbol': {
            'callback': chem_prop('GetSymbol', OneHotEncoder(
                ['B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl'],
                add_unknown=True,
                dtype=str
            )),
            'description': 'One hot encoding of the atom type',
            'is_type': True,
            'encodes_symbol': True,
        },
        'charge': {
            'callback': chem_prop('GetFormalCharge', list_identity),
            'description': 'Encodes the charge of an atom (whether it is positively charged or negatively)',
            'is_type': True,
            'encodes_charge': True,
        }
    }

    edge_attribute_map = {
        'bond_type': {
            'callback': chem_prop('GetBondType', OneHotEncoder(
                [1, 2, 3, 12],
                add_unknown=False,
                dtype=int,
                # These are the (somewhat) human readable string representations of the bond types
                # which are used for the "edge_bonds" list of the graph representation.
                string_values=['S', 'D', 'T', 'A'] # Single, Double, Triple, Aromatic
            )),
            'description': 'One hot encoding of the bond type',
            'is_type': True,
            'encodes_bond': True 
        }
    }

    graph_attribute_map = {
        'molecular_weight': {
            'callback': chem_descriptor(Chem.Descriptors.ExactMolWt, list_identity),
            'description': 'The molecular weight of the entire molecule'
        }
    }

    # These are simply utility variables. These object will be needed to query the various callbacks which
    # are defined in the dictionaries above. They obviously won't result in a real value, but they are only
    # needed to get *any* result from the callback, because for the purpose of constructing the description
    # map we only need to know the length of the lists which are returned by them, not the content.
    MOCK_MOLECULE = mol_from_smiles('CC')
    MOCK_ATOM = MOCK_MOLECULE.GetAtoms()[0]
    MOCK_BOND = MOCK_MOLECULE.GetBonds()[0]
    
    def __init__(self, *args, ignore_issues: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_issues = ignore_issues
        
        # This will be an array of node attribute vector indices of all those elements in 
        # the node attribute vector which have been annotated with the special "is_type" flag. 
        # This flag determines that these elements are relevant to match node types.
        self.node_type_indices = np.array(self.get_attribute_indices(
            self.node_attribute_map,
            self.MOCK_ATOM,
            lambda data: 'is_type' in data and data['is_type']
        ), dtype=int)
        
        # This will be an array of edge attribute vector indices of all those elements in the 
        # edge attribute vector which have been annotated with the special "is_type" flag. 
        # This flag determines that these elements are relevant to match edge types.
        self.edge_type_indices = np.array(self.get_attribute_indices(
            self.edge_attribute_map,
            self.MOCK_BOND,
            lambda data: 'is_type' in data and data['is_type']
        ), dtype=int)
        
        # Here we search for the entry in the node attribute map which implements the "encodes_symbol" 
        # flag. This signals it being the Encoder which is responsible for the main atom symbol type 
        # encoding. 
        # Ultimately we want to extract that very Encoder object for future use where we want to 
        # use it for decoding numeric vectors back into symbols as well.
        data = self.get_attribute_data(
            self.node_attribute_map,
            lambda data: 'encodes_symbol' in data and data['encodes_symbol']
        )
        try:
            self.symbol_encoder: t.Optional[t.Any] = data['callback'].callback
        except TypeError:
            if not self.ignore_issues:
                raise AssertionError('None of elements defined in node_attribute_map implement the flag'
                                     '"encodes_symbol". Please make sure to add the flag to identify the '
                                     'the main symbol Encoder which will be required for the processing functions.')

        # This is an array which contains all the node attribute vector indices which can be used 
        # to extract the sub vector responsible for encoding the symbol.
        self.symbol_indices = np.array(self.get_attribute_indices(
            self.node_attribute_map,
            self.MOCK_ATOM,
            lambda data: 'encodes_symbol' in data and data['encodes_symbol']
        ), dtype=int)
        
        # 29.01.24
        self.charge_indices: t.List[int] = np.array(self.get_attribute_indices(
            self.node_attribute_map,
            self.MOCK_ATOM,
            lambda data: 'encodes_charge' in data and data['encodes_charge'],
        ))
        
        data = self.get_attribute_data(
            self.edge_attribute_map,
            lambda data: 'encodes_bond' in data and data['encodes_bond'],
        )
        try:
            self.bond_encoder: t.Optional[t.Any] = data['callback'].callback
        except TypeError:
            if not self.ignore_issues:
                raise AssertionError('None of the elements defined in edge_attribute_map implement the flag '
                                     '"encodes_bond". Please make sure to add the flag to identify the '
                                     'the main bond type Encoder which will be required for the processing functions')
                
        self.bond_indices = np.array(self.get_attribute_indices(
            self.edge_attribute_map,
            self.MOCK_BOND,
            lambda data: 'encodes_bond' in data and data['encodes_bond'],
        ))
        
        
    def get_attribute_data(self,
                           attribute_map: t.Dict[str, dict],
                           condition: t.Callable,
                           ) -> dict:
        for name, data in attribute_map.items():
            if condition(data):
                return data
        
    def get_attribute_indices(self,
                              attribute_map: t.Dict[str, dict],
                              element: t.Any,
                              condition: t.Callable
                              ) -> t.List[int]:
        """
        Given given an ``attribute_map`` which describes the attribute extraction from a molecule, an 
        atom object ``element`` and a callable ``condition``, this method returns a list which contains 
        the integer indices of a node attributes vector / elements of the node attribute map for which 
        the given condition is true.
        
        If the given condition cannot be found at all, this method will return an empty list.
        
        :param attribute_map: The node attribute map which to search.
        :param element: An Atom or Bond ojbect for which the search be executed. Usually MOCK_ATOM
        :param condition: A callable function which receives the ``data`` dict element of the attribute map 
            as a parameter and is supposed to return a boolean value that determines whether that dict 
            fullfills the desired condition.
        
        :returns: A list of integer indices
        """
        indices = []
        index = 0
        for name, data in attribute_map.items():
            callback = data['callback']
            value: list = self.apply_callback(callback, self.MOCK_MOLECULE, element)
            for _ in value:
                if condition(data):
                    indices.append(index)
                index += 1
                
        return indices
    
    def node_match(self, node_attributes_1, node_attributes_2):
        return np.isclose(
            node_attributes_1[self.node_type_indices], 
            node_attributes_2[self.node_type_indices],
        ).all()

    def edge_match(self, edge_attributes_1, edge_attributes_2):
        return np.isclose(
            edge_attributes_1[self.edge_type_indices],
            edge_attributes_2[self.edge_type_indices],
        ).all()
    
    def contains(self,
                 graph: tv.GraphDict,
                 subgraph: tv.GraphDict,
                 check_edges: bool = False,
                 ) -> bool:
        
        graph_nx: nx.Graph = nx_from_graph(graph)
        subgraph_nx: nx.Graph = nx_from_graph(subgraph)
    
        if check_edges:
            edge_match = lambda a, b: self.edge_match(a['edge_attributes'], b['edge_attributes'])
        else:
            edge_match = None
    
        matcher = nx.isomorphism.GraphMatcher(
            graph_nx, subgraph_nx,
            node_match=lambda a, b: self.node_match(a['node_attributes'], b['node_attributes']),
            edge_match=edge_match,
        )
        
        return matcher.subgraph_is_isomorphic()
    
    def extract(self,
                graph: tv.GraphDict,
                mask: np.ndarray,
                clear_aromaticity: bool = True,
                process_kwargs: dict = {},
                unprocess_kwargs: dict = {},
                ) -> t.Tuple[tv.DomainRepr, tv.GraphDict]:
        
        return super().extract(
            graph=graph,
            mask=mask,
            process_kwargs={
                **process_kwargs,
            },
            unprocess_kwargs={
                'clear_aromaticity': clear_aromaticity,
                **unprocess_kwargs,
            }
        )
    
    def unprocess(self,
                  graph: tv.GraphDict,
                  clear_aromaticity: bool = False,
                  **kwargs
                  ) -> tv.DomainRepr:
        """
        Given the ``graph`` dict representation of a molecular graph, this method will transform that graph 
        back into it's domain representation which in this case is a SMILES string.
        
        the aromaticity problem for fragments
        -------------------------------------
        
        This method should also work for molecular fragements, so graph dicts which don't necessarily describe 
        a complete molecule but rather only parts of it that were extracted from other larger molecules. This 
        can cause a problem when atoms were extracted from an aromatic ring but now in the extracted form they 
        are no longer part of a valid aromatic ring. These kinds of smiles cannot be turned back into a Mol 
        object successfully.
        In such cases the ``clear_aromaticity`` flag of this method has to be used, which will erase the 
        aromatic flags such that the resulting SMILES is still somewhat valid, although the molecule which 
        can then be reconstructed from that smiles is not in itself valid!
        
        :param graph: The graph dict to be converted to smiles
        :param clear_aromaticity: If set, this will forcefully clear the aromaticity flags of the molecule  
            which will result in a valid molecular representation as far as RDKit is concerned but not  
            in the chemical sense.
        
        :returns: The SMILES string corresponding to the graph dict
        """
        
        # For molecular graphs the domain representation is SMILES strings. For SMILES strings we need 
        # to rely on the conversion functionality implemented in RDKit and that in turn can perform the 
        # conversion when provided with a Mol object. So what we have to do here is to iteratively 
        # construct such a Mol object from the given graph dict.
        
        mol = Chem.RWMol()
        for node_index in graph['node_indices']:
            node_attributes = graph['node_attributes'][node_index]
            symbol = self.symbol_encoder.decode(node_attributes[self.symbol_indices])
            atom = Chem.AtomFromSmarts(symbol)
            
            if self.charge_indices:
                charge = int(node_attributes[self.charge_indices][0])
                atom.SetFormalCharge(charge)
            
            mol.AddAtom(atom)
        
        for edge_index, (i, j) in enumerate(graph['edge_indices']):
            i, j = int(i), int(j)
            edge_attributes = graph['edge_attributes'][edge_index]
            bond_type = self.bond_encoder.decode(edge_attributes[self.bond_indices])
            # The decoder here only returns the integer representation of the bond type, but the signature 
            # of the AddBond method REALLY wants that to be wrapped as a BondType object...
            bond_type = Chem.BondType(bond_type)
            if not mol.GetBondBetweenAtoms(i, j):
                mol.AddBond(i, j, bond_type)
            
        # If we only extract a sub graph, aromatic bond types will probably cause a problem since at that 
        # point they are no longer part of a valid ring. In that case we need to manually set all the aromatic 
        # flags to false here to fix that.
        # Note: This does not result in a canonical SMILES then in the end!
        if clear_aromaticity:
            for atom in mol.GetAtoms():
                atom.SetIsAromatic(False)

        # The previous RWMol object is a special kind of *mutable* mol object and here we need to 
        # convert that into the regular mol object.
        mol = mol.GetMol()
        return Chem.MolToSmiles(mol)
    
    def apply_callback(self, 
                       callback: t.Callable, 
                       mol: Chem.Mol, 
                       element: t.Any
                       ) -> t.List[float]:
        """
        Given a ``callback`` function, the base ``mol`` molecule object and the ``element`` - an atom or a bond 
        object - on which to apply that callback, this method will apply the callback in the correct manner and 
        return the return value of that callback. 

        The seemingly unnecessary wrapping of this functionality in it's own method here is necessary because the 
        application of the callback is not necessary due to some special rules whic should not be scattered 
        throughout the rest of the code in this class.

        :param callback: _description_
        :type callback: t.Callable
        :param mol: _description_
        :type mol: Chem.Mol
        :param element: _description_
        :type element: t.Any
        
        :return: _description_
        :rtype: t.List[float]
        """
        # Most callbacks are rather simple and only operate on the basis of the element (atom / bond) itself 
        # but there are other - more advanced - callbacks which need the context of the whole molecule for 
        # some more fancy computations. These can be detected by an additional property that was attached to 
        # them. In those cases the signature of the callback is different which is why we need to check this.
        # I know that this is essentially bad design, but the necessity crept up only later and at that 
        # point I needed to maintain back-comp and couldn't generally change the signature.
        if hasattr(callback, 'requires_molecule') and getattr(callback, 'requires_molecule'):
            value = callback(mol, element)
        else:
            value = callback(element)
        
        return value

    def process(self,
                value: str,
                double_edges_undirected: bool = True,
                use_node_coordinates: bool = False,
                node_coordinates: t.Optional[np.ndarray] = None,
                graph_labels: list = [],
                ) -> dict:
        """
        Converts SMILES string into graph representation.

        This command processes the SMILES string and creates the full graph representation of the
        corresponding molecular graph. The node and edge feature vectors will be created in the exact
        same manner as the rest of the dataset, making the resulting representation compatible with
        any model trained on the original VGD.

        This command outputs the JSON representation of the graph dictionary representation to the console.

        :param value: The SMILES string of the molecule to be converted
        :param double_edges_undirected: A boolean flag of whether to represent an edge as two undirected
            edges
        :param use_node_coordinates: A boolean flag of whether to include 3D node_coordinates into the
            graph representation. These would be created by using an RDKit conformer. However, this
            process can fail.
        :param graph_labels: A list containing the various ground truth labels to be additionally associated
            with the element
        """
        # 01.06.23 - When working with the counterfactuals I have noticed that there is a problem where
        # it is not easily possible to maintain the original molecules atom indices through a SMILES
        # conversion, which seriously messes with the localization of which parts were modified.
        # So as a solution we introduce here the option to generate the graph based on a Mol object
        # directly instead of having to use the SMILES every time. This could potentially be slightly
        # more efficient as well.
        if isinstance(value, (Chem.Mol, Chem.RWMol)):
            mol = value
            smiles = Chem.MolToSmiles(mol)
        elif isinstance(value, str):
            smiles = value
            mol = Chem.MolFromSmiles(smiles)
        else:
            raise TypeError('The value argument must be either a SMILES string or a RDKit Mol object')

        atoms = mol.GetAtoms()
        # First of all we iterate over all the atoms in the molecule and apply all the callback
        # functions on the atom objects which then calculate the actual attribute values for the final
        # node attribute vector.
        node_indices: t.List[int] = []
        node_attributes: t.List[t.List[float]] = []
        # Here we want to maintain a list of the the string atom symbols for each of the atom nodes as 
        # we process the molecule graph structure.
        # The goal is to have some kind of information such that a human could reconstruct 
        # the molecule from the graph structure later on as well.
        node_atoms: t.List[str] = []
        for atom in atoms:
            node_indices.append(atom.GetIdx())

            attributes = []
            # "node_attribute_callbacks" is a dictionary which specifies all the transformation functions
            # that are to be applied on each atom to calculate part of the node feature vector
            for name, data in self.node_attribute_map.items():
                callback: t.Callable[[Chem.Mol, Chem.Atom], list] = data['callback']
                value: list = self.apply_callback(callback, mol, atom)

                attributes += value

            node_attributes.append(attributes)

            # "symbol_encoder" is an Encoder specifically designed to encode the atom symbols into a 
            # one-hot encoded vector. It has the additional method "encode_string" which encodes the 
            # symbol into a human readable string.
            if hasattr(self, 'symbol_encoder') and self.symbol_encoder:
                atom_symbol = self.symbol_encoder.encode_string(atom.GetSymbol())
                node_atoms.append(atom_symbol)

        bonds = mol.GetBonds()
        # Next up is the same with the bonds
        edge_indices: t.List[t.Tuple[int, int]] = []
        edge_attributes: t.List[t.List[float]] = []
        # Here we want to maintain a list of the bond types for each of the edge nodes as we process the
        # molecule graph structure. More specifically, we want to safe a human readable string representation 
        # of the bond type. 
        # The goal is to have some kind of information such that a human could reconstruct 
        # the molecule from the graph structure later on as well.
        edge_bonds: t.List[str] = []
        for bond in bonds:
            i = int(bond.GetBeginAtomIdx())
            j = int(bond.GetEndAtomIdx())

            edge_indices.append([i, j])
            if double_edges_undirected:
                edge_indices.append([j, i])

            attributes = []
            for name, data in self.edge_attribute_map.items():
                callback: t.Callable[[Chem.Bond], list] = data['callback']
                value: list = self.apply_callback(callback, mol, bond)
                attributes += value

            # We have to be careful here to really insert the attributes as often as there are index
            # tuples.
            edge_attributes.append(attributes)
            if double_edges_undirected:
                edge_attributes.append(attributes)
                
            if hasattr(self, 'bond_encoder') and self.bond_encoder:
                bond_type: str = self.bond_encoder.encode_string(bond.GetBondType())
                edge_bonds.append(bond_type)
                if double_edges_undirected:
                    edge_bonds.append(bond_type)

        # Then there is also the option to add global graph attributes. The callbacks for this kind of
        # attribute take the entire molecule object as an argument rather than just atom or bond
        graph_attributes = []
        for name, data in self.graph_attribute_map.items():
            callback: t.Callable[[Chem.Mol], list] = data['callback']
            value: list = callback(mol)
            graph_attributes += value

        # Now we can construct the preliminary graph dict representation. All of these properties form the
        # core of the graph dict - they always have to be present. All the following code after that is
        # optional additions which can be added to support certain additional features
        graph: tc.GraphDict = {
            'node_indices':         np.array(node_indices, dtype=int),
            'node_attributes':      np.array(node_attributes, dtype=float),
            'edge_indices':         np.array(edge_indices, dtype=int),
            'edge_attributes':      np.array(edge_attributes, dtype=float),
            'graph_attributes':     np.array(graph_attributes, dtype=float),
            'graph_labels':         np.array(graph_labels),
            # 14.02.24 - Initially I wanted to avoid addding the string graph representation to the graph 
            # dictionary itself, because I kind of wanted all of the values to be numpy array. However, I 
            # have now hit a problem where this is pretty much necessary.
            'graph_repr':           smiles,
        }
        
        # 28.10.24 - Here we save some domain-specific additional information on the node and edge level.
        # More specifically these are lists of human readable strings which contain the atom symbols and
        # bond types respectively. The goal of this additional information is to provide a way for a human 
        # to easily reconstruct the molecule from the graph representation as well.
        if node_atoms:
            graph['node_atoms'] = np.array(node_atoms, dtype=str)
        if edge_bonds:
            graph['edge_bonds'] = np.array(edge_bonds, dtype=str)

        # Optionally, if the flag is set, this will apply a conformer on the molecule which will
        # then calculate the 3D coordinates of each atom in space.
        if use_node_coordinates:
            
            try:
                # # https://sourceforge.net/p/rdkit/mailman/message/33386856/
                # try:
                #     rdkit.Chem.AllChem.EmbedMolecule(mol)
                #     rdkit.Chem.AllChem.MMFFOptimizeMolecule(mol)
                # except:
                #     rdkit.Chem.AllChem.EmbedMolecule(mol, useRandomCoords=True)
                #     rdkit.Chem.AllChem.UFFOptimizeMolecule(mol)
                    
                # 05.02.2024
                # Previously we tried to do the MMFF optimization here, but now we simply commit to the UFF 
                # optimization which is a bit faster and should be sufficient for our purposes.
                mol = Chem.AddHs(mol)
                params = AllChem.ETKDGv3()
                params.useBasicKnowledge = True
                params.useRandomCoords = True
                params.randomSeed = 0xF00D
                AllChem.EmbedMolecule(mol, params)
                #AllChem.UFFOptimizeMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)

                conformer = mol.GetConformers()[0]
                node_coordinates = np.array(conformer.GetPositions())
                graph['node_coordinates'] = node_coordinates

                # Now that we have the 3D coordinates for every atom, we can also calculate the length of all
                # the bonds from that!
                graph['edge_lengths'] = self.calculate_edge_lengths(node_coordinates, graph['edge_indices'])

            except Exception as exc:
                raise ProcessingError(f'Cannot calculate node_coordinates for the given '
                                      f'molecule with smiles code: {smiles}')
                
        # 16.06.2024
        # We want to provide the option to add the node coordinates to the graph dict representation as an 
        # external argument. This is because we also want to support datasets which do contain the atom 
        # coordinates as part of the dataset itself!
        # If that argument is given we will override the calculated node coordinates (if they exist).
        if node_coordinates is not None:
            
            graph['node_coordinates'] = node_coordinates
            graph['edge_lengths'] = self.calculate_edge_lengths(node_coordinates, graph['edge_indices'])

        return graph

    def visualize_as_figure(self,
                            value: str,
                            width: int,
                            height: int,
                            additional_returns: Optional[dict] = None,
                            reference_mol: Optional[Chem.Mol] = None,
                            line_width: float = 5.0,
                            **kwargs,
                            ) -> t.Tuple[plt.Figure, np.ndarray]:
        """
        The normal "visualize" method has to return a numpy array representation of the image. While that
        is a decent choice for printing it to the console, it is not a good choice for direct API usage.
        This method will instead return the visualization as a matplotlib Figure object.

        This method should preferably used when directly interacting with the processing functionality
        with code.
        
        :param value: The domain specific string representation of the graph
        :param width: The width of the resulting visualization image in pixels.
        :param height: The height of the resulting visualization image in pixels.
        :param additional_returns: Optional. A dict object can be passed to this method to collect additional 
            returns of this method. In this specific case, this dict will contain the following keys:
            "svg_string" - the raw string representation of the original svg string that created the molecule 
            visualization
            
        :returns: a tuple where the first value is a plt.Figure object containing the visualization of the 
            molecular graph and the second element is a numpy array of shape (V, 2) where V is the number of 
            nodes in the graph and which assigns the 2D pixel coordinates of each of the nodes.
        """
        if isinstance(value, (Chem.Mol, Chem.RWMol)):
            mol = value
            smiles = Chem.MolToSmiles(mol)
        elif isinstance(value, str):
            smiles = value
            mol = Chem.MolFromSmiles(smiles)
        
        fig, ax = create_frameless_figure(width=width, height=height)
        node_positions, svg_string = visualize_molecular_graph_from_mol(
            ax=ax,
            mol=mol,
            image_width=width,
            image_height=height,
            line_width=line_width,
            # 29.11.24 - This can optionally be used to control the orientation of the molecule in the 
            # created visualization. The main molecule will be aligned in the same way the reference 
            # molecule. This is particularly useful when visualizing many molecules with the same backbone 
            # to make sure that they have the same principal orientation.
            reference_mol=reference_mol,
        )
        
        # The "node_positions" which are returned by the above function are values within the axes object
        # coordinate system. Using the following piece of code we transform these into the actual pixel
        # coordinates of the figure image.
        node_positions = [[int(v) for v in ax.transData.transform((x, y))]
                          for x, y in node_positions]
        node_positions = np.array(node_positions)

        # 24.10.23 - made the additional returns optional and checking for the None state here because that 
        # is better practice than having a mutable dictionary instance as a default value of a method argument.
        if additional_returns is not None:
            additional_returns['svg_string'] = svg_string

        # 24.10.23 - Desperate attempt to somehow fix the performance degradation of this method.
        del ax, mol

        return fig, node_positions

    def visualize(self,
                  value: str,
                  width: int = 1000,
                  height: int = 1000,
                  **kwargs,
                  ) -> np.ndarray:
        """
        Creates a visualization of the given SMILES molecule.

        This command used RDKit to generate a visual representation of the molecular graph corresponding
        to the given SMILES string. This will result in an colored image of the given dimensions WIDTH
        and HEIGHT.

        This command outputs the JSON representation of the RGB image array to the console. This array will
        have the shape (width, height, 3).
        """
        fig, _ = self.visualize_as_figure(
            value=value,
            width=width,
            height=height,
        )

        # This section turns the image which is currently a matplotlib figure object into a numpy array
        # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
        canvas = FigureCanvas(fig)
        canvas.draw()
        array = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        array = array.reshape((height, width, 4))[:, :, :3]  # Remove alpha channel

        plt.close(fig)

        return array

    def create(self,
               value: str,
               index: str = '0',
               name: str = 'molecule',
               double_edges_undirected: bool = True,
               use_node_coordinates: bool = False,
               additional_metadata: dict = {},
               additional_graph_data: dict = {},
               width: int = 1000,
               height: int = 1000,
               create_svg: bool = False,
               output_path: str = os.getcwd(),
               writer: t.Optional[VisualGraphDatasetWriter] = None,
               node_coordinates: t.Optional[np.ndarray] = None,
               **kwargs,
               ):
        """
        Creates VGD element file representations (PNG & JSON).

        This command will create the VGD file representations corresponding to the molecular graph given
        by the SMILES string. This representation includes two files by default: (1) A JSON file which
        contains the full graph representation of the molecular graph and additional metadata. (2) A PNG
        file containing a visual representation of the molecular graph. Both file are the same

        This command will not output anything to the console. Instead, it will create new files within
        the given OUTPUT_PATH folder. The files will have the name which is simply a number that is
        specified with the INDEX option.
        """
        if isinstance(value, (Chem.Mol, Chem.RWMol)):
            mol = value
            smiles = Chem.MolToSmiles(mol)
        elif isinstance(value, str):
            smiles = value
            mol = Chem.MolFromSmiles(smiles)
        
        g = self.process(
            value=value,
            double_edges_undirected=double_edges_undirected,
            use_node_coordinates=use_node_coordinates,
            node_coordinates=node_coordinates,
        )
        g.update(additional_graph_data)
        fig, node_positions = self.visualize_as_figure(
            value=value,
            width=width,
            height=height,
        )
        # This is important! we need to add the node positions to the graph representation to enable the
        # visualizations later on. This array contains the pixel coordinates within the image for each of
        # the nodes
        g['node_positions'] = node_positions

        metadata = {
            'index':    int(index),
            'name':     name,
            'smiles':   smiles,
            'repr':     smiles,
            'graph':    g,
            **additional_metadata,
        }

        # 02.06.2023 - I have added the option to specify a Writer instance with which the elements should
        # actually be saved to the disk. This is optional and if no writer instance is given it will work
        # just as before. But using a writer instance for dataset creation is preferable because it can
        # make use of optimizations while saving the dataset.
        if writer is not None:
            writer.write(
                name=int(index),
                metadata=metadata,
                figure=fig,
            )

        else:
            fig_path = os.path.join(output_path, f'{index}.png')
            self.save_figure(fig, fig_path)
            plt.close(fig)

            metadata_path = os.path.join(output_path, f'{index}.json')
            self.save_metadata(metadata, metadata_path)

        return metadata

    def get_description_map(self) -> dict:
        """
        This method returns a dictionary, which on the top level contains the three keys
        "node_attributes", "edge_attributes" and "graph_attributes". Each of these is a dictionary
        themselves, where the keys are the integer indices of the corresponding attribute vectors of an
        element and the values are strings containing natural language descriptions of the content of each
        index.
        """
        description_map = {
            'node_attributes': {},
            'edge_attributes': {},
            'graph_attributes': {},
        }

        # descriptions for the node attributes
        index = 0
        for name, data in self.node_attribute_map.items():
            description = data['description']
            callback = data['callback']
            value: list = callback(self.MOCK_ATOM)
            for _ in value:
                description_map['node_attributes'][index] = description
                index += 1

        # descriptions for the edge attributes
        index = 0
        for name, data in self.edge_attribute_map.items():
            description = data['description']
            callback = data['callback']
            value: list = callback(self.MOCK_BOND)
            for _ in value:
                description_map['edge_attributes'][index] = description
                index += 1

        
        # descriptions for the graph attributes
        index = 0
        for name, data in self.graph_attribute_map.items():
            description = data['description']
            callback = data['callback']
            value: list = callback(self.MOCK_MOLECULE)
            for _ in value:
                description_map['graph_attributes'][index] = description
                index += 1

        return description_map
    
    def get_num_node_attributes(self) -> int:
        """
        Returns the number of node attributes that are encoded in the node attribute map of this class.
        
        :returns: The number of node attributes
        """
        values = []
        for name, data in self.node_attribute_map.items():
            callback = data['callback']
            values += callback(self.MOCK_ATOM)
            
        return len(values)
    
    def get_num_edge_attributes(self) -> int:
        """
        Returns the number of edge attributes that are encoded in the edge attribute map of this class.
        
        :returns: The number of edge attributes
        """
        values = []
        for name, data in self.edge_attribute_map.items():
            callback = data['callback']
            values += callback(self.MOCK_BOND)
        
        return len(values)

    # -- utils --

    def calculate_edge_lengths(self, 
                               node_coordinates: np.ndarray, 
                               edge_indices: np.ndarray,
                               ) -> np.ndarray:
        """
        Given the array of ``node_coordinates`` with the shape (num_nodes, 3) and the array of the 
        ``edge_indices`` with the shape (num_edges, 2), this method will calculate the length of each
        edge in the graph and return an array of edge lengths with the shape (num_edges, 1).
        
        :param node_coordinates: The array of node coordinates with the shape (num_nodes, 3)
        :param edge_indices: The array of edge indices with the shape (num_edges, 2)
        
        :returns: An array of edge lengths with the shape (num_edges, 1)
        """
        edge_lengths = []
        for i, j in edge_indices:
            c_i = node_coordinates[i]
            c_j = node_coordinates[j]
            length = la.norm(c_i - c_j)
            edge_lengths.append([length])

        return np.array(edge_lengths)

    def save_svg(self, content: str, path: str):
        with open(path, mode='w') as file:
            file.write(content)

# --


# The data element pre-processing capabilities defined in the above class can either be accessed by
# importing this object from this module in other code and using the implementations of the methods
# "process", "visualize" and "create".
# Alternatively this module also acts as a command line tool (see below)
processing = MoleculeProcessing()

if __name__ == '__main__':
    # This class inherits from "click.MultiCommand" which means that it will directly work as a cli
    # entry point when using the __call__ method such as here. This will enable this python module to
    # expose the cli commands defined in the above class when invoking it from the command line.
    processing()