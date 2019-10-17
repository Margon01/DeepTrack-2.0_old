from DeepTrack.Distributions import Distribution
from DeepTrack.Backend.Image import Image
from abc import ABC, abstractmethod
import os
import re
import numpy as np
import copy

'''
    Base feature class. A typical lifecycle of a feature F is

    F.__clear__()       => Clears the internal cache of the feature.
    F.__rupdate__()     => Recursively updates the feature and its parent(s).
    F.__resolve__()     => Resolves the image generated by the feature.
    

    __rupdate__() updates all attributes with a defined __sample__() method.
    this includes the parent of the feature, which is why this will
    reursively update the entire tree.

    __resolve__() generates an image based on the current properties
    defined in the tree.

    __clear__() cleans up the tree after execution. Default behavior is
    to recursively clear the cache on each node of the tree.


'''
class Feature(ABC):

    __name__ = "Unnamed feature"

    '''
        All keyword arguments passed to the base Feature class will be 
        wrapped as a Distribution, as such randomized during a sampling
        step. 
    '''
    def __init__(self, **kwargs):
        properties = getattr(self, "__properties__", {})
        for key, value in kwargs.items():
            properties[key] = Distribution(value)  
        self.__properties__ = properties

    '''
        When a feature is sampled, each attribute that is either a Distribution
        or a feature is in turn sampled. This randomizes the feature, and recursively
        samples its parent.
    '''

    def get_properties(self):
        props = {}
        for key, value in self.__properties__.items():
            try: 
                props[key] = value.value
            except AttributeError:
                props[key] = value
        return props 
    
    def get_property(self, key, default=None):
        try: 
            return self.__properties__[key].value
        except AttributeError:
            return self.__properties__[key]
        except KeyError:
            return default
    
    def set_property(self, key, value):
        try: 
            self.__properties__[key].value = key
        except AttributeError:
            self.__properties__[key] = key

    def getRoot(self):
        if hasattr(self, "parent"):
            return self.parent.getRoot()
        else:
            return self

    def setParent(self, Feature):
        if hasattr(self, "parent"):
            G = Group(self)
            G = G.setParent(Feature)
            return G
        else:            
            self.parent = Feature
            return self

    def __rupdate__(self, history):
        self.__update__(history)
        if hasattr(self, "parent"):
            self.parent.__rupdate__(history)

    '''
        Updates the state of all properties.
    '''
    def __update__(self, history):
        if self not in history:
            history.append(self)
            for val in self.__properties__.values():
                val.__update__(history)

    def __input_shape__(self, shape):
        return shape

    '''
        Arithmetic operator overload. Creates copies of objects.
    '''
    def __add__(self, other):
        o_copy = copy.deepcopy(other)
        o_copy = o_copy.setParent(self)
        return o_copy

    def __radd__(self, other): 
        self_copy = copy.deepcopy(self)
        self_copy = self_copy.setParent(other)
        return self_copy

    def __mul__(self, other):
        G = Group(copy.deepcopy(self))
        G.probability = other
        return G

    __rmul__ = __mul__


    '''
    Recursively resolves the feature feature tree backwards, starting at this node. 
    Each recursive step checks the content of "cache" to check if the node has already 
    been calculated. This allows for a very efficient evaluation of more complex structures
    with several outputs.

    The function checks its parent property. For None values, the node is seen as input, 
    and creates a new image. For ndarrays and Images, those values are copied over. For
    Features, the image is calculated by recursivelt calling the __resolve__ method on the 
    parent.

    INPUTS:
        shape:      requested image shape
    
    OUTPUTS:
        Image: An Image instance.
    '''
    def __resolve__(self, shape, **kwargs):

        cache = getattr(self, "cache", None)
        if cache is not None:
            return cache

        parent = getattr(self, "parent", None)
        # If parent does not exist, initiate with zeros
        if parent is None:
            image = Image(np.zeros(self.__input_shape__(shape)))
        # If parent is ndarray, set as ndarray
        elif isinstance(parent, np.ndarray):
            image = Image(parent)
        # If parent is image, set as Image
        elif isinstance(parent, Image):
            image = parent
        # If parent is Feature, retrieve it
        elif isinstance(parent, Feature):
            image = parent.__resolve__(shape, **kwargs)
        # else, pray
        else:
            image = parent
        
        # Get probability of draw
        p = getattr(self, "probability", 1)
        if np.random.rand() <= p:
            properties = self.get_properties()
            # TODO: find a better way to pass information between features
            image = self.get(shape, image, **properties, **kwargs)
            properties["name"] = self.__name__
            image.append(properties)
        
        # Store to cache
        self.cache = copy.deepcopy(image)
        return image
    

    '''
    Recursively clears the __cache property. Should be on each output node between each call to __resolve__
    to ensure a correct initial state.
    '''
    def __clear__(self):
        self.cache = None
        for val in self.__properties__.values():
            try:
                val.__clear__()
            except AttributeError:
                pass
        for val in self.__dict__.values():
            try:
                val.__clear__()
            except AttributeError:
                pass

    @abstractmethod
    def get(self, shape, Image, Optics=None):
        pass


'''
    Allows a tree of features to be seen as a whole.    
'''
class Group(Feature):
    __name__ = "Group"
    def __init__(self, Features):
        self.__properties__ = {"group": Features}
        super().__init__()

    def __input_shape__(self,shape):
        return self.get_property("group").__input_shape__(shape)

    def get(self, shape, Image, group=None, **kwargs):
        return group.__resolve__(shape, **kwargs)

    # TODO: What if already has parent? Possible?
    def setParent(self, Feature):
        self.parent = Feature
        self.get_property("group").getRoot().setParent(Feature)
        return self


class Load(Feature):
    __name__ = "Load"
    def __init__(self,
                    path):
        self.path = path
        self.__properties__ = {"path": path}

        # Initiates the iterator
        self.iter = next(self)
    
    def get(self, shape, image, **kwargs):
        return self.res
    
    def __update__(self,history):
        if self not in history:
            history.append(self)
            self.res = next(self.iter)
            super().__update__(history)
    
    def __next__(self):
        while True:
            file = np.random.choice(self.get_files())
            image = np.load(file)
            np.random.shuffle(image)
            for i in range(len(image)):
                yield image[i]

        


    def setParent(self, F):
        raise Exception("The Load class cannot have a parent. For literal addition, use the Add class")

    def get_files(self):
        if os.path.isdir(self.path):
             return [os.path.join(self.path,file) for file in os.listdir(self.path) if os.path.isfile(os.path.join(self.path,file))]
        else:
            dirname = os.path.dirname(self.path)
            files =  os.listdir(dirname)
            pattern = os.path.basename(self.path)
            return [os.path.join(self.path,file) for file in files if os.path.isfile(os.path.join(self.path,file)) and re.match(pattern,file)]
        
class Update(Feature):
    def __init__(rules, **kwargs):
        self.rules = rules
        super().__init__(**kwargs)
    
    def __call__(F):
        return F + self

    def __resolve__(self, shape, **kwargs):
        