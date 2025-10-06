import numpy as np
import math
from scipy.special import gammaln 

class CTW():
    def __init__(self,alphabet_size_y: int = 2, alphabet_size_x: int = None, depth: int = 1, side_info: bool = True):
        #initialize the alphabet sizes
        self.alphabet_size_y = alphabet_size_y
        self.alphabet_size_x = alphabet_size_x

        #set joint to true if an x size was given 
        self.joint = self.alphabet_size_x is not None

        #set the depth
        self.depth = depth

        #set side info if x size was given 
        self.side_info = side_info if self.alphabet_size_x is not None else False

        #set the tree depth (number of symbols in the context) to twice depth if the tree is joint, and a +1 for side info
        tree_depth = 2 * self.depth + 1 if self.side_info else 2 * self.depth if alphabet_size_x is not None else self.depth
        self.tree = CTree(alphabet_size_y=alphabet_size_y,alphabet_size_x=alphabet_size_x,depth=tree_depth,side_info=side_info)
    
    def update_tree(self,context,symbol, amount=1):
        #update the tree with a context and symbol, returning the pw of the root
        return self.tree.update_nodes(context, symbol, amount)
    
    def get_distribution(self, context):
        node = self.tree.get_context_node(context)
        return node.kts_pred

class CTree():
    def __init__(self, alphabet_size_y: int = None, alphabet_size_x: int = None, depth: int = None, side_info: bool = None):
        #set alphabet sizes, depth, and side info
        self.alphabet_size_y = alphabet_size_y
        self.alphabet_size_x = alphabet_size_x
        self.depth = depth
        self.side_info = side_info

        #calculate the amount of root children based on side info. If side info, the tree starts with Xt and alternates. 
        #if there is no side info, the tree will start with Yt-1. 
        root_children, isY = (self.alphabet_size_x,False) if self.side_info and self.alphabet_size_x is not None else (self.alphabet_size_y,True)
        self.root = Node(context="", len_counts=self.alphabet_size_y,num_children=root_children,alphabet_size=self.alphabet_size_y,isY=isY)

        #initialize a tree with the root and the depth
        self.initialize_tree(self.root, self.depth)

    def initialize_tree(self, node, depth):
        #recursively create the tree
        if depth == 0:
            return 
        
        #for each child of the current node
        for i in range(len(node.children)):
            #add the context which corresponds to the index
            child_context = node.context + [i]

            #the length of counts will always be the y variable
            counts = self.alphabet_size_y

            #alternate isY, if there is no x alphabet then this variable is irrelevant
            isY = not node.isY

            #calculate the number of children based on if the next context is a part of the X or Y alphabet
            child_num_children = (self.alphabet_size_x if not isY and self.alphabet_size_x is not None else self.alphabet_size_y) if depth > 1 else 0
            
            #create the child node
            node.children[i] = Node(context=child_context,len_counts=counts,num_children=child_num_children,alphabet_size=self.alphabet_size_y,parent=node, isY=isY)
            
            #initialize a new tree under the child node with a depth - 1
            self.initialize_tree(node.children[i], depth - 1)
    
    def get_context_node(self, context):
        #find a node based on the context
        #start at the root
        node = self.root

        #for each symbol, navigate down the tree
        for symbol in context:
            #the child to navigate's index is based on the symbol
            idx = int(symbol)

            #go down to the child
            node = node.children[idx]

        return node

    def update_nodes(self, context, symbol, amount):
        #find the leaf context node
        node = self.get_context_node(context)

        #start updating at that node, returning the pw of the root
        return node.update(symbol,amount)
    
    def __repr__(self):
        return self.root.__repr__()

class Node():
    def __init__(self, context= None, len_counts: int = None, num_children: int = None, alphabet_size: int = None, parent: object = None, isY: bool = None):
        #initialize the node's context
        self.context = context or []

        #initialize the Y counts and total counts
        self.counts = np.zeros(len_counts)
        self.total_counts = 0

        #initialize the Y alphabet size
        self.alphabet_size = alphabet_size

        #initialize kts and ws at 0 due to log probs
        self.kts = 0
        self.ws = 0

        #kt and ws distributions
        self.kts_pred = np.zeros(len_counts)

        #initialize children and parent
        self.children = [None] * num_children
        self.parent = parent

        #initialize isY
        self.isY = isY
        
    def __repr__(self, level=0):
        #represent the tree with indents
        indent = "  " * level
        rep = f"{indent}-[{self.context}]- Counts: {self.counts} - kts {self.kts_pred} - ws {self.ws}\n"
        for child in self.children:
            if child is not None and child.total_counts > 0:
                rep += child.__repr__(level + 1)
        return rep
    
    def update(self, symbol, amount):
        #get the index of the symbol
        symbol_idx = int(symbol)

        #update kts
        self.kts = self.log_kt_probability(self.counts)

        #increment counts
        self.counts[symbol_idx] += amount
        self.total_counts += amount

        #update ws
        #if it is a leaf node it is equal to kts
        if all(child is None for child in self.children):
            self.ws = self.kts
        else:
        #if it is an internal node use half kts and half the sum of the children's ws in log space
            k = self.kts
            c = sum(child.ws for child in self.children if child.total_counts > 0) 
            max_val = max(k, c)
            self.ws = max_val + np.log2(1 + 2**(-abs(k-c))) - 1
        
        #update kts 
        self.kts_pred = [(self.counts[i] + 0.5) / (self.total_counts + self.alphabet_size/2) for i in range(len(self.counts))]

        #update the parent unless root where we return the ws
        if self.parent is not None:
            return self.parent.update(symbol, amount)
        else:
            return self.ws
    
    def log_kt_probability(self, counts):
        """
        counts: np.array of symbol counts, shape (alphabet_size,)
        returns: log2(KT probability)
        """
        total = np.sum(counts)
        k = len(counts)

        log_prob = gammaln(k / 2.0)
        log_prob -= gammaln(total + k / 2.0)
        log_prob += np.sum(gammaln(counts + 0.5))
        log_prob -= k * gammaln(0.5)

        return log_prob / np.log(2)