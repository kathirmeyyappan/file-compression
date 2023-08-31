from collections import defaultdict
from heapq import heapify, heappush, heappop
from typing import Dict, List, DefaultDict, Union


class HuffmanNode:
    
    def __init__(self, left: Union['HuffmanNode', str], 
                 right: Union['HuffmanNode', str]):
        self.left = left
        self.right = right
        

def make_huffman_tree(char_counts: DefaultDict) -> HuffmanNode:
    """
    Creates Huffman tree using a queue of characters and our HuffmanNode class.

    Args:
        freqs (DefaultDict): map of all unique characters in the file and their
        respective frequencies

    Returns:
        HuffmanNode: Huffman Tree which can encode and decode file
    """
    # q is the character/node queue which contains elements of the form
    #   ( freq, [char or HuffmanNode] ). We construct it here. Note that i 
    #   is just a simple counter to break arbitrary tuple ties when pushing 
    #   a new item to the queue.
    q = []
    i = 0
    for char, freq in char_counts.items():
        heappush(q, (freq, i, char))
        i += 1

    # keep combining characters and Huffman nodes of the least freqency until
    #   we have one HuffmanNode object which is our tree.
    while len(q) > 1:
        f1, _, c1 = heappop(q)
        f2, _, c2 = heappop(q)
        heappush(q, (f1 + f2, i, HuffmanNode(c1, c2)))
        i += 1
    
    # return final Huffman Node which is out tree for encoding
    return q[0][2]


def make_encoding_dict(tree: HuffmanNode) -> Dict[str, bytes]:
    """
    Creates dictionary which maps characters to binary for file encoding.

    Args:
        huffman_tree (HuffmanNode): Huffman tree for specified file.

    Returns:
        Dict[str, bytes]: resulting mapping of characters to binary codes.
    """
    encoding_dict = {}
    
    def dfs(node: HuffmanNode, cur_bin: str) -> None:
        """
        Recursively constructs binary codes for characters based on 
        Huffman Tree.

        Args:
            node (HuffmanNode): current node
            cur_bin (bytes): binary code for position in tree
        """
        if type(node.left) is str:
            encoding_dict[node.left] = cur_bin + '0'
        else:
            dfs(node.left, cur_bin + '0')
        
        if type(node.right) is str:
            encoding_dict[node.right] = cur_bin + '1'
        else:
            dfs(node.right, cur_bin + '1')
    
    dfs(tree, '')
    return encoding_dict