import argparse
import zipfile
import os
from collections import defaultdict
from typing import Dict, List, DefaultDict, Union
import math

from huffman_tree import HuffmanNode, make_huffman_tree, make_encoding_dict


def compress_file(srcfilepath: str) -> str:
    """
    Creates compressed binary version of file and stores in 
    compressed_files folder.

    Args:
        srcfilepath (str): path to the file which is to be compressed
    
    Returns:
        str: saved filepath
    """
    
    ### READING FILE AND CREATING HUFFMAN ENCODING INFO ###
    
    # read source file
    with open(srcfilepath, 'r') as file:
        file_text = file.read()
    
    # get character frequencies in file
    freqs = defaultdict(int)
    for char in file_text:
        freqs[char] += 1
    
    # create Huffman tree    
    huffman_tree = make_huffman_tree(freqs)
    
    # get encoding dictionary from tree
    encoding_dict = make_encoding_dict(huffman_tree)
    
    
    ### WRITING TO BINARY FILE ###
    
    # separator signature for different parts of the binary file
    section_separator = 'SEPARATOR'.encode('utf-8')
    encoding_dict_separator = b'\xaa\xbb\xcc'
    
    # write to compressed file
    srcfilename = os.path.basename(srcfilepath)
    
    zip_filename = f'compressed_{srcfilename}.zip'
    with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        
        bin_filename = f'compressed_{srcfilename}.bin'
        with zip_file.open(bin_filename, 'w') as binary_file:
            
            # write information for decoding (character frequency dict) here
            for char, freq in freqs.items():
                decoding_char = char.encode('utf-8') + encoding_dict_separator
                decoding_freq = str(freq).encode('utf-8') + encoding_dict_separator
                binary_file.write(decoding_char + decoding_freq)
            
            # separating sections with signature
            binary_file.write(section_separator)
            
            # add text data bits according to encoding dictionary
            text_data = ''
            for char in file_text:
                text_data += encoding_dict[char]
                if len(text_data) % 100000 == 0:
                    # convert string bits to bytes and write to file
                    binary_data_bytes = bytes(int(text_data[i:i+8], 2) \
                        for i in range(0, len(text_data), 8))
                    binary_file.write(binary_data_bytes)
                    text_data = ''
                    
            # Pad '0' bits to make text_data a multiple of 8
            dummy_bit_count = 8 - (len(text_data) % 8)
            text_data += '0' * dummy_bit_count
            
            # convert string bits to bytes and write to file
            binary_data_bytes = bytes(int(text_data[i:i+8], 2) \
                for i in range(0, len(text_data), 8))
            binary_file.write(binary_data_bytes)
            
            # separating sections with signature
            binary_file.write(section_separator)
            
            #adding section to denote number of dummy bits at end of text_data
            binary_file.write(str(dummy_bit_count).encode('utf-8'))
    
    return zip_filename
            
        
if __name__ == "__main__":
    
    # demand filepath for encoding
    parser = argparse.ArgumentParser(description='Returns huffman encoded \
        version of inputted file (in binary form).')
    parser.add_argument('src_file', type=argparse.FileType('r'))

    # get filepath for use
    args = parser.parse_args()
    filepath = args.src_file.name

    # get compressed binary file
    compress_file(filepath)
