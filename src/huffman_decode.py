import argparse
import zipfile
import os
from collections import defaultdict
from typing import Dict, List, DefaultDict, Union
import math

from huffman_tree import HuffmanNode, make_huffman_tree, make_encoding_dict


def decompress_file(zipfilepath: str) -> str:
    """
    Decompresses zipfile by recreating the original file using info in the .bin
    file within the .zip.

    Args:
        zipfilepath (str): path to the zipfile which is to be decompressed
        
    Returns:
        str: saved filepath
    """
    
    ### READING FILE AND RECONSTRUCTING HUFFMAN CODING INFO ###
    
    # read zipfile to get raw compressed data
    with zipfile.ZipFile(zipfilepath, 'r') as zip_file:
        zipfilename = os.path.basename(zipfilepath)
        og_filename = zipfilename[11:-4]
        bin_filename = f'{zipfilename[:-4]}.bin'
        
        with zip_file.open(bin_filename, 'r') as binary_file:
            compressed_data = binary_file.read()
    
    # separator signatures from encoding
    section_separator = 'SEPARATOR'.encode('utf-8')
    decoding_dict_separator = b'\xaa\xbb\xcc'
    
    # recreating character frequency dictionary
    freqs = defaultdict(int)
    dict_end = compressed_data.index(section_separator)    
    decoding_lst = compressed_data[:dict_end].split(decoding_dict_separator)
    for i in range(0, len(decoding_lst) - 1, 2):
        char = decoding_lst[i].decode('utf-8')
        freq = decoding_lst[i + 1].decode('utf-8')
        freqs[char] = int(freq)
    
    # recreate Huffman tree from freqs
    huffman_tree = make_huffman_tree(freqs)
    
    
    ### RECREATING ORIGINAL FILE USING HUFFMAN TREE AND BIT DATA ###
    
    # retrieve number of dummy bits at ext of text data
    dummy_bit_count = int(chr(compressed_data[-1]))
    
    # turning text data (bytes) into pure binary to get back old text
    text_bytes_data = compressed_data[dict_end + 9:-10]
    text_bin_data = ''
    for b in text_bytes_data:
        text_bin_data += format(b, '08b')
    
    # write interpretted characters from binary to file
    with open(f'decompressed_{og_filename}', 'w') as decompressed_file:
        
        # iterate through binary and traverse tree, resetting the value of 
        #   cur every time a character is reached
        cur = huffman_tree
        txt = ''
        for i in range(len(text_bin_data) - dummy_bit_count):
            bit = text_bin_data[i]
            
            # update cur
            if bit == '0':
                cur = cur.left
            elif bit == '1':
                cur = cur.right
            
            # add to file write string if character is reached in tree
            if type(cur) is str:
                txt += cur
                cur = huffman_tree
            
            # write to file if string gets big and reset
            if len(txt) > 10000:
                decompressed_file.write(txt)
                txt = ''
        # final write to file
        decompressed_file.write(txt)
    
    return f"decompressed_{og_filename}"    


if __name__ == "__main__":
    
    # demand filepath for encoding
    parser = argparse.ArgumentParser(description='Returns huffman encoded \
        version of inputted file (in binary form).')
    parser.add_argument('src_file', type=argparse.FileType('r'))

    # get filepath for use
    args = parser.parse_args()
    filepath = args.src_file.name

    # get compressed binary file
    decompress_file(filepath)