# huffman-file-compression
Desktop application that employs lossless compression methods (Huffman coding!) to zip and unzip files with 100% accuracy.

## Demonstration
Below is an example use of the tool. We compress [FILENAME] into [COMPRESSED_FILENAME], which is about [RATIO] the size ([COMPARE]). Then, we unzip [COMPRESSED_FILENAME] into [DECOMPRESSED_FILENAME], which is perfectly identical in size and contents to the original file.

## Methodology
The idea behind lossless compression is that we can re-express data with less information without actually omitting any of it. Typically, in text-based files, one character (as long as it is a "common" enough character) is represented by 1 byte. Using Huffman coding, we can actually encode our data such that one character (on average) is less than a byte!

First, we record the frequency of each character, and contruct a binary tree such that traversals (where right and left correpond to 1 and 0) to leaves each point to a character. This is our key for encoding the data. We write the encoded text data to a binary file along with with binary-ified version of the Huffman tree (for deocding purposes) and zip it up, giving us our compressed file! I also wrote a decoding algorithm to parse the file into the Huffman tree I embedded in the file as well as raw text data.

## Limitations
For very small files, running the Huffman encoding algorithm actually increased size. This is because storing the decoding info (Huffman tree data) within the binary file takes some space, often more than what a very small file may have contained in the first place (a couple hundred bytes).

For files with a large variety of characters (> 2^8), each encoded character will take up more than a byte on average based on our Huffman tree. This may not be result in a net increase in file-size per say (because many types of characters would take up 2 or 3 bytes of there were that many in the first place), but it has diminishing returns.
