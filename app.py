import tkinter as tk
from tkinter import filedialog
import sys
import os
import shutil

# Determine the path of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the 'src' directory to sys.path
src_dir = os.path.join(script_dir, 'src')
sys.path.append(src_dir)

from huffman_encode import compress_file
from huffman_decode import decompress_file


### HELPER FUNCTIONS ###            

def get_compressed_file() -> None:
    
    # ask user to select file
    filepath = filedialog.askopenfilename(title="Choose a (text-based) "
                                          "file to be compressed")
    if filepath:
        # loading status
        compression_err_label.pack_forget()
        decompression_err_label.pack_forget()
        intro_label.pack_forget()
        processing_label.pack()
        complete_label.pack_forget()
        
        try:
            # compress file into .zip and save in directory
            filename = compress_file(filepath)
            
        except Exception as e:
            # bad filetype
            compression_err_label.pack()
            decompression_err_label.pack_forget()
            intro_label.pack_forget()
            processing_label.pack_forget()
            complete_label.pack_forget()
            return
        
        # get user input for location to save file and move it
        save_path = filedialog.asksaveasfilename(title="Choose save location",
                                                 initialfile=filename)
        
        if save_path:
            shutil.move(filename, save_path)
        
            # say complete
            compression_err_label.pack_forget()
            decompression_err_label.pack_forget()
            intro_label.pack_forget()
            processing_label.pack_forget()
            complete_label.pack()
        
        else:
            # say complete
            compression_err_label.pack_forget()
            decompression_err_label.pack_forget()
            intro_label.pack()
            processing_label.pack_forget()
            complete_label.pack_forget()


def get_decompressed_file() -> None:
    
    # ask user to select file
    filepath = filedialog.askopenfilename(title="Please choose huffman-compressed"
                                          " file to be decompressed")
    if filepath:
        # loading status
        compression_err_label.pack_forget()
        decompression_err_label.pack_forget()
        intro_label.pack_forget()
        processing_label.pack()
        complete_label.pack_forget()
        
        try:
            # compress file into .zip and save in directory
            filename = decompress_file(filepath)
            
        except Exception as e:
            # bad filetype
            compression_err_label.pack_forget()
            decompression_err_label.pack()
            intro_label.pack_forget()
            processing_label.pack_forget()
            complete_label.pack_forget()
            return
        
        # get user input for location to save file and move it
        save_path = filedialog.asksaveasfilename(title="Choose save location",
                                                 initialfile=filename)
        
        if save_path:
            shutil.move(filename, save_path)
        
            # say complete
            compression_err_label.pack_forget()
            decompression_err_label.pack_forget()
            intro_label.pack_forget()
            processing_label.pack_forget()
            complete_label.pack()
        
        else:
            # say complete
            compression_err_label.pack_forget()
            decompression_err_label.pack_forget()
            intro_label.pack()
            processing_label.pack_forget()
            complete_label.pack_forget()
    

### GUI APPEARANCE AND RUNNING ###

# initialize window
window = tk.Tk(className="Huffman File Compressor")
window.title("Huffman File Compressor")
window.geometry("400x300")

# Label to display processing status
intro_label = tk.Label(window, text="This app uses Huffman coding "
                        "to decrease file size and interpret text files as "
                        "binary. \n\nChoose file to compress/decompress!",
                        font=("Helvetica", 16), wraplength=380)
processing_label = tk.Label(window, text="processing...",
                        font=("Helvetica", 16), wraplength=380, foreground="blue")
complete_label = tk.Label(window, text="complete!",
                        font=("Helvetica", 16), wraplength=380, foreground="green")
compression_err_label = tk.Label(window, text="Please choose a text-based file "
                                 "for compression. This means NO .pdf, .jpg, "
                                 ".png, .bin, .zip, etc.", font=("Helvetica", 16), 
                                 wraplength=380, foreground="red")
decompression_err_label = tk.Label(window, text="Please choose a file that is "
                                   "the result of Huffman-compression from "
                                   "this application!", font=("Helvetica", 16), 
                                    wraplength=380, foreground="red")

# Create a frame for buttons
button_frame = tk.Frame(window)
button_frame.pack(pady=20)

# Create "Compress" button
compress_button = tk.Button(button_frame, text="Compress File", 
                            command=get_compressed_file)
compress_button.pack(side="left", padx=10)

# Create "Decompress" button
decompress_button = tk.Button(button_frame, text="Decompress File", 
                              command=get_decompressed_file)
decompress_button.pack(side="left", padx=10)


compress_button.pack()
decompress_button.pack()
intro_label.pack()

window.mainloop()