# Will save file info of an npz to xlsx
import os
import time
import zlib
import zipfile
import argparse
import numpy as np
from openpyxl import Workbook
from easyvolcap.utils.console_utils import *


def get_npz_info(npz_file):
    """Extract various information from a .npz file."""

    file_info = []

    with zipfile.ZipFile(npz_file, 'r') as zf:
        for key in zf.namelist():
            # Reading array data using numpy
            with zf.open(key) as array_file:
                array = np.load(array_file, allow_pickle=True)

                # Extracting array details
                size = array.nbytes  # Size of the array in bytes
                compressed_size = zf.getinfo(key).compress_size  # Size after compression
                dtype = str(array.dtype)  # Data type of the array, converted to string

                # Modification time
                mtime = time.ctime(os.path.getmtime(npz_file))
                # Compute CRC32
                crc32 = zf.getinfo(key).CRC

                file_info.append([key, size, compressed_size, dtype, mtime, crc32])

    return file_info


def write_to_excel(data, output_file):
    """Write data to an Excel file."""

    wb = Workbook()
    ws = wb.active
    headers = ['Filename', 'Size (Bytes)', 'Compressed Size (Bytes)', 'Type', 'Modification Time', 'CRC32']
    ws.append(headers)

    for row in data:
        ws.append(row)

    wb.save(output_file)


@catch_throw
def main():
    parser = argparse.ArgumentParser(description='Extract information from a .npz file and save it to an Excel file.')
    parser.add_argument('-i', '--input', default='data/trained_model/scr4dv_0013_01_r4/1599.npz', help='Path to the .npz file')
    parser.add_argument('-o', '--output', default='data/trained_model/scr4dv_0013_01_r4/1599.npz.info.xlsx', help='Output Excel file name')
    args = parser.parse_args()

    print(f"Extracting information from: {blue(args.input)}")
    info = get_npz_info(args.input)

    print(f"Writing data to: {blue(args.output)}")
    write_to_excel(info, args.output)
    print(f"Data successfully saved to {blue(args.output)}")


if __name__ == '__main__':
    main()
