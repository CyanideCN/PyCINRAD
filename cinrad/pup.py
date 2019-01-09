import numpy as np
import datetime

test_file = r'C:\Users\27455\source\repos\ReadPUPData19\ReadPUPData19\20060803.000324.01.19.753'

message_header = [("product_code", ">i2"),
                  ("date", ">i2"),
                  ("time", ">i4"),
                  ("file_length", ">i4"),
                  ("stid", ">i2"),
                  ("dst_id", ">i2"),
                  ("block_num", ">i2")]

message_header_dtype = np.dtype(message_header)

product_dsc = [("divider", ">i2"),
               ("Latitude", ">i4"),
               ("Longitude", ">i4"),
               ("height", ">i2"),
               ("product_code_", ">i2"),
               ("op_mode", ">i2"),
               ("vcp", ">i2"),
               ("seq_num", ">i2"),
               ("vol_scan_num", ">i2"),
               ("vol_start_date", ">i2"),
               ("vol_start_time", ">i4"),
               ("prod_gen_date", ">i2"),
               ("prod_gen_time", ">i4"),
               ("dependence_1", "2>i2"),
               ("el_num", ">i2"),
               ("dependence_2", ">i2"),
               ("threshold", "16>i2"),
               ("dependence_3", "6>i2"),
               ("ver", ">i2"),
               ("gap_sym", ">i4"),
               ("gap_graphic", ">i4"),
               ("gap_alpha", ">i4")]

product_dsc_dtype = np.dtype(product_dsc)

prod_symbology = [("divider_2", ">i2"),
                  ("block_id", ">i2"),
                  ("block_length", ">i4"),
                  ("layer_num", ">i2")]

prod_symbology_dtype = np.dtype(prod_symbology)

f = open(test_file, 'rb')
data = np.frombuffer(f.read(message_header_dtype.itemsize), message_header_dtype)
for idx in data.dtype.fields.keys():
    print(idx, data[idx])
data = np.frombuffer(f.read(product_dsc_dtype.itemsize), product_dsc_dtype)
for idx in data.dtype.fields.keys():
    print(idx, data[idx])
data = np.frombuffer(f.read(prod_symbology_dtype.itemsize), prod_symbology_dtype)
for idx in data.dtype.fields.keys():
    print(idx, data[idx])


prod_code = np.frombuffer(f.read(2), '>i2')
julian_date = np.frombuffer(f.read(2), '>i2')
time_offset = np.frombuffer(f.read(4), '>i4')
data_time = datetime.datetime(1970, 1, 1) + datetime.timedelta(days=int(julian_date[0]), milliseconds=int(time_offset[0]))
header_length = np.frombuffer(f.read(4), '>i4')
source_id = np.frombuffer(f.read(2), '>i2')
dest_id = np.frombuffer(f.read(2), '>i2')
block_num = np.frombuffer(f.read(2), '>i2')
block_divider = np.frombuffer(f.read(2), '>i2')
latitude = np.frombuffer(f.read(4), '>i4') / 1000
longitude = np.frombuffer(f.read(4), '>i4') / 1000
height = np.frombuffer(f.read(2), '>i2')