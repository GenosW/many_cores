#!fish

wget -P data/buffer -i urls.txt
mv data/buffer/* data/.

# rename some files because I forgot to set them correctly
mv data/ph_data_vexcl_ocl2.csv data/ph_data_vexcl_boost.csv
mv data/ph_data_vexcl_ocl3.csv data/ph_data_vexcl_cuda.csv