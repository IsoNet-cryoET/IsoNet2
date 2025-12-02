refine

If we have evn/odd tomogram but we want to use IsoNet2 refine without noise2noise integrated training.
isonet.py prepare_star -e EVN -o ODD --create_average True
Then you will find the star file with column rlnTomoName #2

Then run isonet.py with --method isonet2 specified such as
isonet.py refine tomograms.star --method isonet2

if we do not have evn/odd toomograms, when we prepare star, we use command such as
isonet.py prepare_star tomograms
or
isonet.py prepare_star -f tomograms

for predict refine make_mask and deconv, all of them have parameter called --input_column, if you want to specify what is the input for the command. For example,
isonet.py make_mask tomograms.star --input_column rlnDeconvTomoName
meaning generating mask using tomograms in the column of rlnDeconvTomoName

or
isonet.py refine tomograms.star --input_column rlnDeconvTomoName --method isonet2 --noise_level 0.2

or
isonet.py predict tomograms.star isonet_maps/network_isonet2_unet-medium_96_full.pt --tomo_idx 1 --input_column rlnDeconvTomoName
