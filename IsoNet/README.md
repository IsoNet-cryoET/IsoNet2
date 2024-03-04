
# IsoNet version 1.0

Update on Feb 7 2024

Single Particle IsoNet (IsoNet) is designed to correct for the preferred orientation problem in cryoEM by self-supervised deep learning, by recovering missing information from well-sampled orientations in Fourier space. 

Unlike conventional supervised deep learning methods that need explicit input-output pairs for training, IsoNet autonomously extracts supervisory signals from the original data, ensuring the reliability of the information used for training.

IsoNet is designed for single particle analysis and subtomogram averaging. For the correcting missing wedge in cryoET, please refer to IsoNet.

Please find tutorial/IsoNet_v1.0_Tutorial.md for detailed document.

## Google group
We maintain an IsoNet Google group for discussions or news.

To subscribe or visit the group via the web interface please visit https://groups.google.com/u/1/g/isonet. 

If you do not have and are not willing to create a Google login, you can also request membership by sending an email to yuntao@g.ucla.edu

To post to the forum you can either use the web interface or email to isonet@googlegroups.com

# 1. Installation

We suggest using anaconda environment to manage the IsoNet package.

1. Install cudatoolkit and cudnn on your computer.
2. Install pytorch from https://pytorch.org/ 
3. Create an conda virtual environment and install dependencies by running install.sh in the IsoNet folder or by pip install
   The dependencies include tqdm, matplotlib, scipy, numpy, scikit-image, mrcfile, fire
4. For example, add the following lines in your ~/.bashrc

    export PATH=PATH_TO_ISONET_FOLDER/bin:$PATH 

    export PYTHONPATH=PATH_TO_PARENT_FOLDER_OF_ISONET_FOLDER:$PYTHONPATH 
    or you can run source source-env.sh in your terminal, which will export required variables into your environment.
5. Now IsoNet is available to use.

The environment we verified are:
1. cuda11.8 cudnn8.5 pytorch2.0.1, pytorch installed with pip.
2. cuda11.3 cudnn8.2 pytorch1.13.1, pytorch installed with conda.