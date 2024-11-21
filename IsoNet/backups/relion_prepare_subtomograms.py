#!/usr/env/bin/python
print ':: RELION sub-tomogram averaging ::' 
print 'This python script was written by Tanmay Bharat to support sub-tomogram averaging in RELION.' 
print 'Please ensure that you have IMOD and RELION commands in your path and that you have CTFFIND installed.' 
print 'Please report bugs and comments to tbharat@mrc-lmb.cam.ac.uk or scheres@mrc-lmb.cam.ac.uk'
print 'Please read the documentation on the RELION wiki, several questions are answered there.'  
print 'This version can set defocus values above a certain tilt to the defocus value of the zero degree tilt'
print 'This version will write out all the CTF reconstruction commands in the master file' 
print 'Please make sure to indicate whether you will use RELION 2.0 or not'
#Mon May 23 18:06:33 BST 2016

import os, sys, commands, math, time, stat, glob, shutil

######### INPUT #########################################

## Input STAR file with all tomograms
TomogramStarFileName = 'all_tomograms.star'

## suffix for subtomograms. For RELION 2.0 this only denotes the .star file name. 
RootName = 'subtomo'

## Skip CTF correction. The 3D CTF model will have no CTF modulations, but will still use the Tilt and Bfactor weighting.
SkipCTFCorrection = False

## Is RELION 2.0 going to be used (True or False)?
Relion2 = True

## Extract job alias. Alias of the job given during particle extraction. Only for RELION 2.0
ExtractJobAlias = 'extract_tomo'

## CTFFIND CTF estimation input
#################################
# Microscope voltage in kV
Voltage = 300              
# Spherical aberration coefficient in mm
Cs = 2.7
# Magnification of the image
Magnification = 53000       
# Pixel size of the detector (in micron)
DPixSize = 11.57  
# Path to CTFFIND (version 3 or 4)
PathToCtffind = '/public/EM/ctffind/ctffind.exe'     
# If CTFFIND crashed in the middle, you can turn this to True to resume CTF estimations only for unfinished images
OnlyDoUnfinishedCTFs = False                            
# Boxsize for CTFFIND
BoxSize = 256
# Lower resolution limit for CTFFIND
LowResLimit = 50
# Higher resolution limit for CTFFIND fitting
HighResLimit = 8
# Lowest nominal defocus tilt series in the data set
LowDefocusLimit = 20000
# Highest nominal defocus tilt series in the data set
HighDefocusLimit = 50000
# Step search for searching defocus values
DefocusStep = 1000
# Amplitude contrast in images, normally you do not need to change this
AmpContrast = 0.07
# Expected astigmatism in the images, should be higher than for SPA due to tilting
Astigmatism = 2000
# Only run CTFFIND for unfinished images?
OnlyDoUnfinishedCTFs = False
# Skip running CTFFIND but re-run the rest of the setup script? True will skip running CTFFIND because it was run previously.
ReRunCtffindSkip = False
#################################

## Other options to improve CTF accuracy
#################################
# Use trials for CTFFIND. Please keep a tomogram.trial stack in the Tomograms directory containing two trials from either side of the record region. Please note that the tilt order of the files should be same as the aligned stack. 
UseTrialsForCtffind = False
# If you don't have extra trials, then maybe you can set an upper limit of abs(tilt), over which the average defocus value from lower tilts is used.
UseOnlyLowerTiltDefoci = True
UseOnlyLowerTiltDefociLimit = 30.0
## 3D CTF model weighting B-factor per e-/A2
Bfactor = 4.0
#################################

###########################################################



######## FUNCTIONS ########################################
#print 'functions'

#
def ensure_dir(f):
  d = os.path.dirname(f)
  if not os.path.exists(d):
    #print 'Making directory'
    os.makedirs(d)
#

# To read the STAR files. Please note that this STAR file parser is only meant for setting up the sub-tomogram averaging scripts.
# RELION has a more comprehensive parser in the main code.
def read_relion_star(filename):
  starfile=open(filename, 'r')
  j=-1

  micnames=[]
  defociu=[]
  defociv=[]
  for line in starfile:

    #print line

    emptycheck = line.isspace()
    if(emptycheck):
      #print 'empty line found'
      continue
    
    fields = line.split()
    firstfield = fields[0]
    if firstfield[0] == 'd':
      #print 'data_ line found'
      continue
    if firstfield[0] == 'l':
      #print 'loop_ line found'
      continue
    j=j+1 

    if firstfield == '_rlnMicrographName':
      imgnamecolumn = j
      continue
    if firstfield == '_rlnDefocusU':
      defocusucolumn = j
      continue
    if firstfield == '_rlnDefocusV':
      defocusvcolumn = j
      continue
    #if firstfield == '_rlnCtfFigureOfMerit':
    #  ctffigureofmeritcolumn = j
    #  continue
    if firstfield[0] == '_':
      continue
    

    micnames.append(fields[imgnamecolumn])
    if 'defocusucolumn' in locals():
      defociu.append(fields[defocusucolumn])
      defociv.append(fields[defocusvcolumn])

  starfile.close()
  if len(defociu) > 0:
    return micnames,defociu,defociv
  if len(defociu) == 0:
    return micnames
#

#########################################################




######## RUNNING THE SCRIPT #################


#################################
print 'Running the script'

# This is to ensure that each entered variable has the correct form
Voltage = float(Voltage)              
Cs = float(Cs)                   
Magnification = float(Magnification)     
DPixSize = float(DPixSize)
PathToCtffind = str(PathToCtffind)     
BoxSize = float(BoxSize)
LowResLimit = float(LowResLimit)
HighResLimit = float(HighResLimit)
LowDefocusLimit = float(LowDefocusLimit)
HighDefocusLimit = float(HighDefocusLimit)
DefocusStep = float(DefocusStep)
AmpContrast = float(AmpContrast)
Astigmatism = float(Astigmatism)
UseOnlyLowerTiltDefociLimit = float(UseOnlyLowerTiltDefociLimit)
Bfactor = float(Bfactor)

# If you do not want CTF correction, and only want to have a weighted missing wedge
if SkipCTFCorrection == True:
  Cs = 0.0
  AmpContrast = 1.0
  UseTrialsForCtffind = False
  UseOnlyLowerTiltDefoci = False

#sys.exit()
# Text file containing all RELION commands
reliontextfile = open('relion_subtomo_commands.txt', 'w')

## Looping through the micrographs
ScriptDir = os.getcwd() + '/'
print ScriptDir

micnames = read_relion_star(TomogramStarFileName)
print micnames

# Shell script to do 3D CTF model reconstruction
ctfreconstmastername = ScriptDir + 'do_all_reconstruct_ctfs.sh'
ctfreconstmasterfile = open(ctfreconstmastername, 'w')
os.chmod(ctfreconstmastername, stat.S_IRWXU)

#
# This is the master STAR file for refinement later on
subtomostarname = ScriptDir + 'particles_' + RootName + '.star'
subtomostarfile = open(subtomostarname, 'w')
# writing out the header of the list star file
subtomostarfile.write('data_' + '\n' + '\n')
subtomostarfile.write('loop_' + '\n')
subtomostarfile.write('_rlnMicrographName #1' + '\n')
subtomostarfile.write('_rlnCoordinateX #2' + '\n')
subtomostarfile.write('_rlnCoordinateY #3'+ '\n')
subtomostarfile.write('_rlnCoordinateZ #4' + '\n')
subtomostarfile.write('_rlnImageName #5' + '\n')
subtomostarfile.write('_rlnCtfImage #6' +'\n')
# RELION 2.0
if Relion2 == True:
  subtomostarfile.write('_rlnMagnification #7' + '\n')
  subtomostarfile.write('_rlnDetectorPixelSize #8' + '\n')
#

for mic in micnames:

  #
  # Parsing the micrograph names 
  #micsplit = mic.split('.')
  micsplit = os.path.splitext(mic)
  microot = micsplit[0]
  dirsplit = microot.split('/')
  MicDirName = ""
  for dircount in range(0,(len(dirsplit)-1)):
    MicDirName = MicDirName + dirsplit[dircount]
    MicDirName = MicDirName + '/'
  MicRootName = dirsplit[len(dirsplit)-1]
  
  #print MicDirName
  #print MicRootName
  
  micname = MicDirName + MicRootName + '.mrc'
  stackname = MicDirName + MicRootName + '.mrcs'
  ordername = MicDirName + MicRootName + '.order'
  coordsname = MicDirName + MicRootName + '.coords'
  trialname = MicDirName + MicRootName + '.trial'
  alitiltname = MicDirName + MicRootName + '.tlt'
  #print micname, stackname, ordername, coordsname, trialname, alitiltname
  # Parsing the micrograph names 
  #

  #sys.exit()

  ##### Running CTFFIND on all images of the tilt series  ##########

  CtffindDirName = 'ctffind/'
  OutputDir = ScriptDir + MicDirName + CtffindDirName
  newstackroot = MicDirName + CtffindDirName + MicRootName +  '_image'
  #print OutputDir

  ## Making a new directory to output the results of CTFFIND
  ensure_dir(OutputDir)

  ## Extracting the tilt information with the IMOD command extracttilts
  if not os.path.isfile(alitiltname):
    extracttile_scratchname = OutputDir + 'extracttilt_output.txt'
    print ':: RELION sub-tomogram averaging :: ' + '\n' + 'Using IMOD extracttilts to get tilt angles' + '\n' 
    exttltline = 'extracttilts -InputFile ' + stackname + ' -tilts -OutputFile ' + OutputDir +  'tiltangles.txt > ' + extracttile_scratchname +  '\n'
    print(exttltline)
    os.system(exttltline)
    os.remove(extracttile_scratchname)
  if os.path.exists(alitiltname):
    outtiltnametemp = OutputDir + 'tiltangles.txt'
    shutil.copyfile(alitiltname,outtiltnametemp)
  
  #sys.exit()

  ## If trials are being used for CTFFIND
  if UseTrialsForCtffind == True:
    extracttrial_scratchname = OutputDir + 'extracttilt_output_trial.txt'
    print ':: RELION sub-tomogram averaging :: ' + '\n' + 'Using IMOD extracttilts to get tilt angles for trial images' + '\n' 
    exttrialline = 'extracttilts -InputFile ' + trialname + ' -tilts -OutputFile ' + OutputDir +  'trial_tiltangles.txt > ' + extracttrial_scratchname +  '\n'
    print(exttrialline)
    os.system(exttrialline)
    os.remove(extracttrial_scratchname)
  ##
  ##
  print ':: RELION sub-tomogram averaging :: ' + '\n' + 'Tilt values extracted ' + '\n'

  ##
  tiltanglesfilename = OutputDir + 'tiltangles.txt'
  tiltfile = open(tiltanglesfilename, 'r')
  
  trial_tiltanglesname = OutputDir + 'trial_tiltangles.txt'

  ctffindstarname = OutputDir + MicRootName + '_images.star'
  ctffindstarfile = open(ctffindstarname, 'w')
  ctffindstarfile.write('data_' + '\n' + '\n')
  ctffindstarfile.write('loop_' + '\n')
  ctffindstarfile.write('_rlnMicrographName #1' + '\n')
  
  if UseTrialsForCtffind == True:
    with open(trial_tiltanglesname) as f:
      lines = f.readlines()
  #print lines
  
  trial_exttilts=[]
  exttilts=[]

  ctffindstarname = OutputDir + MicRootName + '_images.star'
  ctffindstarfile = open(ctffindstarname, 'w')
  ctffindstarfile.write('data_' + '\n' + '\n')
  ctffindstarfile.write('loop_' + '\n')
  ctffindstarfile.write('_rlnMicrographName #1' + '\n')
  
  i=-1
  for line in tiltfile:
    #print 'hello'
    
    pair = line.split()
    #print pair
    i=i+1
    
    # Tilt of the stage for the current image
    tilt = float(pair[0])
    #roundtilt = round(tilt)
    exttilts.append(tilt)
    if UseTrialsForCtffind==True:
      trial_tilt1 = lines[2*i]
      trial_tilt2 = lines[(2*i)+1]
      #print float(trial_tilt1)
      trial_exttilts.append(float(trial_tilt1))
      trial_exttilts.append(float(trial_tilt2))
    #print(str(int(roundtilt)))
    
    # extracting each image using the IMOD command newstack
    print ':: RELION sub-tomogram averaging :: ' + '\n' + 'Extracting tilt series image ' + '\n' 
    newstack_scratchname = OutputDir + 'temp_newstack_out.txt'
    if UseTrialsForCtffind == False:
      extracted_image_name = newstackroot + str(tilt) + '_' + str(i) + '.mrc'
      newstackline = 'newstack -secs ' + str(i) + ' ' + stackname + ' ' +  extracted_image_name + ' > ' + newstack_scratchname +'\n'
      print(newstackline)
      ctffindstarfile.write(extracted_image_name + '\n')
      os.system(newstackline)
      os.remove(newstack_scratchname)
    
    if UseTrialsForCtffind == True:
      extracted_image_name1 = newstackroot + str(tilt) + '_' + str(2*i) + '.mrc'
      extracted_image_name2 = newstackroot + str(tilt) + '_' + str((2*i)+1) + '.mrc'
      newstackline1 = 'newstack -secs ' + str(2*i) + ' ' + trialname + ' ' +  extracted_image_name1 + ' > ' + newstack_scratchname +'\n'
      newstackline2 = 'newstack -secs ' + str((2*i) + 1) + ' ' + trialname + ' ' +  extracted_image_name2 + ' > ' + newstack_scratchname +'\n'
      print(newstackline1)
      print(newstackline2)
      ctffindstarfile.write(extracted_image_name1 + '\n')
      ctffindstarfile.write(extracted_image_name2 + '\n')
      os.system(newstackline1)
      os.system(newstackline2)
      os.remove(newstack_scratchname)
  
  ctffindstarfile.close()
  #print trial_exttilts
  
  # running CTFFIND using the RELION command relion_run_ctffind
  # RELION 1.4
  if SkipCTFCorrection == False:
    if Relion2 == False:
      outputstarname =  OutputDir + MicRootName +  '_ctffind.star'
      outputstarname_read = outputstarname
      print ':: RELION sub-tomogram averaging :: ' + '\n' + 'Running relion_run_ctffind ' + '\n' 
      relion_ctffindline = 'relion_run_ctffind --i ' + ctffindstarname + ' --o ' + outputstarname + ' --CS ' + str(Cs) + ' --HT ' + str(Voltage) +  ' --ctfWin -1 --AmpCnst ' + str(AmpContrast) +  ' --DStep ' + str(DPixSize) +  ' --XMAG ' + str(Magnification) + ' --Box ' + str(BoxSize) +  ' --dFMin ' + str(LowDefocusLimit) + ' --dFMax ' + str(HighDefocusLimit) + ' --FStep ' + str(DefocusStep) + ' --dAst ' + str(Astigmatism) + ' --ResMin ' + str(LowResLimit) + ' --ResMax ' + str(HighResLimit) + ' --ctffind_exe \"' + PathToCtffind + '  --omp-num-threads 1 --old-school-input\"' 
    # RELION 2.0
    if Relion2 == True:
      outputstarname =  OutputDir + 'results'       
      outputstarname_read = outputstarname + '/micrographs_ctf.star'
      print ':: RELION sub-tomogram averaging :: ' + '\n' + 'Running relion_run_ctffind ' + '\n'
      relion_ctffindline = 'relion_run_ctffind --i ' + ctffindstarname + ' --o ' + outputstarname + ' --CS ' + str(Cs) + ' --HT ' + str(Voltage) +  ' --ctfWin -1 --AmpCnst ' + str(AmpContrast) +  ' --DStep ' + str(DPixSize) +  ' --XMAG ' + str(Magnification) + ' --Box ' + str(BoxSize) +  ' --dFMin ' + str(LowDefocusLimit) + ' --dFMax ' + str(HighDefocusLimit) + ' --FStep ' + str(DefocusStep) + ' --dAst ' + str(Astigmatism) + ' --ResMin ' + str(LowResLimit) + ' --ResMax ' + str(HighResLimit) + ' --ctffind_exe \"' + PathToCtffind + '  --omp-num-threads 1 --old-school-input\"' 

    # If some are unfinished 
    if OnlyDoUnfinishedCTFs == True:
      relion_ctffindline = relion_ctffindline + ' --only_do_unfinished'
    print(relion_ctffindline)
    if ReRunCtffindSkip == False:
      os.system(relion_ctffindline)
    #
    reliontextfile.write(relion_ctffindline + '\n')

    print ':: RELION sub-tomogram averaging :: ' + '\n' + 'CTF Parameters of all tilt series images were estimated using RELION\'s  relion_run_ctffind ' + '\n' 
    print ':: RELION sub-tomogram averaging :: ' + '\n' + 'Parameters have been saved in ' + outputstarname_read + '\n' 
    
  if SkipCTFCorrection == True:
    if Relion2 == False:
      outputstarname_read =  OutputDir + MicRootName +  '_ctffind.star'
    # RELION 2.0
    if Relion2 == True:
      outputstarname_read =  OutputDir + 'results/micrographs_ctf.star'

    ctffindoutstarfile = open(outputstarname_read, 'w')
    ctffindoutstarfile.write('data_' + '\n' + '\n')
    ctffindoutstarfile.write('loop_' + '\n')
    ctffindoutstarfile.write('_rlnMicrographName #1' + '\n')
    ctffindoutstarfile.write('_rlnDefocusU #2' + '\n')
    ctffindoutstarfile.write('_rlnDefocusV #3' + '\n')

    micnames = read_relion_star(ctffindstarname)
    #print micnames
    for kk in range(0,len(micnames)):
      coline = micnames[kk] + '\t' + '0.000' + '\t' + '0.000' + '\n'
      ctffindoutstarfile.write(coline)

    ctffindoutstarfile.close()

  tiltfile.close()

  ##### Running CTFFIND on all images of the tilt series  ##########

  #sys.exit()

  ##### Making .star files for each 3D CTF Volume #################
  
  RelionPartName = 'Particles/'
  RelionPartDir = ScriptDir + RelionPartName
  RelionRecDir = RelionPartDir + MicDirName
  RelionRecFileName =  RelionPartName + MicDirName + MicRootName + '_rec_CTF_volumes.sh'
  RelionRecFileName_for_script =  MicRootName + '_rec_CTF_volumes.sh'

  ## Making a new directory to output the results of CTFFIND
  ensure_dir(RelionRecDir)
  
  coordfile = open(coordsname, 'r')
  relionfile = open(RelionRecFileName, 'w')

  # Getting the tilt order
  print ':: RELION sub-tomogram averaging :: ' + '\n' + 'Reading tilt series order file for dose dependent B-Factor weighting ' + '\n' 
  tiltorderfile = open(ordername, 'r')
  tiltorder=[]
  accumulated_dose=[]

  for line in tiltorderfile:
    
    emptycheck = line.isspace()
    if(emptycheck):
      #print 'empty line found'
      continue
    
    pair=line.split()
    tiltorder.append(float(pair[0]))
    accumulated_dose.append(float(pair[1]))

  #print tiltorder, accumulated_dose
  tiltorderfile.close()
  #

  # Reading the output of CTFFIND
  micnames, avgdefoci, defocusv = read_relion_star(outputstarname_read)
  final_avgdefoci=[]
  if UseTrialsForCtffind == False:
    print 'Using actual tilt images for CTF estimation' + '\n'
    final_avgdefoci = avgdefoci
    #print 'DEBUG', final_avgdefoci, exttilts
  
  if UseTrialsForCtffind == True:
    print 'Using extra trial images for CTF estimation' + '\n'
    for nums in range(0,len(avgdefoci),2):
      defavg = ( float(avgdefoci[nums]) + float(avgdefoci[nums+1]) ) / 2
      final_avgdefoci.append(defavg)
 
    
  # If Higher tilts do not give reliable CTF estimations, then the lower tilts are used for CTF estimation
  print 'Using only lower tilts for CTF correction with the upper limit of ' + str(UseOnlyLowerTiltDefociLimit) + '\n'
  if UseOnlyLowerTiltDefoci == True:
    ct = 0.0
    td = 0.0
    for ii in range(0, len(final_avgdefoci)):
      #print exttilts[ii], final_avgdefoci[ii]
      if abs(exttilts[ii]) < UseOnlyLowerTiltDefociLimit:
        td = td + float(final_avgdefoci[ii])
        ct=ct+1
    avg_lower_tilt_defocus = td/ct
    print 'Average defocus from the lower tilt images below ' + str(UseOnlyLowerTiltDefociLimit) + ' is ' + str(avg_lower_tilt_defocus) + '\n'
        
    for ii in range(0, len(final_avgdefoci)):
      #print exttilts[ii], final_avgdefoci[ii]
      if abs(exttilts[ii]) > UseOnlyLowerTiltDefociLimit:
        final_avgdefoci[ii] = avg_lower_tilt_defocus

  #print 'DEBUG2', final_avgdefoci

  print len(final_avgdefoci)
  #print exttilts
  #print avgdefoci
  #print len(exttilts), len(tiltorder)
  
  #sys.exit()

  if len(tiltorder) != len(exttilts):
    print ':: RELION sub-tomogram averaging :: ' + '\n' + 'The number of images in the aligned stack file and the tilt order file are different. Exiting'
    sys.exit()
  
  if UseTrialsForCtffind == True:
    if len(tiltorder) != len(trial_exttilts)/2:
      print ':: RELION sub-tomogram averaging :: ' + '\n' + 'The number of images in the trial stack and the tilt order file are different. Exiting'
      sys.exit()

  print ':: RELION sub-tomogram averaging :: ' + '\n' + 'The number of images in the CTFFIND output file and the tilt order file are the same. Continuing.'
  
  print ':: RELION sub-tomogram averaging :: ' + '\n' + 'Writing out .star files to make 3D CTF volumes ' + '\n' 

  # Pixelsize calculation
  PixelSize = DPixSize/Magnification*10000
  #print PixelSize

  # getting tomogram size using the IMOD program header
  print ':: RELION sub-tomogram averaging :: ' + '\n' + 'Using IMOD header program to get the size of the tomogram ' + '\n' 
  headerline = 'header -brief -size -input ' + micname 
  print headerline
  status, sizevals = commands.getstatusoutput(headerline)
  tomosize=sizevals.split()
  #print tomosize
  xlimit = float(tomosize[0])
  zlimit = float(tomosize[2])

  print ':: RELION sub-tomogram averaging :: ' + '\n' + 'Writing out .star files to make 3D CTF volumes ' + '\n' 
  subtomonum=0
  for line in coordfile:

    subtomonum = subtomonum+1

    cols = line.split()
    
    # Coordinates of the sub-tomogram in the tomogram
    X = float(cols[0])
    Y = float(cols[1])
    Z = float(cols[2])

    # Output 3D CTF volume and .star file
    if SkipCTFCorrection == False: 
      outstarname = RelionPartName + MicDirName + MicRootName + '_ctf' + str("%06d" % subtomonum) + '.star' 
      #outstarname_for_rec_script = MicRootName + '_ctf' + str("%06d" % subtomonum) + '.star'
      outctfname = RelionPartName + MicDirName + MicRootName + '_ctf' + str("%06d" % subtomonum) + '.mrc' 
      #outctfname_for_rec_script = MicRootName + '_ctf' + str("%06d" % subtomonum) + '.mrc'
      outfile = open(outstarname, 'w')
    
    if SkipCTFCorrection == True and subtomonum == 1:
      outstarname = RelionPartName + MicDirName + MicRootName + '_ctf.star' 
      outctfname = RelionPartName + MicDirName + MicRootName + '_ctf.mrc' 
      outfile = open(outstarname, 'w')
      
    
    # Writing out the header of the ctf star file    
    if not outfile.closed:
      outfile.write('data_images' + '\n')
      outfile.write('loop_' + '\n')
      outfile.write('_rlnDefocusU #1 ' + '\n')
      outfile.write('_rlnVoltage #2 ' + '\n') 
      outfile.write('_rlnSphericalAberration #3 ' + '\n')
      outfile.write('_rlnAmplitudeContrast #4 ' + '\n')
      outfile.write('_rlnAngleRot #5 ' + '\n')
      outfile.write('_rlnAngleTilt #6' + '\n')
      outfile.write('_rlnAnglePsi #7 ' + '\n')
      if Relion2 == False:
        outfile.write('_rlnBfactor #8 ' + '\n')
      # RELION 2.0
      if Relion2 == True:
        outfile.write('_rlnCtfBfactor #8 ' + '\n')
      outfile.write('_rlnCtfScalefactor #9 ' + '\n')

    for j in range(0,len(exttilts)):
      
      avgdefocus = float(final_avgdefoci[j])
      tilt_radians = (exttilts[j]*math.pi/180)
      tilt_degrees = exttilts[j]
      #print tilt_radians, tilt_degrees

      xtomo = float(X - (xlimit/2) )*PixelSize
      ztomo = float(Z - (zlimit/2) )*PixelSize
      #print xtomo, ztomo

      # Calculating the height difference of the particle from the tilt axis
      ximg = (xtomo*(math.cos(tilt_radians))) + (ztomo*(math.sin(tilt_radians)))
      deltaD = ximg*math.sin(tilt_radians)
      ptcldefocus = avgdefocus + deltaD
      if SkipCTFCorrection == True:
        ptcldefocus = avgdefocus        # Should be 0.000
      #print ptcldefocus
      #

      # Now weighting the 3D CTF model using the tilt dependent scale factor and the dose dependent B-Factor
      tiltscale = math.cos(abs(tilt_radians))
      #print tiltscale

      tiltstep = (max(exttilts) - min(exttilts))/(len(exttilts)-1)
      besttiltdiff = tiltstep + 0.5


      for k in range(0,len(tiltorder)):
          
          tiltdiff = abs(tilt_degrees-tiltorder[k])
          
          if tiltdiff < (tiltstep+0.25):
            if tiltdiff < besttiltdiff:
              besttiltdiff = tiltdiff
              accumulated_dose_current = accumulated_dose[k]

      doseweight = accumulated_dose_current * Bfactor
      #print exttilts, tiltorder, accumulated_dose, besttiltdiff, accumulated_dose_current
      #print doseweight
      #

      # Writing parameters in the .star file for each 2D slice of the 3D CTF model volume
      ang_rot = '0.0'
      ang_psi = '0.0'
      if not outfile.closed:
        ctfline =  str("%.2f" % ptcldefocus) + '\t' + str(Voltage) + '\t' + str(Cs) + '\t' + str(AmpContrast) + '\t' + ang_rot + '\t' + str(tilt_degrees) + '\t' + ang_psi + '\t' + str(doseweight) + '\t' + str("%.2f" % tiltscale) + '\n'
        outfile.write(ctfline)

    # This is for parallilzation of the CTF reconstructions
    if not outfile.closed:
      reconstructline2 = 'relion_reconstruct --i ' + outstarname + ' --o ' + outctfname + ' --reconstruct_ctf ' + '$1' + ' --angpix ' + str("%.2f" % PixelSize) + '\n'
      ctfreconstmasterfile.write(reconstructline2)
    
    # writing the .star file for refinement
    if Relion2 == False:
      currentsubtomoname = RelionPartName+ MicDirName +  MicRootName + '_' + RootName + str("%06d" % subtomonum) + '.mrc'
      subtomostarline = micname + '\t' + str(X) + '\t' + str(Y) + '\t' + str(Z) + '\t' + currentsubtomoname + '\t' + outctfname + '\n'
      subtomostarfile.write(subtomostarline)
    # RELION 2.0
    if Relion2 == True:
      currentsubtomoname = 'Extract/' + ExtractJobAlias + '/' + MicDirName +  MicRootName + str("%06d" % subtomonum) + '.mrc'
      subtomostarline = micname + '\t' + str(X) + '\t' + str(Y) + '\t' + str(Z) + '\t' + currentsubtomoname + '\t' + outctfname + '\t' + str(Magnification) + '\t' + str(DPixSize) + '\n'
      subtomostarfile.write(subtomostarline)


    outfile.close()
  
  relionfile.close()
  #ctfreconstmasterfile.write('cd ' + RelionPartName + MicDirName + '\n')
  #ctfreconstmasterfile.write( RelionRecFileName_for_script + ' $1''\n')
  #ctfreconstmasterfile.write('cd ' + ScriptDir + '\n')
  os.chmod(RelionRecFileName, stat.S_IRWXU)
  print ':: RELION sub-tomogram averaging :: ' + '\n' + '.star files to make 3D CTF volumes were written out in ' + RelionRecDir + '\n' 
  print ':: RELION sub-tomogram averaging :: ' + '\n' + 'shell script to reconstruct the 3D CTF volumes is ' + RelionRecFileName + '\n'

  #sys.exit()
  ##### Making .star files for each 3D CTF Volume #################

subtomostarfile.close()
ctfreconstmasterfile.close()
reliontextfile.close()
print ':: RELION sub-tomogram averaging :: ' 
print 'Please extract sub-tomograms using the RELION GUI. Remember to use the same subtomoname as you gave in this script'
print 'Please run the 3D CTF model volume reconstructions using the .sh scripts written in the working directory' 
print 'run this script from the command line with the command '
print 'do_all_reconstruct_ctfs.sh SubtomogramSize '
print 'STAR file to use for refinement (after sub-tomogram extraction and 3D CTF volume reconstruction) was written in ' + subtomostarname 
