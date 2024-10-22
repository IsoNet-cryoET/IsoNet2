import logging
from IsoNet.preprocessing.cubes import create_cube_seeds,crop_cubes
from IsoNet.utils.utils import mkfolder
from IsoNet.utils.plot_metrics import plot_metrics
from IsoNet.utils.dict2attr import save_args_json,load_args_from_json
from IsoNet.utils.utils import process_gpuID, process_ncpus, process_batch_size, mkfolder
import sys
import numpy as np
import shutil
import os
from IsoNet.preprocessing.prepare import prepare_first_iter
from IsoNet.utils.noise import get_noise_level


def run(params):
    if params.log_level == "debug":
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
    else:
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
        #logging.basicConfig(format='%(asctime)s.%(msecs)03d, %(levelname)-8s %(message)s',
        #datefmt="%Y-%m-%d,%H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
    try:
        logging.info('\n######Isonet starts refining######\n')

        params = parse_args(params)

        ###  find current iteration ###        
        current_iter = params.iter_count if hasattr(params, "iter_count") else 1
        if params.continue_from is not None:
            current_iter += 1

        from IsoNet.models.network import Net
        network = Net()

        ###  Main Loop ###
        ###  1. find network model file ###
        ###  2. prediction if network found ###
        ###  3. prepare training data ###
        ###  4. training and save model file ###
        print(current_iter)
        for num_iter in range(current_iter,params.iterations + 1):        
            logging.info("Start Iteration{}!".format(num_iter))

            # ### TODO Select a subset of subtomos, useful when the number of subtomo is too large ###
            # if params.select_subtomo_number is not None:
            #     params.mrc_list = np.random.choice(params.mrc_list, size = int(params.select_subtomo_number), replace = False)


            ### Update the iteration count ###
            params.iter_count = num_iter

            if params.pretrained_model is not None:
                ### use pretrained model ###
                mkfolder(params.result_dir)  
                shutil.copyfile(params.pretrained_model,'{}/model_iter{:0>2d}.h5'.format(params.result_dir,num_iter-1))
                logging.info('Use Pretrained model as the output model of iteration {} and predict subtomograms'.format(num_iter-1))
                params.pretrained_model = None
                network.predict_subtomos(params)
            elif params.continue_from is not None:
                ### Continue from a json file ###
                logging.info('Continue from previous model: {}/model_iter{:0>2d}.h5 of iteration {} and predict subtomograms \
                '.format(params.result_dir,num_iter -1,num_iter-1))
                params.continue_from = None
                network.predict_subtomos(params)
            elif num_iter == 1:
                ### First iteration ###
                mkfolder(params.output_dir,remove=False)
                #prepare_first_model(params)
                print("here")
                prepare_first_iter(params)
            else:
                ### Subsequent iterations for all conditions ###
                network.predict_subtomos(params)

            #params.init_model = "{}/model_iter{:0>2d}.h5".format(params.output_dir, num_iter-1)
           
            ### Noise settings ###
            num_noise_volume = 1000
            if num_iter>=params.noise_start_iter[0] and (not os.path.isdir(params.noise_dir) or len(os.listdir(params.noise_dir))< num_noise_volume ):
                from IsoNet.utils.noise import make_noise_folder
                print(params.noise_mode)
                make_noise_folder(params.noise_dir,params.noise_mode,params.cube_size,num_noise_volume,ncpus=params.ncpus)
            noise_level_series = get_noise_level(params.noise_level,params.noise_start_iter,params.iterations)
            params.noise_level_current =  noise_level_series[num_iter]
            logging.info("Noise Level:{}".format(params.noise_level_current))

            ### remove data_dir and generate training data in data_dir###
            try:
                shutil.rmtree(params.data_dir)     
            except OSError:
                pass
            get_cubes_list(params)
            logging.info("Done preparing subtomograms!")

            ### remove all the mrc files in results_dir ###
            if params.remove_intermediate is True:
                logging.info("Remove intermediate files in iteration {}".format(params.iter_count-1))
                for mrc in params.mrc_list:
                    root_name = mrc.split('/')[-1].split('.')[0]
                    current_mrc = '{}/{}_iter{:0>2d}.mrc'.format(params.result_dir,root_name,params.iter_count-1)
                    os.remove(current_mrc)

            ### start training and save model and json ###
            logging.info("Start training!")
            network.train(params.data_dir, 
                                    params.output_dir, 
                                    batch_size=params.batch_size,
                                    outmodel_path='{}/model_iter{:0>2d}.pt'.format(params.output_dir,params.iter_count),
                                    epochs = params.epochs, 
                                    steps_per_epoch=params.steps_per_epoch, 
                                    acc_batches = 1,
                                    mixed_precision=False,
                                    learning_rate=params.learning_rate) #train based on init model and save new one as model_iter{num_iter}.h5
            params.losses = network.metrics['average_loss']
            save_args_json(params,params.output_dir+'/refine_iter{:0>2d}.json'.format(num_iter))
            logging.info("Done training!")

            ### for last iteration predict subtomograms ###
            if num_iter == params.iterations and params.remove_intermediate == False:
                logging.info("Predicting subtomograms for last iterations")
                params.iter_count +=1 
                network.predict_subtomos(params)
                params.iter_count -=1 

            logging.info("Done Iteration{}!".format(num_iter))

    except Exception:
        import traceback
        error_text = traceback.format_exc()
        f =open('log.txt','a+')
        f.write(error_text)
        f.close()
        logging.error(error_text)
        #logging.error(exc_value)


def parse_args(args):
    '''
    Consume all the argument parameters
    '''

    if args.continue_from is not None:
        logging.info('\n######Isonet Continues Refining######\n')
        args_continue = load_args_from_json(args.continue_from)
        for item in args_continue.__dict__:
            if args_continue.__dict__[item] is not None and (args.__dict__ is None or not hasattr(args, item)):
                args.__dict__[item] = args_continue.__dict__[item]


    if args.gpuID is None:
        import torch
        gpu_list = list(range(torch.cuda.device_count()))
        args.gpuID=','.join(map(str, gpu_list))
        print("using all GPUs in this node: %s" %args.gpuID)  
    ngpus, args.gpuID, gpuID_list = process_gpuID(args.gpuID)
    args.ncpus = process_ncpus(args.ncpus)
    args.batch_size = process_batch_size(args.batch_size, ngpus)

    #environment
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuID

    import starfile
    star = starfile.read(args.subtomo_star)
    #*******set fixed parameters*******
    args.crop_size = star.loc[0,'rlnCropSize']
    args.cube_size = star.loc[0,'rlnCubeSize']

    #*******calculate parameters********

    if args.data_dir is None:
        args.data_dir = args.output_dir + '/data'
    if args.iterations is None:
        args.iterations = 30
    if args.output_dir is None:
        args.output_dir = 'results'
    if args.batch_size is None:
        args.batch_size = max(4, 2 * ngpus)
    if args.filter_base is None:
        args.filter_base = 64
    if args.steps_per_epoch is None:
        if args.select_subtomo_number is None:
            args.steps_per_epoch = min(int(len(star) * 6/args.batch_size) , 200)
        else:
            args.steps_per_epoch = min(int(int(args.select_subtomo_number) * 6/args.batch_size) , 200)
    if args.learning_rate is None:
        args.learning_rate = 0.0004

    if args.noise_mode is None:
        args.noise_mode = 'noFilter'
    if args.noise_dir is None:
        args.noise_dir = args.output_dir +'/training_noise'
    if args.log_level is None:
        args.log_level = "info"

    if len(star) <=0:
        logging.error("Subtomo list is empty!")
        sys.exit(0)
    args.mrc_list = []
    if "rlnParticleName" in star.columns.to_list():
        for i,it in enumerate(star.iterrows()):
            args.mrc_list.append(it[1]['rlnParticleName'])
    return args











def run_training(data_list, epochs = 10, mixed_precision = False,
               output_dir = "results", outmodel_path="half", pretrained_model=None,
               ncpus=16, batch_size = 8, acc_batches=2, learning_rate= 3e-4):

    logging.info("Start training!")
    from IsoNet.models.network import Net
    network = Net(filter_base = 64,unet_depth=3, add_last=True)
    if pretrained_model is not None:
       print(f"loading previous model {pretrained_model}")
       network.load(pretrained_model)
    if epochs > 0:
       network.train(data_list, output_dir, outmodel_path=outmodel_path, batch_size=batch_size, epochs = epochs, steps_per_epoch = 1000, 
                           mixed_precision=mixed_precision, acc_batches=acc_batches, learning_rate = learning_rate) #train based on init model and save new one as model_iter{num_iter}.h5
    plot_metrics(network.metrics, f"{output_dir}/loss.png")
    return network

def run_old(star_file, epochs = 10, mixed_precision = False,
               output_dir = "results", output_base="half", n_subvolume = 50, pretrained_model=None,
               cube_size = 64, predict_crop_size=96, batch_size = 8, acc_batches=2, learning_rate= 3e-4, limit_res=None):


    logging.info("Start training!")
    from IsoNet.models.network import Net
    network = Net(filter_base = 64,unet_depth=3, add_last=True)
    if pretrained_model is not None:
        print(f"loading previous model {pretrained_model}")
        network.load(pretrained_model)
    if epochs > 0:
        network.train(star_file, output_dir, alpha=1, output_base=output_base, batch_size=batch_size, epochs = epochs, steps_per_epoch = 1000, 
                            mixed_precision=mixed_precision, acc_batches=acc_batches, learning_rate = learning_rate) #train based on init model and save new one as model_iter{num_iter}.h5
    plot_metrics(network.metrics, f"{output_dir}/loss_{output_base}.png")

    logging.info("Start predicting!")           
    #out_map = network.predict_map(halfmap, output_dir=output_dir, cube_size = cube_size, crop_size=predict_crop_size, output_base=output_base)

    if limit_res is None:
        out_name = f"{output_dir}/corrected_{output_base}.mrc"
    else:
        out_name = f"{output_dir}/corrected_{output_base}_filtered.mrc"


import starfile
from IsoNet.preprocessing.prepare import get_cubes_list
def run_refine(star_file, epochs = 10, mixed_precision = False,
               output_dir = "results", output_base="half", n_subvolume = 50, pretrained_model=None,
               cube_size = 64,ncpus=16, predict_crop_size=96, batch_size = 8, acc_batches=2, learning_rate= 3e-4, limit_res=None):
    star = starfile.read(star_file)

    mkfolder(output_dir)
    data_dir = output_dir+"/tmpdata"
    mkfolder(data_dir)
    get_cubes_list(star, data_dir,ncpus=ncpus)
    logging.info("Start training!")
    from IsoNet.models.network import Net
    network = Net(filter_base = 64,unet_depth=3, add_last=False)
    if pretrained_model is not None:
       print(f"loading previous model {pretrained_model}")
       network.load(pretrained_model)
    if epochs > 0:
       network.train(data_dir, output_dir, output_base=output_base, batch_size=batch_size, epochs = epochs, steps_per_epoch = 1000, 
                           mixed_precision=mixed_precision, acc_batches=acc_batches, learning_rate = learning_rate) #train based on init model and save new one as model_iter{num_iter}.h5
    plot_metrics(network.metrics, f"{output_dir}/loss_{output_base}.png")

