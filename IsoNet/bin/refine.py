import logging
from IsoNet.utils.fileio import create_folder
from IsoNet.utils.dict2attr import save_args_json
import sys
import shutil
import os
from IsoNet.preprocessing.prepare import prepare_first_iter, extract, get_cubes_list
from IsoNet.utils.noise import get_noise_level
def run(params):
    if params["log_level"] == "debug":
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
    else:
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
    try:
        logging.info('\n######Isonet starts refining######\n')

        ### Extract first
        params["mrc_list"] = extract(params["star_file"],
            input_column = params["input_column"],
            subtomo_dir = params["output_dir"] + "/subtomos", 
            cube_size = params["cube_size"],
            crop_size = params["crop_size"], 
            tomo_idx = None,
            uniform_extract=False)

        ###  find current iteration ###        
        current_iter = params["iter_count"] if "iter_count" in params else 1
        if params["continue_from"] is not None:
            current_iter += 1

        from IsoNet.models.network import Net
        network = Net(method="regular", arch=params["arch"])
        ###  Main Loop ###
        ###  1. find network model file ###
        ###  2. prediction if network found ###
        ###  3. prepare training data ###
        ###  4. training and save model file ###
        for num_iter in range(current_iter, params["iterations"] + 1):        
            logging.info("Start Iteration{}!".format(num_iter))

            # ### TODO Select a subset of subtomos, useful when the number of subtomo is too large ###
            # if params["select_subtomo_number"] is not None:
            #     params["mrc_list"] = np.random.choice(params["mrc_list"], size = int(params["select_subtomo_number"]), replace = False)

            ### Update the iteration count ###
            params["iter_count"] = num_iter

            if params["pretrained_model"] is not None:
                ### use pretrained model ###
                create_folder(params["result_dir"])  
                shutil.copyfile(params["pretrained_model"], '{}/model_iter{:0>2d}.h5'.format(params["result_dir"], num_iter-1))
                logging.info('Use Pretrained model as the output model of iteration {} and predict subtomograms'.format(num_iter-1))
                params["pretrained_model"] = None
                network.predict_subtomos(params)
            elif params["continue_from"] is not None:
                ### Continue from a json file ###
                logging.info('Continue from previous model: {}/model_iter{:0>2d}.h5 of iteration {} and predict subtomograms \
                '.format(params["result_dir"], num_iter -1, num_iter-1))
                params["continue_from"] = None
                network.predict_subtomos(params)
            elif num_iter == 1:
                ### First iteration ###
                create_folder(params["output_dir"],remove=False)
                prepare_first_iter(params)
            else:
                ### Subsequent iterations for all conditions ###
                network.predict_subtomos(params)

            ### Noise settings ###
            num_noise_volume = 1000
            if num_iter >= params["noise_start_iter"][0] and (not os.path.isdir(params["noise_dir"]) or len(os.listdir(params["noise_dir"])) < num_noise_volume ):
                from IsoNet.utils.noise import make_noise_folder
                print(params["noise_mode"])
                make_noise_folder(params["noise_dir"], params["noise_mode"], params["cube_size"], num_noise_volume, ncpus=params["ncpus"])
            noise_level_series = get_noise_level(params["noise_level"], params["noise_start_iter"], params["iterations"])
            params["noise_level_current"] = noise_level_series[num_iter]
            logging.info("Noise Level:{}".format(params["noise_level_current"]))

            ### remove data_dir and generate training data in data_dir ###
            try:
                shutil.rmtree(params["data_dir"])     
            except OSError:
                pass
            get_cubes_list(params)
            logging.info("Done preparing subtomograms!")

            ### remove all the mrc files in results_dir ###
            if params["remove_intermediate"] is True:
                logging.info("Remove intermediate files in iteration {}".format(params["iter_count"]-1))
                for mrc in params["mrc_list"]:
                    root_name = mrc.split('/')[-1].split('.')[0]
                    current_mrc = '{}/{}_iter{:0>2d}.mrc'.format(params["result_dir"], root_name, params["iter_count"]-1)
                    os.remove(current_mrc)

            ### start training and save model and json ###
            logging.info("Start training!")

            training_params = {
                "method": "regular",
                "data_path": params["data_dir"],
                "outmodel_path": '{}/model_iter{:0>2d}.pt'.format(params["output_dir"], params["iter_count"]),
                "batch_size": params["batch_size"],
                "acc_batches": 1,
                "epochs": params["epochs"],
                "steps_per_epoch": params["steps_per_epoch"],
                "learning_rate": params["learning_rate"],
                "mixed_precision": False,
                "compile_model": False,
                "ncpus": params["ncpus"],
                "T_max": 10,
                "learning_rate_min": 3e-4,
                "learning_rate_max": 3e-4,
                "loss_func": "L1",
                "correct_CTF": False,
                "output_dir": params["output_dir"],
                "arch": params["arch"],
                "cube_size": params["cube_size"],
                "split": "full",
                "starting_epoch": 0,
            }
            network.train(training_params) #train based on init model and save new one as model_iter{num_iter}.h5
            params["losses"] = network.metrics["average_loss"]
            save_args_json(params, params["output_dir"] + '/refine_iter{:0>2d}.json'.format(num_iter))
            logging.info("Done training!")

            ### for last iteration predict subtomograms ###
            if num_iter == params["iterations"] and params["remove_intermediate"] is False:
                logging.info("Predicting subtomograms for last iterations")
                params["iter_count"] += 1 
                network.predict_subtomos(params)
                params["iter_count"] -= 1 

            logging.info("Done Iteration{}!".format(num_iter))

    except Exception:
        import traceback
        error_text = traceback.format_exc()
        f = open('log.txt', 'a+')
        f.write(error_text)
        f.close()
        logging.error(error_text)
