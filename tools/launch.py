from pathlib import Path
from nni.experiment import Experiment

'''
search_space = {
    "dropout_rate": { "_type": "uniform", "_value": [0.5, 0.9] },
    "conv_size": { "_type": "choice", "_value": [2, 3, 5, 7] },
    "hidden_size": { "_type": "choice", "_value": [124, 512, 1024] },
    "batch_size": { "_type": "choice", "_value": [16, 32] },
    "learning_rate": { "_type": "choice", "_value": [0.0001, 0.00005] }
}
'''

search_space = {
    "bs":{"_type":"choice","_value":[4, 8, 16, 32]},
    "epoch":{"_type":"choice","_value":[70, 80]},
    "learning_rate": {"_type": "uniform", "_value": [0.00001, 0.000005]}
}

experiment = Experiment('local')
experiment.config.experiment_name = 'fetal_monitoring'
experiment.config.trial_concurrency = 1
experiment.config.max_trial_number = 1000000
experiment.config.search_space = search_space
experiment.config.trial_command = 'python ./run.py --bs 32 --epoch 100 --source ../data/Data_0310'
# experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.training_service.use_active_gpu = True

experiment.run(8080)