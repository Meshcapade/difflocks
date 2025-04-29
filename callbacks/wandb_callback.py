from callbacks.callback import *
import wandb
import hjson

class WandBCallback(Callback):

    def __init__(self, experiment_name, config_path, entity):
        self.experiment_name=experiment_name
        # loading the config file like this and giving it to wandb stores them on the website
        with open(config_path, 'r') as j:
            cfg = hjson.loads(j.read())
        # Before this init can be run, you have to use wandb login in the console you are starting the script from (https://docs.wandb.ai/ref/cli/wandb-login, https://docs.wandb.ai/ref/python/init)
        # entity= your username
        wandb.init(project=experiment_name, entity=entity,config = cfg)
        

    def after_forward_pass(self, phase, loss, loss_kl, **kwargs):

        # / act as seperators. If you would like to log train and test separately you would log test loss in test/loss 
        wandb.log({'train/loss': loss}, step=phase.iter_nr)
        if loss_kl!=0:
            wandb.log({'train/loss_kl': loss_kl}, step=phase.iter_nr)


    def epoch_ended(self, phase, **kwargs):
        pass