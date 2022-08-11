class Config(object):

    def __init__(self):

        self.path_prefix = "../probav_data/"
        self.beta = 50.0

        self.device = "cuda"
        self.validate = True

        # QANet architecture 
        self.N_modules = 12
        self.N_resblocks = 8
        self.N_feats = 36
        self.reduction = 16
        self.N_heads = 1
        self.scale = 3
        self.N_lrs = 9
        self.rcab_bn = False

        # learning
        self.batch_size = 10
        self.N_epoch = 520
        self.learning_rate = 1e-4
        self.workers = 4
        self.patch_size = 32

        # logging
        self.val_step = 5