
DATASET_PATH = ''
DATASET_DIR = ''
SCHEMA_PATH = ''

class Option:
    def __init__(self, **kwargs) -> None:
        self.batch_size = 8
        self.gradient_descent_step = 6
        self.learning_rate = 1e-4
        self.epochs = 1500
        self.validation_steps = 50
        self.seed = 42

        self.sa_mode = '000'
        self.sa_epochs = 30000
        self.sa_report_interval = 500
        self.validation_sa_steps = 1000

        self.eval_model_suffix = ''

        self.model_cls = 'No'
        self.device = '1'
        self.dataset = 'IMDB'
        self.ql = 'AQL'
        self.use_pir = True
        self.use_skeleton = False
        self.method = 'PolyTrans'
        self.model_name_or_path = ''
        
        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)

        pir_sign = '(PIR)' if self.use_pir else ''
        skeleton_sign = 'Sk' if self.use_skeleton else 'NoSk'
        self.exp_name = f'{self.method}-{self.dataset}-{self.ql}{pir_sign}-{self.model_cls}-{skeleton_sign}'

        self.save_path = f'./models/{self.exp_name}'
        self.use_adafactor = True
        self.mode = 'train'
        self.num_beams = 4
        self.num_return_sequences = 4

        self.output_path = f'./output/{self.exp_name}.log'

        self.nfolds = 4

    def __repr__(self):
        d = ''
        for k in dir(self):
            if not k.startswith('_'):
                d += f'{k}: {getattr(self, k)}\n'
        return d
    