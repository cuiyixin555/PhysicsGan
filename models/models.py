
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
#        assert(opt.dataset_mode == 'aligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'cycle_gan_deblur':
        from .cycle_gan_model_deblur import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'cycle_gan_sr':
        from .cycle_gan_model_sr import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'cycle_gan_dehaze':
        from .cycle_gan_model_dehaze import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
