from run_InfoGAN import *


def test_InfoGAN(noises, conti_codes, discr_codes, InfoGAN_Gen_path, save_experiments_folder, use_gpu):
    InfoGAN_Gen = torch.load(InfoGAN_Gen_path)
    """train InfoGAN and print out the losses for D and G"""
    if use_gpu:
        noises = noises.cuda()
        conti_codes = conti_codes.cuda()
        discr_codes = discr_codes.cuda()

        # generate fake images (learn from data distribution)
    gen_inputs = torch.cat((noises, conti_codes, discr_codes), 1)
    fake_inputs = InfoGAN_Gen(gen_inputs)
    fake_image = tensor2im(fake_inputs.data)
    cv2.imshow('generate_image', fake_image)
    cv2.imwrite(save_experiments_folder + 'test.jpg', fake_image)
    cv2.waitKey()
    while cv2.waitKey() != 27:
        pass


def run_test_InfoGAN(noises, conti_codes, discr_codes,
                     save_model_folder, test_model_name, save_experiments_folder, use_gpu):
    if not os.path.isdir(save_experiments_folder):
        os.mkdir(save_experiments_folder)

    InfoGAN_Gen_path = save_model_folder + test_model_name
    # _, testloader = load_dataset(batch_size=batch_size)
    test_InfoGAN(noises, conti_codes, discr_codes, InfoGAN_Gen_path, save_experiments_folder, use_gpu)


if __name__ == '__main__':
    batch_size = 10
    n_discrete = 1
    noise_dim = 10
    num_category = 10
    n_conti = 2
    mean = 0.0
    std = 0.5
    noises = Variable(gen_noise(batch_size, n_dim=noise_dim))

    # discr_codes = Variable(gen_discrete_code(batch_size, n_discrete,
    #                                          num_category))

    conti_codes = Variable(gen_conti_codes(batch_size, n_conti,
                                           mean, std))

    for i in range (0,10):
        discr_codes = np.zeros((batch_size, num_category))

        # control codes here
        # discr_codes[i] = 1
        # discr_codes[1] = 1
        # discr_codes[3] = 1
        # discr_codes[5] = 1
        discr_codes[i][0] = 0
        discr_codes[i][1] = 10

        discr_codes = Variable(torch.Tensor(discr_codes))
        print conti_codes.data
        print discr_codes.data

        run_test_InfoGAN(noises, conti_codes, discr_codes,
                         save_model_folder='./models/',
                         test_model_name='7_checkpoint.pth',
                         save_experiments_folder='./experiments/',
                         use_gpu=True)