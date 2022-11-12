import torch
import time
from datasets import load_dataset
from dataset import TrainDataset, EvalDataset
from model import SimpleSR
from train import Trainer

if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    # print(device)

    # init_time = time.time()

    # train_dataset = TrainDataset(load_dataset('eugenesiow/Div2k', 'bicubic_x2', split='train'))
    # test_dataset = EvalDataset(load_dataset('eugenesiow/Div2k', 'bicubic_x2', split='validation'))

    # ssr = SimpleSR(ch_in=3,upsample_factor=2,n_blocks=5)
    # ssr = ssr.to(device)

    # base_folder = './drive/MyDrive/Purdue/S7/Project/'
    
    from datasets import load_dataset

    base_folder = './drive/MyDrive/Purdue/S7/Project/'
    raw_train = load_dataset('eugenesiow/Div2k', 'bicubic_x2', split='train')
    raw_test = load_dataset('eugenesiow/Div2k', 'bicubic_x2', split='validation')

    raw_train.save_to_disk(f'{base_folder}data/train')
    raw_test.save_to_disk(f'{base_folder}data/test')


    # i = 0
    # state_dict = base_folder + f'weights{i}.pt'
    # weights_name = base_folder + f'weights{i+1}.pt'
    # trainer = Trainer(ssr,train_dataset,test_dataset,epochs=2,batch_size=64,weights_name=weights_name,state_dict_file=state_dict)
    
    # time1 = time.time()
    # trainer.train()
    # time2 = time.time()
    # print(f'Setup time: {time1-init_time}\nRun time: {time2 - time1}')

