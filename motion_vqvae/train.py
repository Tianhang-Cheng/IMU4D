import argparse
import json
import numpy as np
import os
import pdb
import random
import time
import torch
import glob

from .model import vqvae
from time import gmtime, strftime
from .process_data_local import create_dataloaders
import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg') # switch to non-interactive backend

imu_meaning = {0: 'acc_x', 1: 'acc_y', 2: 'acc_z',
               3: 'gyro_x', 4: 'gyro_y', 5: 'gyro_z',
               6: 'mag_x', 7: 'mag_y', 8: 'mag_z',
               
               9: 'rot_x', 10: 'rot_y', 11: 'rot_z',
               12: 'x', 13: 'y', 14: 'z',

               15: 'rot_x', 16: 'rot_y', 17: 'rot_z',
               18: 'x', 19: 'y', 20: 'z',
               }

def main(device, config, save_dir, data_init_loc, args):
    # Create/overwrite checkpoints folder and results folder
    # if os.path.exists(os.path.join(save_dir, 'checkpoints')):
    #     print('Checkpoint Directory Already Exists - if continue will overwrite files inside. Press c to continue.')
    #     pdb.set_trace()
    # else:
    #     os.makedirs(os.path.join(save_dir, 'checkpoints'))
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'viz'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)
    # logger.log_parameters(config)

    # Run start training
    vqvae_config, summary = start_training(device=device, vqvae_config=config['vqvae_config'],
                                           save_dir=save_dir,
                                           data_init_loc=data_init_loc, args=args)

    # Save config file
    config['vqvae_config'] = vqvae_config
    print('CONFIG FILE TO SAVE:', config)

    # Create Configs folder
    if os.path.exists(os.path.join(save_dir, 'configs')):
        print('Saved Config Directory Already Exists - if continue will overwrite files inside. Press c to continue.')
        pdb.set_trace()
    else:
        os.makedirs(os.path.join(save_dir, 'configs'))

    # Save the json copy
    with open(os.path.join(save_dir, 'configs', 'config_file.json'), 'w+') as f:
        json.dump(config, f, indent=4)

    # Save the Master File
    summary['log_path'] = os.path.join(save_dir)
    master['summaries'] = summary
    print('MASTER FILE:', master)
    with open(os.path.join(save_dir, 'master.json'), 'w') as f:
        json.dump(master, f, indent=4)


def start_training(device, vqvae_config, save_dir, data_init_loc, args):
    # Create summary dictionary
    summary = {}
    general_seed = args.seed
    summary['general_seed'] = general_seed
    torch.manual_seed(general_seed)
    random.seed(general_seed)
    np.random.seed(general_seed)

    torch.backends.cudnn.deterministic = False

    summary['data initialization location'] = data_init_loc
    summary['device'] = device  # add the cpu/gpu to the summary

    # Setup model
    model = vqvae(vqvae_config)  # Initialize model
    
    print('Total # trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    args.init_epoch = 0

    continue_training = True
    successfully_loaded = False
    if continue_training:
        import glob
        pretrain_paths = glob.glob(os.path.join(save_dir, 'checkpoints')+'/*.pth')
        pretrain_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # print('pretrain_paths:', pretrain_paths)
        if len(pretrain_paths) == 0:
            print('No pretrained model found. Please check the save directory.')
        else:
            # model = torch.load(pretrain_paths[-1])
            pretrained_state = torch.load(pretrain_paths[-1], map_location='cpu', weights_only=False)
            # import pdb;pdb.set_trace()
            init_epoch = int(pretrain_paths[-1].split('_')[-3])
            args.init_epoch = init_epoch
            model_dict = model.state_dict()
            for name, param in model_dict.items():
                if name in pretrained_state and pretrained_state[name].shape == param.shape:
                    model_dict[name] = pretrained_state[name]
                else:
                    print(f'Parameter {name} not found in pretrained state or shape mismatch')
            model.load_state_dict(model_dict)
            print(f'Successfully loaded pretrained model! Init epoch {init_epoch}')
            successfully_loaded = True

    if not successfully_loaded:
        # If not continuing training, we need to initialize the model
        pretrained_path = f'motion_vqvae/pretrained_weight/{compression_rate}_init.pth'
        # pretrained needs to be the path to the trained model if you want it to load
        pretrained_state = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        
        old_code_book_size = pretrained_state['vq._embedding.weight'].data.shape[0]
        new_code_book_size = model.vq._embedding.weight.data.shape[0]
        assert new_code_book_size % old_code_book_size == 0, f'new codebook size {new_code_book_size} is not divisible by old codebook size {old_code_book_size}'
        ratio = new_code_book_size // old_code_book_size

        pretrained_embedding = pretrained_state['vq._embedding.weight'].data.clone()
        copies = [pretrained_embedding]
        for i in range(1, ratio):  # 2nd, 3rd, 4th copy
            noise_strength = i / ratio
            noise = torch.randn_like(pretrained_embedding) * noise_strength * 0
            copies.append(pretrained_embedding + noise)
        new_embedding = torch.cat(copies, dim=0)
        assert new_embedding.shape == (new_code_book_size, 64)

        model.vq._embedding.weight.data.copy_(new_embedding)
        model_dict = model.state_dict()

        for name, param in model_dict.items():
            if name == 'vq._embedding.weight':
                # model_dict[name] = new_embedding
                continue
            elif name in pretrained_state:
                if pretrained_state[name].shape == param.shape:
                    model_dict[name] = pretrained_state[name]
            else:
                print(f'Parameter {name} not found in pretrained state')

        model.load_state_dict(model_dict)
        print("Successfully loaded embedding and matching layers!")

    summary['vqvae_config'] = vqvae_config  # add the model information to the summary

    # Start training the model
    start_time = time.time()
    model = train_model(model, device, vqvae_config, save_dir, args=args)

    # Save full pytorch model
    torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoints/final_model.pth'))

    # Save and return
    summary['total_time'] = round(time.time() - start_time, 3)
    return vqvae_config, summary


def train_model(model, device, vqvae_config, save_dir, args):
    # Set the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    # Setup model (send to device, set to train)
    model.to(device)
    start_time = time.time()
    print('BATCHSIZE:', args.batchsize)

    window_size = vqvae_config['compression_factor']

    if vqvae_config['compression_factor'] == 12:
        seq_length = 36
    elif vqvae_config['compression_factor'] == 16:
        seq_length = 48
    else:
        seq_length = 40
    dataloaders = create_dataloaders(batch_size=args.batchsize, seq_length=seq_length, val_ratio=0.05)

    train_loader = dataloaders['train']
    vali_loader = dataloaders['val']

    print('train_loader:', len(train_loader))
    print('vali_loader:', len(vali_loader))
    
    # do + 0.5 to ciel it
    total_epoch = 5
    for epoch in tqdm.tqdm(range(args.init_epoch, total_epoch)):
        model.train()

        for i, data in tqdm.tqdm(enumerate(train_loader)):
            bs, ntime, nvars = data.shape

            data = data.float().to(device)
            
            # mask = mask.bool().to(device)
            # mask = mask[..., None].expand(-1, -1, nvars)

            # random mask
            # B, T = bs*nvars, ntime
            # mask = torch.rand((B, T)).to(device)
            # mask[mask <= args.mask_ratio] = 0  # masked
            # mask[mask > args.mask_ratio] = 1  # remained
            # inp = batch_x_cuda.masked_fill(mask == 0, 0)

            batch = data.reshape(bs, -1, window_size, nvars)  # [bs, n_window, window_size, nvars]
            n_window = batch.shape[1]
            batch_mean = batch.mean(dim=2, keepdim=True) # [bs, n_window, 1, nvars]
            batch_mean = batch_mean.expand(-1, -1, window_size, -1)
            batch_std = batch.std(dim=2, keepdim=True, unbiased=False).clamp(min=1e-5)
            batch_std = batch_std.expand(-1, -1, window_size, -1)
            batch_norm = (batch - batch_mean) / batch_std # [bs, n_window, window_size, nvars]

            batch = batch.reshape(bs, ntime, nvars)  # [bs, ntime, nvars]
            batch = torch.permute(batch, (0, 2, 1)).reshape(bs*nvars, ntime)  # [bs * nvars, ntime]

            batch_norm = batch_norm.reshape(bs, ntime, nvars)  # [bs, ntime, nvars]
            batch_norm = torch.permute(batch_norm, (0, 2, 1)).reshape(bs*nvars, ntime)  # [bs * nvars, ntime]
            batch_mean = batch_mean.reshape(bs, ntime, nvars)
            batch_mean = torch.permute(batch_mean, (0, 2, 1)).reshape(bs*nvars, ntime)  # [bs * nvars, ntime]
            batch_std = batch_std.reshape(bs, ntime, nvars)
            batch_std = torch.permute(batch_std, (0, 2, 1)).reshape(bs*nvars, ntime)  # [bs * nvars, ntime]

            input_dict = {'batch': batch, 'batch_norm':batch_norm, 'mean': batch_mean, 'std': batch_std}
            loss, vq_loss, recon_error, x_recon, x_indices, perplexity = model(input_dict, optimizer, 'train')

            if i % 10 == 0:
                print(f'Iter:{i}, Epoch: {epoch}, Loss: {loss.item():.4f}, Recon Error: {recon_error.item():.4f}, VQ Loss: {vq_loss.item():.4f}, Perplexity: {perplexity.item():.4f}')
                with open(os.path.join(save_dir, f'logs/train_recon_error.txt'), 'a') as f:
                    f.write(f'{recon_error.item()}\n')
                with open(os.path.join(save_dir, f'logs/train_vq_loss.txt'), 'a') as f:
                    f.write(f'{vq_loss.item()}\n')
                with open(os.path.join(save_dir, f'logs/train_perplexity.txt'), 'a') as f:
                    f.write(f'{perplexity.item()}\n')

                recon_loss_read = np.loadtxt(os.path.join(save_dir, f'logs/train_recon_error.txt'))
                fig = plt.figure(figsize=(10, 5))
                plt.plot(recon_loss_read, label='train recon error')
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.title('Train Recon Loss')
                plt.savefig(os.path.join(save_dir, f'logs/train_loss.png'))
                plt.close()

                vq_loss_read = np.loadtxt(os.path.join(save_dir, f'logs/train_vq_loss.txt'))
                fig = plt.figure(figsize=(10, 5))
                plt.plot(vq_loss_read, label='train vq loss')
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.title('Train VQ Loss')
                plt.savefig(os.path.join(save_dir, f'logs/train_vq_loss.png'))
                plt.close()

                perplexity_read = np.loadtxt(os.path.join(save_dir, f'logs/train_perplexity.txt'))
                fig = plt.figure(figsize=(10, 5))
                plt.plot(perplexity_read, label='train perplexity')
                plt.xlabel('epoch')
                plt.ylabel('perplexity')
                plt.title('Train Perplexity')
                plt.savefig(os.path.join(save_dir, f'logs/train_perplexity.png'))
                plt.close()

            if i % 50 == 0:
                val_recon_errors = []
                with (torch.no_grad()):
                    model.eval()
                    for val_i, data in enumerate(vali_loader):
                        data = data.to(device)
                        bs, ntime, nvars = data.shape

                        batch = data.reshape(bs, -1, window_size, nvars)  # [bs, n_window, window_size, nvars]
                        n_window = batch.shape[1]
                        batch_mean = batch.mean(dim=2, keepdim=True) # [bs, n_window, 1, nvars]
                        batch_mean = batch_mean.expand(-1, -1, window_size, -1)
                        batch_std = batch.std(dim=2, keepdim=True, unbiased=False).clamp(min=1e-5)
                        batch_std = batch_std.expand(-1, -1, window_size, -1)
                        batch_norm = (batch - batch_mean) / batch_std # [bs, n_window, window_size, nvars]

                        batch = batch.reshape(bs, ntime, nvars)  # [bs, ntime, nvars]
                        batch = torch.permute(batch, (0, 2, 1)).reshape(bs*nvars, ntime)  # [bs * nvars, ntime]

                        batch_norm = batch_norm.reshape(bs, ntime, nvars)  # [bs, ntime, nvars]
                        batch_norm = torch.permute(batch_norm, (0, 2, 1)).reshape(bs*nvars, ntime)  # [bs * nvars, ntime]
                        batch_mean = batch_mean.reshape(bs, ntime, nvars)
                        batch_mean = torch.permute(batch_mean, (0, 2, 1)).reshape(bs*nvars, ntime)  # [bs * nvars, ntime]
                        batch_std = batch_std.reshape(bs, ntime, nvars)
                        batch_std = torch.permute(batch_std, (0, 2, 1)).reshape(bs*nvars, ntime)  # [bs * nvars, ntime]

                        input_dict = {'batch': batch, 'batch_norm': batch_norm, 'mean': batch_mean, 'std': batch_std}
                        val_loss, val_vq_loss, val_recon_error, val_x_recon, val_x_indices, val_perplexity = model(input_dict, optimizer, 'val')

                        val_x_recon_reshape = val_x_recon.reshape(bs, nvars, ntime).permute(0, 2, 1).cpu().numpy() # [bs, ntime, nvars]
                        # val_x_recon_reshape = func(val_x_recon_reshape)
                        # batch_x_gt_reshape = func(batch_x.numpy())
                        batch_x_gt_reshape = batch.reshape(bs, nvars, ntime).permute(0, 2, 1).cpu().numpy() # [bs, ntime, nvars]

                        mse_loss = (val_x_recon_reshape - batch_x_gt_reshape) ** 2
                        mse_loss = np.mean(mse_loss, axis=(1,2))
                        val_recon_errors.append(mse_loss)

                        for seq_index in [0,1]:
                            os.makedirs(os.path.join(save_dir, f'viz/{seq_index}'), exist_ok=True)
                            fig, axes = plt.subplots(7, 7, figsize=(20, 20))
                            fig.suptitle('Original vs Predicted for Each Variable', fontsize=16)
                            for j, var_idx in enumerate(range(49)):
                                row, col = divmod(j, 7)
                                ax = axes[row, col]

                                a = batch_x_gt_reshape[seq_index, :, var_idx]
                                b = val_x_recon_reshape[seq_index, :, var_idx]

                                ax.plot(a, label='original')
                                ax.plot(b, label='predicted')
                                ax.set_title(f'Variable {var_idx}')
                                # ax.set_title(f'{imu_meaning[var_idx]}')

                                ax.legend()

                            plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for the suptitle
                            # plt.show()
                            # plt.savefig(os.path.join(save_dir, f'viz/epoch_{epoch}_iter_{i}_seq_{seq_index}.png'))
                            # plt.savefig(os.path.join(save_dir, f'viz/epoch_{epoch}_iter_{i}.png'))
                            plt.savefig(os.path.join(save_dir, f'viz/{seq_index}/epoch_{epoch}_iter_{i}.png'))
                            plt.close()
                        break
                    
                    val_recon_errors = np.concatenate(val_recon_errors, axis=0)
                    val_recon_errors = np.mean(val_recon_errors, axis=0)
                    with open(os.path.join(save_dir, f'logs/val_recon_errors.txt'), 'a') as f:
                        f.write(f'{val_recon_errors}\n')
                    print(f'Epoch: {epoch}, Iter: {i}, Val Recon Error: {val_recon_errors.item():.4f}')

                    val_recon_errors_read = np.loadtxt(os.path.join(save_dir, f'logs/val_recon_errors.txt'))
                    fig = plt.figure(figsize=(10, 5))
                    plt.plot(val_recon_errors_read, label='val recon error')
                    plt.xlabel('epoch')
                    plt.ylabel('mse error')
                    plt.title('Val Recon Error')
                    plt.savefig(os.path.join(save_dir, f'logs/val_recon_errors.png'))
                    plt.close()

            if i % 300 == 0:
                checkpoints_total_limit = 3
                checkpoints = sorted(glob.glob(os.path.join(save_dir, 'checkpoints/*.pth')), key=os.path.getmtime)
                if len(checkpoints) > checkpoints_total_limit:
                    os.remove(checkpoints[0])
                # save statedict
                torch.save(model.state_dict(), os.path.join(save_dir, f'checkpoints/model_epoch_{epoch}_iter_{i}.pth'))
                print(f'Saved model from epoch {epoch} iter {i}.')

    print('total time: ', round(time.time() - start_time, 3))
    return model


if __name__ == '__main__':
    #create argument parser to read in from the python terminal call
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=False, default='',
                        help='path to specific config file once already in the config folder')
    parser.add_argument('--model_init_num_gpus', type=int, required=False, default=0,
                        help='number of gpus to use, 0 indexed, so if you want 1 gpu say 0')
    parser.add_argument('--data_init_cpu_or_gpu', type=str, required=False,
                        help='the data initialization location')
    parser.add_argument('--save_path', type=str, required=False,
                        help='where were going to save the checkpoints')
    parser.add_argument('--batchsize', type=int,
                        # default=1024,
                        default=500,
                        # default=4096,
                        # default=512,
                        help='batchsize')
    parser.add_argument('--compression_rate', type=int, default=4, help='compression rate for the vqvae model')

    parser.add_argument('--mask_ratio', type=float,help='how much to mask the data', default=0.)

    parser.add_argument('--revined_data', type=str,  help='if true use revin, if false do something else', default='false')

    parser.add_argument('--seed', type=int,  help='the seed to use', default=42)

    # if use imu data, call this to add imu data
    parser.add_argument('--large', type=str, default='false', help='if true, use large model, if false, use small model')

    args = parser.parse_args()

    compression_rate = int(args.compression_rate)

    # Get config file
    config_file = f'motion_vqvae/configs/train_{compression_rate}.json'
    print('Config folder:\t {}'.format(config_file))

    # Load JSON config file
    with open(config_file, 'r') as f:
        config = json.load(f)
    print('Running Config:', config_file)

    # save directory --> will be identically named to config structure
    save_folder_name = ('EDIM' + str(config['vqvae_config']['embedding_dim']) +
                        '_ENUM' + str(config['vqvae_config']['num_embeddings']) +
                        '_CF' + str(config['vqvae_config']['compression_factor']) +
                        '_BS' + str(args.batchsize) +
                        '_seed' + str(args.seed) +
                        '_local')
    
    save_path = 'motion_vqvae/exp'

    save_dir = os.path.join(save_path, save_folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    master = {
        'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
        'config file': config_file,
        'save directory': save_dir,
        'gpus': args.model_init_num_gpus,
    }

    # Set up GPU / CPU``
    if torch.cuda.is_available() and args.model_init_num_gpus >= 0:
        assert args.model_init_num_gpus < torch.cuda.device_count()  # sanity check
        device = 'cuda:{:d}'.format(args.model_init_num_gpus)
    else:
        device = 'cpu'

    # Where to init data for training (cpu or gpu) -->  will be trained wherever args.model_init_num_gpus says
    if args.data_init_cpu_or_gpu == 'gpu':
        data_init_loc = device  # we do this so that data_init_loc will have the correct cuda:X if gpu
    else:
        data_init_loc = 'cpu'

    # call main
    main(device, config, save_dir, data_init_loc, args)