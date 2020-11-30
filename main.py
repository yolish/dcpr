"""
Entry point for deep camera pose regression
"""
import argparse
import torch
import numpy as np
import json
from torchvision import transforms
import logging
from .util import utils
import time
from .datasets.CameraPoseDataset import CameraPoseDataset
from .models.pose_losses import CameraPoseLoss
from .models.pose_regressors import get_model


if __name__ == "__main__":
    # parse the arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model_name",
                            help="name of model to create (e.g. posenet, transposenet")
    arg_parser.add_argument("mode", help="train or eval'")
    arg_parser.add_argument("backbone_path", help="path to backbone .pth - e.g. efficientnet, resnet50")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("labels_file", help="path to a file mapping images to their poses")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained model (should match the model indicated in model_name")

    args = arg_parser.parse_args()
    utils.init_logger()

    checkpoint_prefix = ""
    logging.info("Start {} with {}".format(args.model_name, args.mode))

    use_cuda = torch.cuda.is_available()
    # Set the seeds and the device
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = 'cuda:0'
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Read config
    with open('config.json', "r") as read_file:
        config = json.load(read_file)
    model_params = config[args.model_name]
    general_params = config['general']
    config = {**model_params, **general_params}
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the model
    model = get_model(args.model_name, args.backbone_path, config).to(device)
    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.backbone_path, map_location=device_id))

    if args.mode == 'train':
        # Set to train mode
        model.train()

        # Freeze parts of the network if indicated
        freeze = config.get("freeze")
        freeze_exclude_phrase = config.get("freeze_exclude_phrase")
        if freeze:
            for name, parameter in model.named_parameters():
                if freeze_exclude_phrase not in name:
                    parameter.requires_grad_(False)

        # Set the loss
        loss = CameraPoseLoss(config)

        # Set the optimizer and scheduler
        params = list(model.parameres()) + list(loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        # Set the dataset and data loader
        data_transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
        ])
        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, data_transform)
        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")

        # Train
        checkpoint_prefix = utils.get_stamp_from_log()
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        for epoch in range(n_epochs):

            # Resetting temporal loss used for logging
            running_loss = 0.0
            n_samples = 0

            for batch_idx, minibatch in enumerate(dataloader):
                img = minibatch.get('img').to(device)
                gt_pose = minibatch.get('pose').to(device).to(dtype=torch.float32)
                batch_size = img.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                if freeze: # For TransPoseNet
                    model.eval()
                    with torch.no_grad():
                        global_desc_t, global_desc_rot = model.forward_encoder(img)
                    model.train()

                # Zero the gradients
                optim.zero_grad()

                # Forward pass to estimate the pose
                if freeze:
                    est_pose = model.forward_heads(global_desc_t, global_desc_rot)
                else:
                    est_pose = model(img)

                # Pose loss
                criterion = loss(est_pose, gt_pose)

                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    posit_err, orient_err = utils.pose_err(est_pose.detach(), gt_pose.detach())
                    logging.info("[{}/{}] running camera pose loss: {}, camera pose error: {}[m], {}[deg]".format(epoch,
                                                                                                batch_idx, running_loss/n_samples),
                                                                                                posit_err.mean().item(),
                                                                                               orient_err.mean().item())
            # Save checkpoint
            if epoch % n_freq_checkpoint == 0 and epoch != 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format())

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        # Plot the loss function
        loss_fig_path = checkpoint_prefix + "_loss_fig.png"



    else: # Test
        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        transform = data_transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
        ])
        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        stats = np.zeros((len(dataloader.dataset), 3))
        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                img = minibatch.get('img').to(device)
                gt_pose = minibatch.get('pose').to(device).to(dtype=torch.float32)

                # Forward pass to predict the pose
                tic = time.time()
                est_pose = model(img)
                toc = time.time() - tic

                # Evaluate error
                posit_err, orient_err = utils.pose_err(est_pose, gt_pose)

                # Collect statistics
                stats[i, 0] = posit_err.item()
                stats[i, 1] = orient_err.item()
                stats[i, 2] = (toc - tic)*1000

                # Record
                logging.info("Pose error: {}[m], {}[deg], inferrred in {}[ms]".format(
                    stats[i, 0],  stats[i, 1],  stats[i, 2]))

        # Record statistics
        logging.info("Median pose error: {}[m], {}[deg]".format(np.median(stats[0], stats[1])))
        logging.info("Mean inference time:{}[ms".format(np.median(stats[2])))





