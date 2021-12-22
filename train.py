import os
import json
import timeit
import argparse
from tqdm import tqdm
import tensorflow as tf

from networks.AugmentCE2P import ResNet
# from .utils.schp as schp
from utils import schp as schp
from datasets.target_generation import generate_edge_tensor
from utils.criterion import CriterionAll
from utils.callbacks import SGDRScheduler
from datasets.datasets import ATRDataset


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    # Network Structure
    parser.add_argument("--arch", type=str, default='resnet101')
    # Data Preference
    parser.add_argument("--data-dir", type=str, default='/content/Segmentation_SCHP/ATR')
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-size", type=str, default='512,512')
    parser.add_argument("--num-classes", type=int, default=18)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    # Training Strategy
    parser.add_argument("--learning-rate", type=float, default=7e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval-epochs", type=int, default=10)
    parser.add_argument("--imagenet-pretrain", type=str, default='./pretrain_model/resnet101-imagenet.pth')
    parser.add_argument("--log-dir", type=str, default='./log')
    parser.add_argument("--model-restore", type=str, default='./log/checkpoint.pth.tar')
    parser.add_argument("--schp-start", type=int, default=100, help='schp start epoch')
    parser.add_argument("--cycle-epochs", type=int, default=10, help='schp cyclical epoch')
    parser.add_argument("--schp-restore", type=str, default='./log/schp_checkpoint.pth.tar')
    parser.add_argument("--lambda-s", type=float, default=1, help='segmentation loss weight')
    parser.add_argument("--lambda-e", type=float, default=1, help='edge loss weight')
    parser.add_argument("--lambda-c", type=float, default=0.1, help='segmentation-edge consistency loss weight')
    return parser.parse_args()




def main():
    args = get_arguments()
    print(args)

    start_epoch = 0
    cycle_n = 0

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, 'args.json'), 'w') as opt_file:
        json.dump(vars(args), opt_file)


    gpus = [int(i) for i in args.gpu.split(',')]
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    input_size = list(map(int, args.input_size.split(',')))


    def transforms(image):
        """
        convert image to tensor, and normalize it with mean and std
        This is rewrite from the code:

        """
        img = tf.convert_to_tensor(image)
        norm_img = tf.image.per_image_standardization(img)

        return norm_img

    # Model initialization
    restore_from = args.model_restore
    if os.path.exists(restore_from):
        print('Resume training from {}'.format(restore_from))
        model = tf.keras.models.load_model(os.path.join('restore_from','epoch.h5'))  # Callbacks
        # start_epoch = checkpoint['epoch']
        schp_model = tf.keras.models.load_model(os.path.join('restore_from','schp_epoch.h5'))

    else:
        model = ResNet(num_classes=18)
        schp_model = ResNet(num_classes=18)



    # Loss Function
    criterion = CriterionAll(lambda_1=args.lambda_s, lambda_2=args.lambda_e, lambda_3=args.lambda_c,
                             num_classes=args.num_classes)

    # Data Loader

    train_dataset = ATRDataset(root=args.data_dir, dataset='val', crop_size=[512,512], transform=transforms)
    train_data = train_dataset.load_data_train()
    train_data = train_data.shuffle(4096).batch(4, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)


    # Optimization
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=args.momentum, nesterov=False, name="SGD")

    scheduler = SGDRScheduler(base_lr=args.learning_rate)
    callbacks = tf.keras.callbacks.CallbackList(
        callbacks=[scheduler], add_history=True, model=model
    )

    # Training Loop
    losses = []
    callbacks.on_train_begin(optimizer)
    for epoch in range(start_epoch, args.epochs):
        with tqdm(train_data) as pbar:
            pbar.set_description(f"[Epoch {epoch}]")
            for step, X in enumerate(pbar):

                images, labels = X
                edges = generate_edge_tensor(labels)

                # Online Self Correction Cycle with Label Refinement
                if cycle_n >= 1:
                    soft_preds = [schp_model(images,training=True)]
                    soft_parsing = []
                    soft_edge = []
                    for soft_pred in soft_preds:
                        soft_parsing.append(soft_pred[0][-1])
                        soft_edge.append(soft_pred[1][-1])
                    soft_preds = tf.concat(soft_parsing, dim=0)
                    soft_edges = tf.concat(soft_edge, dim=0)
                else:
                    soft_preds = None
                    soft_edges = None

                with tf.GradientTape() as d_tape:
                    preds = model(images, training = True)
                    loss = criterion([labels, edges, soft_preds, soft_edges],[preds, cycle_n])
                # Calculate gradient
                gradients = d_tape.gradient(loss, model.trainable_variables)
                # gradients = d_tape.gradient(loss, model.t)
                # Update params
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                print("LOSS RETURN: ", loss)

                # Self Correction Cycle with Model Aggregation
                if (epoch + 1) >= args.schp_start and (epoch + 1 - args.schp_start) % args.cycle_epochs == 0:
                    print('Self-correction cycle number {}'.format(cycle_n))
                    schp.moving_average(schp_model, model, 1.0 / (cycle_n + 1))
                    cycle_n += 1
                    schp.bn_re_estimate(train_data, schp_model) 
                    schp.save('schp_epoch_{}_cycle_n{}.h5'.format(epoch, cycle_n))
                losses.append(loss)
        callbacks.on_epoch_end(epoch, model, losses)
    callbacks.on_train_end()

    end = timeit.default_timer()
    # print('Training Finished in {} seconds'.format(end - start))
if __name__=="__main__":
    main()