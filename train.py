import os
import json
import timeit
import argparse
from tqdm import tqdm
import tensorflow as tf
# from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers.experimental.preprocessing import Normalization

from networks.AugmentCE2P import ResNet
# from .utils.schp as schp
from utils import schp as schp
from datasets.target_generation import generate_edge_tensor
from utils.criterion import CriterionAll
from utils.callbacks import SGDRScheduler
from datasets.datasets import ATRDataset
import warnings
import numpy as np
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    # Network Structure
    parser.add_argument("--arch", type=str, default='resnet101')
    # Data Preference
    parser.add_argument("--data-dir", type=str, default='/content/Segmentation_SCHP/ATR_small_mini')
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-size", type=str, default='512,512')
    parser.add_argument("--num-classes", type=int, default=18)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    # Training Strategy
    parser.add_argument("--restore", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=7e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval-epochs", type=int, default=10)
    parser.add_argument("--imagenet-pretrain", type=str, default='./pretrain_model/resnet101-imagenet.pth')
    parser.add_argument("--log-dir", type=str, default='./log')
    parser.add_argument("--model-restore", action='store_false')
    parser.add_argument("--schp-start", type=int, default=100, help='schp start epoch')
    parser.add_argument("--cycle-epochs", type=int, default=10, help='schp cyclical epoch')
    parser.add_argument("--schp-restore", type=str, default=False)
    parser.add_argument("--lambda-s", type=float, default=1, help='segmentation loss weight')
    parser.add_argument("--lambda-e", type=float, default=1, help='edge loss weight')
    parser.add_argument("--lambda-c", type=float, default=0.1, help='segmentation-edge consistency loss weight')
    return parser.parse_args()


def refinement(label, predict):
    b, h, w = label.shape
    # print("label shape: {0}, predict shape {1}: ".format(label.shape, predict.shape))
    tmp = tf.identity(label)
    tmp = tf.reshape(tmp, (-1,))
    valid = tf.not_equal(tmp, 255)
    # print("valid shape: ", valid.shape)

    # tf.argmax(tf.nn.softmax(preds[0][0]))
    predict = tf.compat.v1.image.resize_bilinear(images = predict, size = (h,w), align_corners=True)
    predict = tf.nn.softmax(predict, axis = -1)
    # print("predict after softmax shape: ", predict.shape)
    predict = tf.argmax(predict, axis = -1)
    # print("predict after argmax shape: ", predict.shape)

    label = tf.reshape(label, (-1))
    # print("label after reshaping shape: ", label.shape)
    predict = tf.reshape(predict, (-1))
    # print("predict after reshaping shape: ", predict.shape)

    vlabel = tf.boolean_mask(label, valid)
    vpredict = tf.boolean_mask(predict, valid)

    return vlabel, vpredict

def main():
    args = get_arguments()
    print(args)

    # Setup GPU
    gpus = tf.config.list_physical_devices('GPU')
    try:
        # Currently, memory growth needs to be the same across GPUs
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print("Using ", gpus[0], ", ", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)



    normalize = Normalization(mean=[0.406, 0.456, 0.485], variance=[0.225, 0.224, 0.229])

    def transforms(image, nor=normalize):
        """
        convert image to tensor, and normalize it with mean and std
        This is rewrite from the code:

        """
        img = tf.convert_to_tensor(image)
        img = tf.image.convert_image_dtype(img, tf.float32)
        norm_img = nor(img)

        return norm_img

    start_epoch = 0
    cycle_n = 0
    # Model initialization
    if args.restore != None:
        try:
            # args.restore is path to json file checkpoint
            f = open(args.restore)
            logs = json.load(f)
            start_epoch = logs['epoch'] + 1
            if int(logs['epoch']) + 1 < 100:
                model = ResNet(num_classes=18)
                schp_model = ResNet(num_classes=18)
                model.build((None, 512, 512, 3))
                schp_model.build((None, 512, 512, 3))
                model.load_weights(logs['path_cp'])
                print('Load model checkpoint from: ', logs['path_cp'])
            else:
                cycle_n = int((logs['epoch'] - 99) / 10 + 1)
                model = ResNet(num_classes=18)
                model.build((None, 512, 512, 3))
                schp_model = ResNet(num_classes=18)
                schp_model.build((None, 512, 512, 3))
                model.load_weights(logs['path_cp'])
                print('Load model checkpoint from: ', logs['path_cp'])
                schp_epoch = 100 + (cycle_n - 1) * 10 - 1
                schp_path = os.path.join('Checkpoints', 'schp_epoch_{}'.format(schp_epoch),
                                         'schp_epoch_{}'.format(schp_epoch))
                schp_model.load_weights(schp_path)
                print('Load schp model checkpoint from: ', schp_path)
        except RuntimeError as e:
            print(e)
    else:
        model = ResNet(num_classes=18)
        schp_model = ResNet(num_classes=18)
        schp_model.build((None, 512, 512, 3))

    path_to_checkpoints_metric = '/content/Segmentation_SCHP/log'



    # Loss Function
    criterion = CriterionAll(lambda_1=args.lambda_s, lambda_2=args.lambda_e, lambda_3=args.lambda_c,
                             num_classes=args.num_classes)

    # Data Loader

    train_dataset = ATRDataset(root=args.data_dir, dataset='train', crop_size=[512,512], transform=transforms)
    train_data = train_dataset.load_data_train()
    train_data = train_data.shuffle(4096).batch(4, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    
    val_dataset = ATRDataset(root=args.data_dir, dataset='val', crop_size=[512,512], transform=transforms)
    val_data = val_dataset.load_data_train()
    val_data = val_data.shuffle(4096).batch(4, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)


    # Optimization
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=args.momentum, nesterov=False, name="SGD")

    scheduler = SGDRScheduler()
    callbacks = tf.keras.callbacks.CallbackList(
        callbacks=[scheduler], model=model
    )
    model.compile(optimizer=optimizer, loss=criterion)

    #metric
    train_metric = tf.keras.metrics.MeanIoU(num_classes=18)
    val_metric = tf.keras.metrics.MeanIoU(num_classes=18)

    # Training Loop
    losses = []
    callbacks.on_train_begin(start_epoch)
    for epoch in range(start_epoch, args.epochs):
        losses_list = []
        metric_list = []
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
                    soft_preds = tf.concat(soft_parsing, axis=0)
                    soft_edges = tf.concat(soft_edge, axis=0)
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

                #update training metric:
                vtlabel, vtpred = refinement(labels,preds[0][0])
                train_metric.update_state(vtlabel, vtpred)
                # print("LOSS: {0} -- LR on iter: {1} -- meanIOU:{2}".format(loss, model.optimizer.learning_rate,float(train_metric.result())))

            # Self Correction Cycle with Model Aggregation
            if (epoch + 1) >= args.schp_start and (epoch + 1 - args.schp_start) % args.cycle_epochs == 0:
                print('Self-correction cycle number {}'.format(cycle_n))
                schp.moving_average(schp_model, model, 1.0 / (cycle_n + 1))
                cycle_n += 1
                schp.bn_re_estimate(train_data, schp_model)
                folder_schp = os.path.join('Checkpoints', 'schp_epoch_{}'.format(epoch))
                schp_model.save_weights(os.path.join(folder_schp, 'schp_epoch_{}'.format(epoch)))
            losses_list.append(loss.numpy())
            metric_list.append(train_metric.result())
        losses.append(float(np.mean(losses_list)))
        print("LOSS of epoch {0}: {1}".format(epoch, losses[-1]))
        print("MeanIOU: ", train_metric.result())

        train_metric.reset_state()
        

        #Validation
        for XV in val_data:
            val_images, val_labels = XV
            val_preds = model(val_images, training=False)
            #update val metrics
            vvlabel, vvpred = refinement(val_labels, val_preds[0][0])
            val_metric.update_state(vvlabel, vvpred)
        val_miou = val_metric.result()
        val_metric.reset_state()
        print("Validation MeanIOU: ", float(val_miou))
        path_json = os.path.join(path_to_checkpoints_metric, 'epoch_{}.json'.format(epoch))
        logger = {}
        logger['lr'] = float(model.optimizer.learning_rate)
        logger['epoch'] = epoch
        logger['loss_train'] = float(losses[-1])
        logger['mIOU_train'] = float(metric_list[-1])
        logger['mIOU_val'] = float(val_miou)
        with open(path_json, 'w') as f:
                json.dump(logger, f)
        f.close()
        callbacks.on_epoch_end(epoch, losses)
    callbacks.on_train_end()

    end = timeit.default_timer()
    # print('Training Finished in {} seconds'.format(end - start))
if __name__=="__main__":
    main()
