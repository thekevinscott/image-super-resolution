import yaml
import numpy as np
from time import time
from tqdm import tqdm
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras import backend as K
from ISR.utils.datahandler import DataHandler
from ISR.utils.train_helper import TrainerHelper
from ISR.utils.metrics import PSNR
from ISR.utils.logger import get_logger


class Trainer:
    """Class object to setup and carry the training.

    Takes as input a generator that produces SR images.
    Conditionally, also a discriminator network and a feature extractor
        to build the components of the perceptual loss.
    Compiles the model(s) and trains in a GANS fashion if a discriminator is provided, otherwise
    carries a regular ISR training.

    Args:
        generator: Keras model, the super-scaling, or generator, network.
        discriminator: Keras model, the discriminator network for the adversarial
            component of the perceptual loss.
        feature_extractor: Keras model, feature extractor network for the deep features
            component of perceptual loss function.
        lr_train_dir: path to the directory containing the Low-Res images for training.
        hr_train_dir: path to the directory containing the High-Res images for training.
        lr_valid_dir: path to the directory containing the Low-Res images for validation.
        hr_valid_dir: path to the directory containing the High-Res images for validation.
        learning_rate: float.
        loss_weights: dictionary, use to weigh the components of the loss function.
            Contains 'MSE' for the MSE loss component, and can contain 'discriminator' and 'feat_extr'
            for the discriminator and deep features components respectively.
        logs_dir: path to the directory where the tensorboard logs are saved.
        weights_dir: path to the directory where the weights are saved.
        dataname: string, used to identify what dataset is used for the training session.
        weights_generator: path to the pre-trained generator's weights, for transfer learning.
        weights_discriminator: path to the pre-trained discriminator's weights, for transfer learning.
        n_validation:integer, number of validation samples used at training from the validation set.
        T: 0 < float <1, determines the 'flatness' threshold level for the training patches.
            See the TrainerHelper class for more details.
        lr_decay_frequency: integer, every how many epochs the learning rate is reduced.
        lr_decay_factor: 0 < float <1, learning rate reduction multiplicative factor.

    Methods:
        train: combines the networks and triggers training with the specified settings.

    """

    def __init__(
        self,
        generator,
        discriminator,
        feature_extractor,
        lr_train_dir,
        hr_train_dir,
        lr_valid_dir,
        hr_valid_dir,
        learning_rate=0.0004,
        loss_weights={'MSE': 1.0},
        logs_dir='logs',
        weights_dir='weights',
        dataname=None,
        weights_generator=None,
        weights_discriminator=None,
        n_validation=None,
        T=0.01,
        lr_decay_frequency=100,
        lr_decay_factor=0.5,
        fallback_save_every_n_epochs=2,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=0.00001,
    ):
        if discriminator:
            assert generator.patch_size * generator.scale == discriminator.patch_size
        if feature_extractor:
            assert generator.patch_size * generator.scale == feature_extractor.patch_size
        self.generator = generator
        self.discriminator = discriminator
        self.feature_extractor = feature_extractor
        self.scale = generator.scale
        self.lr_patch_size = generator.patch_size
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights
        self.weights_generator = weights_generator
        self.weights_discriminator = weights_discriminator
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_frequency = lr_decay_frequency
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.dataname = dataname
        self.T = T
        self.n_validation = n_validation

        self.helper = TrainerHelper(
            generator=self.generator,
            weights_dir=weights_dir,
            logs_dir=logs_dir,
            lr_train_dir=lr_train_dir,
            feature_extractor=self.feature_extractor,
            discriminator=self.discriminator,
            dataname=dataname,
            weights_generator=self.weights_generator,
            weights_discriminator=self.weights_discriminator,
            fallback_save_every_n_epochs=fallback_save_every_n_epochs,
        )

        self.model = self._combine_networks()

        self.train_dh = DataHandler(
            lr_dir=lr_train_dir,
            hr_dir=hr_train_dir,
            patch_size=self.lr_patch_size,
            scale=self.scale,
            n_validation_samples=None,
            T=T,
        )
        self.valid_dh = DataHandler(
            lr_dir=lr_valid_dir,
            hr_dir=hr_valid_dir,
            patch_size=self.lr_patch_size,
            scale=self.scale,
            n_validation_samples=n_validation,
            T=0.0,
        )
        self.logger = get_logger(__name__)

        self.settings = self.get_training_config()

    def _combine_networks(self):
        """
        Constructs the combined model which contains the generator network,
        as well as discriminator and geature extractor, if any are defined.
        """

        lr = Input(shape=(self.lr_patch_size,) * 2 + (3,))
        sr = self.generator.model(lr)
        outputs = [sr]
        losses = ['mse']
        loss_weights = [self.loss_weights['MSE']]
        if self.discriminator:
            self.discriminator.model.trainable = False
            validity = self.discriminator.model(sr)
            outputs.append(validity)
            losses.append('binary_crossentropy')
            loss_weights.append(self.loss_weights['discriminator'])
        if self.feature_extractor:
            self.feature_extractor.model.trainable = False
            sr_feats = self.feature_extractor.model(sr)
            outputs.extend([*sr_feats])
            losses.extend(['mse'] * len(sr_feats))
            loss_weights.extend([self.loss_weights['feat_extr'] / len(sr_feats)] * len(sr_feats))
        combined = Model(inputs=lr, outputs=outputs)
        # https://stackoverflow.com/questions/42327543/adam-optimizer-goes-haywire-after-200k-batches-training-loss-grows
        optimizer = Adam(
            beta_1=self.beta_1, beta_2=self.beta_2, lr=self.learning_rate, epsilon=self.epsilon
        )
        combined.compile(
            loss=losses, loss_weights=loss_weights, optimizer=optimizer, metrics={'generator': PSNR}
        )
        return combined

    def _lr_scheduler(self, epoch):
        """ Scheduler for the learning rate updates. """

        n_decays = epoch // self.lr_decay_frequency
        # no lr below minimum control 10e-6
        return max(1e-6, self.learning_rate * (self.lr_decay_factor ** n_decays))

    def _load_weights(self):
        """
        Loads the pretrained weights from the given path, if any is provided.
        If a discriminator is defined, does the same.
        """

        if self.weights_generator:
            self.model.get_layer('generator').load_weights(self.weights_generator)

        if self.discriminator:
            if self.weights_discriminator:
                self.model.get_layer('discriminator').load_weights(self.weights_discriminator)
                self.discriminator.model.load_weights(self.weights_discriminator)

    def _format_losses(self, prefix, losses, model_metrics):
        """ Creates a dictionary for tensorboard tracking. """

        return dict(zip([prefix + m for m in model_metrics], losses))

    def get_training_config(self):
        """ Summarizes training setting. """

        settings = {}
        settings['generator'] = {}
        settings['generator']['name'] = self.generator.name
        settings['generator']['parameters'] = self.generator.params

        if self.discriminator:
            settings['discriminator'] = {}
            settings['discriminator']['name'] = self.discriminator.name
        else:
            settings['discriminator'] = None

        if self.discriminator:
            settings['feature_extractor'] = {}
            settings['feature_extractor']['name'] = self.discriminator.name
        else:
            settings['feature_extractor'] = None

        settings['training_parameters'] = {}
        settings['training_parameters']['scale'] = self.scale
        settings['training_parameters']['lr_patch_size'] = self.lr_patch_size
        settings['training_parameters']['learning_rate'] = self.learning_rate
        settings['training_parameters']['loss_weights'] = self.loss_weights
        settings['training_parameters']['weights_discriminator'] = self.weights_discriminator
        settings['training_parameters']['weights_generator'] = self.weights_generator
        settings['training_parameters']['lr_decay_factor'] = self.lr_decay_factor
        settings['training_parameters']['lr_decay_frequency'] = self.lr_decay_frequency
        settings['training_parameters']['beta_1'] = self.beta_1
        settings['training_parameters']['beta_2'] = self.beta_2
        settings['training_parameters']['epsilon'] = self.epsilon
        settings['training_parameters']['dataname'] = self.dataname
        settings['training_parameters']['T'] = self.T
        settings['training_parameters']['n_validation'] = self.n_validation

        return settings

    def train(self, epochs, steps_per_epoch, batch_size, monitored_metrics):
        """
        Carries on the training for the given number of epochs.
        Sends the losses to Tensorboard.

        Args:
            epochs: how many epochs to train for.
            steps_per_epoch: how many batches epoch.
            batch_size: amount of images per batch.
            monitored_metrics: dictionary, the keys are the metrics that are monitored for the weights
                saving logic. The values are the mode that trigger the weights saving ('mix' vs 'max').
        """

        self.settings['training_parameters']['steps_per_epoch'] = steps_per_epoch
        self.settings['training_parameters']['batch_size'] = batch_size
        starting_epoch = self.helper.initialize_training(
            self
        )  # load_weights, creates folders, creates basename

        self.tensorboard = TensorBoard(log_dir=self.helper.callback_paths['logs'])
        self.tensorboard.set_model(self.model)

        # validation data
        validation_set = self.valid_dh.get_validation_set(batch_size)
        y_validation = [validation_set['hr']]
        if self.discriminator:
            discr_out_shape = list(self.discriminator.model.outputs[0].shape)[1:4]
            valid = np.ones([batch_size] + discr_out_shape)
            fake = np.zeros([batch_size] + discr_out_shape)
            validation_valid = np.ones([len(validation_set['hr'])] + discr_out_shape)
            y_validation.append(validation_valid)
        if self.feature_extractor:
            validation_feats = self.feature_extractor.model.predict(validation_set['hr'])
            y_validation.extend([*validation_feats])

        for epoch in range(starting_epoch, epochs):
            self.logger.info('Epoch {e}/{tot_eps}'.format(e=epoch, tot_eps=epochs))
            K.set_value(self.model.optimizer.lr, self._lr_scheduler(epoch=epoch))
            self.logger.info('Current learning rate: {}'.format(K.eval(self.model.optimizer.lr)))
            epoch_start = time()
            for step in tqdm(range(steps_per_epoch)):
                batch = self.train_dh.get_batch(batch_size)
                y_train = [batch['hr']]
                training_losses = {}

                ## Discriminator training
                if self.discriminator:
                    sr = self.generator.model.predict(batch['lr'])
                    d_loss_real = self.discriminator.model.train_on_batch(batch['hr'], valid)
                    d_loss_fake = self.discriminator.model.train_on_batch(sr, fake)
                    d_loss_fake = self._format_losses(
                        'train_d_fake_', d_loss_fake, self.discriminator.model.metrics_names
                    )
                    d_loss_real = self._format_losses(
                        'train_d_real_', d_loss_real, self.discriminator.model.metrics_names
                    )
                    training_losses.update(d_loss_real)
                    training_losses.update(d_loss_fake)
                    y_train.append(valid)

                ## Generator training
                if self.feature_extractor:
                    hr_feats = self.feature_extractor.model.predict(batch['hr'])
                    y_train.extend([*hr_feats])

                model_losses = self.model.train_on_batch(batch['lr'], y_train)
                model_losses = self._format_losses('train_', model_losses, self.model.metrics_names)
                training_losses.update(model_losses)

                self.tensorboard.on_epoch_end(epoch * steps_per_epoch + step, training_losses)
                self.logger.debug('Losses at step {s}:\n {l}'.format(s=step, l=training_losses))

            elapsed_time = time() - epoch_start
            self.logger.info('Epoch {} took {:10.1f}s'.format(epoch, elapsed_time))

            validation_losses = self.model.evaluate(
                validation_set['lr'], y_validation, batch_size=batch_size
            )
            validation_losses = self._format_losses(
                'val_', validation_losses, self.model.metrics_names
            )

            if epoch == starting_epoch:
                remove_metrics = []
                for metric in monitored_metrics:
                    if (metric not in training_losses) and (metric not in validation_losses):
                        msg = ' '.join([metric, 'is NOT among the model metrics, removing it.'])
                        self.logger.error(msg)
                        remove_metrics.append(metric)
                for metric in remove_metrics:
                    _ = monitored_metrics.pop(metric)

            # should average train metrics
            end_losses = {}
            end_losses.update(validation_losses)
            end_losses.update(training_losses)

            self.helper.on_epoch_end(
                epoch=epoch,
                losses=end_losses,
                generator=self.model.get_layer('generator'),
                discriminator=self.discriminator,
                metrics=monitored_metrics,
            )
            self.tensorboard.on_epoch_end(epoch, validation_losses)
        self.tensorboard.on_train_end(None)
