# VGG 1D-Convolution Architecture in Keras - For both Classification and Regression Problems
"""Reference: [Very Deep Convolutional Networks for Large-Scale Image Recognition] (https://arxiv.org/abs/1409.1556)"""

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.layers import Conv1D, MaxPooling1D


def Conv_1D_Block(inputs, model_width, kernel):
    # 1D Convolutional Block with BatchNormalization
    conv = Conv1D(model_width, kernel, padding='same')(inputs)
    batch_norm = BatchNormalization()(conv)
    activate = Activation('relu')(batch_norm)

    return activate


class VGG:
    def __init__(self, length, num_channel, num_filters, problem_type='Regression', output_nums=1, dropout_rate=False):
        self.length = length
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.dropout_rate = dropout_rate

    def VGG11(self):
        inputs = Input((self.length, self.num_channel))  # The input tensor
        # Block 1
        x = Conv_1D_Block(inputs, self.num_filters * (2 ** 0), 3)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 2
        x = Conv_1D_Block(x, self.num_filters * (2 ** 1), 3)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 3
        x = Conv_1D_Block(x, self.num_filters * (2 ** 2), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 2), 3)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 4
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 5
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Fully Connected (MLP) block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)
        if self.dropout_rate:
            x = Dropout(self.dropout_rate, name='Dropout')(x)
        outputs = Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = Dense(self.output_nums, activation='softmax')(x)

        # Create model.
        model = Model(inputs=inputs, outputs=outputs)

        return model

    def VGG16(self):
        inputs = Input((self.length, self.num_channel))  # The input tensor
        # Block 1
        x = Conv_1D_Block(inputs, self.num_filters * (2 ** 0), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 0), 3)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 2
        x = Conv_1D_Block(x, self.num_filters * (2 ** 1), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 1), 3)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 3
        x = Conv_1D_Block(x, self.num_filters * (2 ** 2), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 2), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 2), 3)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 4
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 5
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Fully Connected (MLP) block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)
        if self.dropout_rate:
            x = Dropout(self.dropout_rate, name='Dropout')(x)
        outputs = Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = Dense(self.output_nums, activation='softmax')(x)

        # Create model.
        model = Model(inputs=inputs, outputs=outputs)

        return model

    def VGG16_v2(self):
        inputs = Input((self.length, self.num_channel))  # The input tensor
        # Block 1
        x = Conv_1D_Block(inputs, self.num_filters * (2 ** 0), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 0), 3)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 2
        x = Conv_1D_Block(x, self.num_filters * (2 ** 1), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 1), 3)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 3
        x = Conv_1D_Block(x, self.num_filters * (2 ** 2), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 2), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 2), 1)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 4
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 1)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 5
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 1)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Fully Connected (MLP) block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)
        if self.dropout_rate:
            x = Dropout(self.dropout_rate, name='Dropout')(x)
        outputs = Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = Dense(self.output_nums, activation='softmax')(x)

        # Create model.
        model = Model(inputs=inputs, outputs=outputs)

        return model

    def VGG19(self):
        inputs = Input(shape = (self.length, self.num_channel))  # The input tensor
        # Block 1
        x = Conv_1D_Block(inputs, self.num_filters * (2 ** 0), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 0), 3)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 2
        x = Conv_1D_Block(x, self.num_filters * (2 ** 1), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 1), 3)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 3
        x = Conv_1D_Block(x, self.num_filters * (2 ** 2), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 2), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 2), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 2), 3)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 4
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 5
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        if x.shape[1] <= 2:
            x = MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Fully Connected (MLP) block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)
        if self.dropout_rate:
            x = Dropout(self.dropout_rate, name='Dropout')(x)
        outputs = Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = Dense(self.output_nums, activation='softmax')(x)

        # Create model.
        model = Model(inputs=inputs, outputs=outputs)

        return model
