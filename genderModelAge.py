from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Activation
from tensorflow.keras.regularizers import l2


def create_age_model(input_shape=(200, 200, 1), num_classes=7):

    inputs = Input(shape=input_shape)

    # ----- BLOCK 1 -----
    x = Conv2D(64, (5,5), padding='same', strides=(1,1), kernel_regularizer=l2(0.01))(inputs)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.1)(x)

    # ----- BLOCK 2 -----
    x = Conv2D(64, (3,3), padding='same', strides=(1,1), kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    # ----- BLOCK 3 -----
    x = Conv2D(128, (3,3), padding='same', strides=(1,1), kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    # ----- BLOCK 4 -----
    x = Conv2D(256, (3,3), padding='same', strides=(1,1), kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    # ----- FULLY CONNECTED LAYERS -----
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)

    # ----- OUTPUT -----
    output = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=output)
    return model
