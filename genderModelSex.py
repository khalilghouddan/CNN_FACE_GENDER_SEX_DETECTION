
# Model	Keras functional API model object
from tensorflow.keras import Model


# Input	Defines the input shape of your model (e.g., images)
# Conv2D	2D convolution layer for feature extraction
# MaxPooling2D	Reduces spatial dimensions (downsampling)
# Flatten	Converts 2D feature maps into 1D vector for Dense layers
# Dense	Fully connected layer (for classification / regression)
# Dropout	Regularization to prevent overfitting
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Activation
from tensorflow.keras.regularizers import l2



# Function to create the gender classification model
def create_gender_model(input_shape):
    
    #define the input layer
    inputs = Input(shape=input_shape)

    #firstblock 
    #this layer detect edges and simple patterns
    conv1 = Conv2D(64,(5,5),padding = 'same',strides=(1,1),kernel_regularizer=l2(0.01))(inputs)
    #max(0, x)
    #activation avec NeLU
    conv1 = Activation('relu')(conv1)
    #MaxPooling2D (2×2) → halves spatial dimensions → 50×50×64
    pool1=MaxPooling2D(pool_size=(2,2))(conv1)
    #Slightly slows down training because neurons are randomly ignored.
    #Should be used carefully — too high dropout can underfit the model.
    pool1=Dropout(0.1)(pool1)
    
    #Les couches suivantes détectent des motifs plus complexes à partir des combinaisons des premières caractéristiques.
    conv2 = Conv2D(64, (3,3), padding='same', strides=(1,1), kernel_regularizer=l2(0.001))(pool1)
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(0.1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(128, (3,3), padding='same', strides=(1,1), kernel_regularizer=l2(0.001))(pool2)
    conv3 = Activation('relu')(conv3)
    conv3 = Dropout(0.1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
    
    #Les dernières couches combinent toutes les caractéristiques pour faire la classification.
    conv4 = Conv2D(256, (3,3), padding='same', strides=(1,1), kernel_regularizer=l2(0.001))(pool3)
    conv4 = Activation('relu')(conv4)
    conv4 = Dropout(0.1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

    #Needed because fully connected layers expect 1D input
    #output just before flatten is 12×12×256
    flatten=Flatten()(pool4)
    
    dense_1 = Dense(128, activation='relu')(flatten)
    
    #Dropout(0.2) → drops 20% of neurons to reduce overfitting
    drop_1 = Dropout(0.2)(dense_1)
    
    #2 neurons for binary classification (male/female
    output = Dense(2, activation="softmax")(drop_1)
    
    model = Model(inputs=inputs, outputs=output)
    return model

































