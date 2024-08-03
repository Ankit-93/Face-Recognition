#import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

def build_mobilenetv2(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze layers in the base model
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

# Example usage:
# num_classes = 1000  # Change this to the number of classes in your task
# model = build_mobilenetv2(num_classes)
