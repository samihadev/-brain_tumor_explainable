import tensorflow as tf
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def build_model(model_name='resnet50', input_shape=(224, 224, 3)):
    """
    بناء نموذج ResNet50 أو DenseNet121 مع طبقة تصنيف ثنائية.
    """
    if model_name.lower() == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name.lower() == 'densenet121':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("الرجاء اختيار 'resnet50' أو 'densenet121'")

    # تجميد الطبقات الأساسية في البداية
    for layer in base_model.layers:
        layer.trainable = False

    # إضافة طبقات التصنيف المخصصة
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model