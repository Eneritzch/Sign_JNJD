import tensorflow as tf

print("Cargando modelo dinámico...")
model = tf.keras.models.load_model("static/models/cnn_bigru_dynamic.h5")
model.summary()  


base_cnn = model.layers[1]  # TimeDistributed(EfficientNetB0)
gap = model.layers[2]       # TimeDistributed(GlobalAveragePooling2D)

# Crear modelo que procese UNA imagen
img_input = tf.keras.Input(shape=(224, 224, 3))
features = base_cnn.layer(img_input)  # EfficientNetB0 sin TimeDistributed
features = gap.layer(features)        # GlobalAveragePooling2D

feature_extractor = tf.keras.Model(inputs=img_input, outputs=features)
feature_extractor.save("static/models/efficientnetb0_feature_extractor.h5")
print("Extractor de características guardado: efficientnetb0_feature_extractor.h5")