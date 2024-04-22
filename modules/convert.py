import tensorflow as tf


new_model = tf.keras.models.load_model('models/model_1710271526.733847.h5')

# Check its architecture
new_model.summary()

#Save the model into h5 format
new_model.save('models/model_1710271526.733847.keras')
