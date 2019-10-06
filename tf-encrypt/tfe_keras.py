import tf_encrypted as tfe
from tf_encrypted.keras.losses import BinaryCrossentropy
from tf_encrypted.keras.optimizers import SGD

from common import DataOwner

num_features = 10
batch_size = 100
steps_per_epoch = (training_set_size // batch_size)
epochs = 20

# Provide encrypted training data
data_owner = DataOwner('data-owner', batch_size)
x_train, y_train = data_owner.provide_private_training_data()

# Define model
model = tfe.keras.Sequential()
model.add(tfe.keras.layers.Dense(1, batch_input_shape=[batch_size, num_features]))
model.add(tfe.keras.layers.Activation('sigmoid'))

# Specify optimizer and loss
model.compile(optimizer=SGD(lr=0.01),
              loss=BinaryCrossentropy())

# Start training
model.fit(x_train,
          y_train,
          epochs=epochs,
          steps_per_epoch=steps_per_epoch)