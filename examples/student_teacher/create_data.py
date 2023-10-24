import jax
import jax.numpy as jnp
from flax import linen as nn
import tensorflow_datasets as tfds
import tensorflow as tf

class Teacher(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(8)(x)
        return x

def create_data(rng_key, num_samples):
    teacher = Teacher()
    params = teacher.init(rng_key, jnp.ones((10,)))
    x = jax.random.normal(rng_key, (num_samples,10))
    y = teacher.apply(params, x)
    return x,y

def create_dataset(x,y):
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.shuffle(buffer_size=1024).batch(64).prefetch(1)
    return dataset

rng = jax.random.PRNGKey(0)
x,y, = create_data(rng, 100)
train_dataset = create_dataset(x,y)
x,y, = create_data(rng, 100)
test_dataset = create_dataset(x,y)
print(train_dataset)
print(test_dataset)

