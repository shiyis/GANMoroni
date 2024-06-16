## GANMoroni or The Great Angel Moroni project breakdown

According to the Prophet Joseph Smith from the LDS (or the Church of Jesus Christ of Later Day Saints), angel moroni was the guardian of the golden plates buried near his home in western New York, 
which Latter Day Saints believe were the source of the Book of Mormon. An important figure in the theology of the Latter Day Saint movement, Moroni is featured prominently in its architecture and art.

Emblematic of the great creative power of JC our savior and heavenly father, I created this project - a play-on-words - that utilizes the generative adversarial network to forge an image of prophetic faces that resemble the almighty power and spiritual embodiment. 
So please bear with me here, as I walk you through the generalities of the deep learning model, which is quite uncanny that it does what the great angel Moroni represents, which is the symbol of art and architecture. 

According to ChatGPT, Generative Adversarial Networks (GANs) can indeed represent art and architecture in abstract and innovative ways. GANs, a class of machine learning frameworks designed by Ian Goodfellow and his colleagues in 2014, consist of two neural networks: the generator and the discriminator, which work in tandem to create data that mimics a given dataset. Here’s how GANs can be applied to art and architecture:

### 1. **Creating Abstract Art**
- **Process**: A GAN is trained on a large dataset of abstract artworks. The generator tries to produce images that resemble the abstract art in the dataset, while the discriminator tries to distinguish between real and generated images.
- **Output**: The result is new abstract art pieces that have never been seen before, often combining styles and elements in unique ways.
- **Example**: Projects like DeepArt and AI Painter use GANs to generate art that can range from abstract to impressionistic, mimicking the styles of famous artists or creating entirely new styles.

### 2. **Architectural Design**
- **Process**: GANs can be trained on architectural blueprints, floor plans, and images of buildings. The generator creates new designs, while the discriminator evaluates them for realism.
- **Output**: This can lead to innovative building designs that incorporate various architectural styles or introduce new concepts that might not have been conceived by human architects alone.
- **Example**: GANs have been used to generate new floor plans and building facades, providing architects with creative inspiration and novel design options.

### 3. **Style Transfer**
- **Process**: GANs, particularly those designed for style transfer like CycleGAN, can apply the stylistic elements of one set of images (e.g., paintings) to another set (e.g., photographs of buildings).
- **Output**: This allows for the transformation of architectural photos into stylized images that look like they were painted by artists in a particular style, or vice versa.
- **Example**: Transforming photos of modern buildings to appear as if they were designed in a Gothic or Baroque style.

### 4. **Cultural Fusion**
- **Process**: By training GANs on datasets from multiple cultures, GANs can blend architectural elements from different cultural styles.
- **Output**: The creation of new architectural designs that fuse elements from various traditions, resulting in unique and innovative structures.
- **Example**: Mixing elements of Japanese and Scandinavian architecture to create minimalist yet traditionally inspired designs.

### 5. **Concept Art and Visualization**
- **Process**: GANs can generate concept art for films, video games, or architectural visualization by learning from existing concept artworks.
- **Output**: This helps artists and designers quickly prototype ideas and explore a wide range of visual styles and concepts.
- **Example**: Generating various futuristic cityscapes or fantasy landscapes for use in game design or movies.

### Code Example: Basic GAN for Image Generation
Here’s a simplified example of how a GAN can be set up to generate images, which can be adapted for art and architecture:

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU
from tensorflow.keras.models import Sequential
import numpy as np

# Generator Model
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(28 * 28 * 1, activation='tanh'))
    model.add(Reshape((28, 28, 1)))
    return model

# Discriminator Model
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Compile the GAN
def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Instantiate and compile models
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Training loop (simplified)
def train_gan(gan, generator, discriminator, epochs=10000, batch_size=64):
    for epoch in range(epochs):
        # Generate random noise as input for the generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        generated_images = generator.predict(noise)
        
        # Create labels
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        # Train the discriminator
        discriminator.trainable = True
        discriminator.train_on_batch(real_images, real_labels)
        discriminator.train_on_batch(generated_images, fake_labels)
        
        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        discriminator.trainable = False
        gan.train_on_batch(noise, real_labels)

# Example usage
# Assuming `real_images` is a batch of real images
train_gan(gan, generator, discriminator)
```

### Summary
GANs have significant potential in the realms of art and architecture, offering new ways to create, innovate, and visualize. By understanding and harnessing the power of GANs, artists and architects can push the boundaries of their fields, blending tradition with cutting-edge technology to explore new creative frontiers.
