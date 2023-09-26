"""
final_project.py

Source Code for "Towards Class-Shielding Data Poisoning", submission for the final project for
CSS 586 A Spring Quarter, University of Washington Bothell

Written 2023 by Thomas Pinkava
"""

import json
import time
import ssl  # For model download signing

import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

from tensorflow.keras.applications import vgg16, resnet50


# ========== DATA PIPELINE ==========

# Printing utility for showing progress bar (as we're using a manual training loop)
def print_progress_bar(val, total):
    BAR_WIDTH = 40
    w = (val / float(total)) * BAR_WIDTH
    bar_string = ''.join([' ' if x > w else '=' for x in range(BAR_WIDTH)])
    print(f"\r[{bar_string}]",end='')
    

# Data import utility: loads imagenet images from data_path, where data_path is a directory containing subdirectories
# whose names are the synset codes of images contained therein
def import_data(data_path, batch_triple_count, validation_split, victim_preprocessor,
    class_lookup_path = "ImageNet/resnet_class_lookup.json"):

    batch_size = 3 * batch_triple_count     # must be divisible by three; model requires data triplets.    

    # Load dataset from subset path
    train_ds, val_ds = image_dataset_from_directory(directory=data_path, batch_size = None, image_size=(224, 224),
        validation_split = validation_split, subset = 'both', follow_links = True, seed = 123)
    old_class_gather = tf.constant(train_ds.class_names)

    # Batch the datasets and mark them for caching and prefetching
    train_ds = train_ds.cache().prefetch(buffer_size = tf.data.AUTOTUNE).batch(batch_size, drop_remainder = True)
    val_ds = val_ds.cache().prefetch(buffer_size = tf.data.AUTOTUNE).batch(batch_size, drop_remainder = True)


    # Preprocess images and remap class labels to victim model's labels
    with open(class_lookup_path) as class_lookup_file:
        class_lookup = json.load(class_lookup_file)
    synset_to_victim_class = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(tf.constant([k for k in class_lookup.keys()]), 
        tf.constant([int(v) for v in class_lookup.values()])), default_value = 0)
 
    train_ds = train_ds.map(lambda image, label : (victim_preprocessor(image), synset_to_victim_class.lookup(tf.gather(old_class_gather, label))))
    val_ds = val_ds.map(lambda image, label : (victim_preprocessor(image), synset_to_victim_class.lookup(tf.gather(old_class_gather, label))))

    return (train_ds, val_ds)




# ========== POISONER MODEL ==========


# Constructs and returns the poisoner
def construct_poisoner_model():

    # Establish input dimensions
    batch_spec = train_ds.element_spec[0]
    tupled_shape_list = batch_spec.shape.as_list()
    tupled_shape_list.insert(1, 2)
    input_spec = tf.TensorSpec(tf.TensorShape(tupled_shape_list), dtype=batch_spec.dtype)

    # Inputs: split shield and poison images
    inputs = tf.keras.Input(type_spec = input_spec)
    input_split = tf.unstack(inputs, axis=1)
    shield_input = input_split[0]
    target_input = input_split[1]
    
    # The poison perturber: generates residual perturbations that will be added to the shield
    # to produce the poison output.
    # Step 1: Feature extraction from shield and target
    feature_extractor = resnet50.ResNet50(weights='imagenet', include_top=False)
    feature_extractor.trainable = False

    shield_feature_extractor = feature_extractor(shield_input)
    shield_feature_flattened = tf.keras.layers.GlobalAveragePooling2D()(shield_feature_extractor)

    target_feature_extractor = feature_extractor(target_input)
    target_feature_flattened = tf.keras.layers.GlobalAveragePooling2D()(target_feature_extractor)

    # Step 2: Latent-space processing
    latent_output_dimension = 16
    latent_output_units = latent_output_dimension * latent_output_dimension * 3
    shield_target_latent_concatenated = tf.keras.layers.Concatenate()([shield_feature_flattened, target_feature_flattened])

    latent_1 = tf.keras.layers.Dense(2048, activation='relu')(shield_target_latent_concatenated)
    latent_2 = tf.keras.layers.Dense(latent_output_units, activation='relu')(latent_1)

    # Step 3: Featured upsampling back to image residue
    latent_to_2D = tf.keras.layers.Reshape((latent_output_dimension, latent_output_dimension, 3))(latent_2)

    upsample_1 = tf.keras.layers.Conv2DTranspose(3, (1,1), strides=(2,2))(latent_to_2D)
    upsample_2 = tf.keras.layers.Conv2DTranspose(3, (1,1), strides=(7, 7))(upsample_1)


    # Final, residual addition to create perturbed shield.
    shield_plus_poison = tf.keras.layers.Add()([shield_input, upsample_2])
    
    return(tf.keras.Model(inputs=inputs, outputs=shield_plus_poison), tf.keras.optimizers.legacy.Adam())






# ========== TRAINING LOOP ==========

# Constructs and returns the victim model
#def construct_victim_model():
#    res_base = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
#    res_base.trainable = False
#    inputs = tf.keras.Input(type_spec = train_ds.element_spec[0])
#    b = res_base(inputs)
#    c = tf.keras.layers.GlobalAveragePooling2D()(b)
#    outputs = tf.keras.layers.Dense(1000, activation='softmax')(c)
#
#    return(tf.keras.Model(inputs=inputs, outputs=outputs), tf.keras.optimizers.legacy.Adam())

# Constructs and returns the victim model
def construct_victim_model():
    vgg_base = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    vgg_base.trainable = False
    inputs = tf.keras.Input(type_spec = train_ds.element_spec[0])
    b = vgg_base(inputs)
    c = tf.keras.layers.GlobalAveragePooling2D()(b)
    outputs = tf.keras.layers.Dense(1000, activation='softmax')(c)

    return(tf.keras.Model(inputs=inputs, outputs=outputs), tf.keras.optimizers.legacy.Adam())


# Reverse batched sparse categorical crossentropy: a horizontally flipped and offset categorical crossentropy
# Used for the target class, so that total uncertainty about the target is perfect (zero) loss, while increasing
# certainty of targethood is penalized exponentially 
@tf.function
def reverseSparseCategoricalCrossentropy(y_true, y_pred):
    label_onehots = tf.one_hot(y_true, depth = y_pred.shape[1])
    masked_probs = y_pred * label_onehots
    probs = tf.reduce_sum(masked_probs, axis=1)
    flipped_neg_logs = tf.math.log((probs * -1.0) + 1.0) * -1.0
    return (tf.reduce_sum(flipped_neg_logs) / y_pred.shape[0])
    

# Custom training step
@tf.function
def train_step(x, y):

    # Split batch into shield items, target items, and control items
    x_shield, x_target, x_control = tf.split(x, [batch_size_third, batch_size_third, batch_size_third])
    y_shield, y_target, y_control = tf.split(y, [batch_size_third, batch_size_third, batch_size_third])


    # A vexing quirk of TensorFlow's GradientTape auto-differentiator is that it can't see across
    # non-tensorflow operations, so our nontrivial generate-train-predict-loss-update architecture is 
    # non-tapeable.
    # Fortunately, since the poisoning process is entirely deterministic, we can get around this by generating
    # poisons once, training the victim, then generating the exact same poisons again; flogging them through
    # the victim model as shield items gives us the input-to-loss flow needed for the GradientTape.
    # Actually it's a vexing quirk of TensorFlow as a whole: TF isn't a Python library, it's an application
    # living in its own little universe whose scripting language is unhappily mostly interoperable and
    # interoperated with the Python Interpreter.


    # Generate poison
    shield_target_pairs = tf.stack([x_shield, x_target], axis=1) 
    poison = poisoner_model(shield_target_pairs, training=False)

    # Train victim on poison
    for poison_sub_batch, label_sub_batch in zip(tf.split(poison, sub_batches_per_batch), tf.split(y_shield, sub_batches_per_batch)):

        # Watch the victim as estimates are produced
        with tf.GradientTape() as victim_tape:
            victim_pred = victim_model(poison_sub_batch)
            victim_loss = victim_loss_fn(label_sub_batch, victim_pred)     

        # Update victim weights
        gradients = victim_tape.gradient(victim_loss, victim_model.trainable_weights)
        victim_optimizer.apply_gradients(zip(gradients, victim_model.trainable_weights))

    # RE-Generate poison
    with tf.GradientTape() as poisoner_tape:

        poison = poisoner_model(shield_target_pairs, training=True)
        victim_pred_shield = victim_model(poison)
        victim_pred_target = victim_model(x_target)
        victim_pred_control = victim_model(x_control)

        shield_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)(y_shield, victim_pred_shield)
        control_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)(y_control, victim_pred_control)
        ordinary_loss = (shield_loss + control_loss) / 2.0

        target_loss = reverseSparseCategoricalCrossentropy(y_target, victim_pred_target)

        total_loss = (ordinary_loss + target_loss * TARGET_LOSS_FACTOR) / 2.0
        
    # Update poisoner weights
    gradients = poisoner_tape.gradient(total_loss, poisoner_model.trainable_weights)
    poisoner_optimizer.apply_gradients(zip(gradients, poisoner_model.trainable_weights))


    # Update metrics
    ordinary_loss_metric.update_state(ordinary_loss)
    target_loss_metric.update_state(target_loss)
    total_loss_metric.update_state(total_loss)
    accuracy_shield_control_metric.update_state(y_shield, victim_pred_shield)
    accuracy_shield_control_metric.update_state(y_control, victim_pred_control)
    accuracy_target_metric.update_state(y_target, victim_pred_target)
    
    

# Custom validation step
@tf.function
def val_step(x, y):
    # Split batch into shield items, target items, and control items
    x_shield, x_target, x_control = tf.split(x, [batch_size_third, batch_size_third, batch_size_third])
    y_shield, y_target, y_control = tf.split(y, [batch_size_third, batch_size_third, batch_size_third])

    # Generate poison
    shield_target_pairs = tf.stack([x_shield, x_target], axis=1)
    poison = poisoner_model(shield_target_pairs)
    
    # Train victim on poison
    for poison_sub_batch, label_sub_batch in zip(tf.split(poison, sub_batches_per_batch), tf.split(y_shield, sub_batches_per_batch)):
        with tf.GradientTape() as victim_tape:
            victim_pred = victim_model(poison)
            victim_loss = victim_loss_fn(y_shield, victim_pred)     

        # Update victim weights
        gradients = victim_tape.gradient(victim_loss, victim_model.trainable_weights)
        victim_optimizer.apply_gradients(zip(gradients, victim_model.trainable_weights))

    # Victim predicts on shield, target and control classes
    victim_pred_shield = victim_model(x_shield)
    victim_pred_target = victim_model(x_target)
    victim_pred_control = victim_model(x_control)

    # Compute and report accuracies
    accuracy_shield_control_metric.update_state(y_shield, victim_pred_shield)
    accuracy_shield_control_metric.update_state(y_control, victim_pred_control)
    accuracy_target_metric.update_state(y_target, victim_pred_target)


# Reset the victim dummy's trainable weights
def reset_victim_weights():
    for layer in victim_model.layers:
        if layer.trainable and hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            weights, biases = layer.get_weights()
            layer.set_weights([layer.kernel_initializer(shape=weights.shape), layer.bias_initializer(shape=biases.shape)])


    



# ========== TRAIN/VAL DRIVER ==========

# Undowloaded models won't donwload without an ssl context (even an unverified one - peculiar!)
ssl._create_default_https_context = ssl._create_unverified_context

# Hyperparameters
TARGET_LOSS_FACTOR = 1.0

# Establish batch sizes (TODO tune)
sub_batch_size = 10
sub_batches_per_batch = 10
batch_size_third = sub_batch_size * sub_batches_per_batch


# Import dataset
train_ds, val_ds = import_data("imagenet_subset", batch_size_third, 0.2, vgg16.preprocess_input)
print(f"Running with {len(train_ds)} batches.")

# Define victim and poisoner models
victim_model, victim_optimizer = construct_victim_model()
poisoner_model, poisoner_optimizer = construct_poisoner_model()
victim_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()


# Set up training metric trackers
ordinary_loss_metric = tf.keras.metrics.Mean(name="ordinary loss")
target_loss_metric = tf.keras.metrics.Mean(name="target loss")
total_loss_metric = tf.keras.metrics.Mean(name="total loss")
accuracy_shield_control_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="shield/control accuracy")
accuracy_target_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="target accuracy")
metrics = [ordinary_loss_metric, target_loss_metric, total_loss_metric, accuracy_shield_control_metric, accuracy_target_metric]

# Set up validation metric trackers (duplicate some of the extant trackers)
validation_metrics = [accuracy_shield_control_metric, accuracy_target_metric]


# Iterate over epochs
epoch_count = 10
for epoch in range(epoch_count):
    print(f"\nEPOCH {epoch}:\nTraining:")

    for batch_index, (x, y) in enumerate(train_ds):
        
        # Execute training step
        train_step(x, y)

        # Reset the victim victim_model's weights
        reset_victim_weights()

        # Report progress at intervals
        #if batch_index % 5 == 0:
        if True:    # Too few batches to bother intervalling
            print_progress_bar(batch_index, train_ds.cardinality().numpy())
            for metric in metrics:
                print(f'  {metric.name}: {metric.result():.6}', end='')
            print('      ', end='')
            
    print('\nValidating:')


    # Reset metrics
    for metric in metrics:
        metric.reset_states()


    # Per-epoch validation
    for batch_index, (x, y) in enumerate(val_ds):
        
        # Execute validation step
        val_step(x, y)

        # Reset victim weights
        #reset_victim_weights() # TODO: not in the val step?
        
        print_progress_bar(batch_index, val_ds.cardinality().numpy())
        for metric in validation_metrics:
            print(f'  {metric.name}: {metric.result():.6}', end='')
        print('             ', end='')
    print('')


    # Re-reset metrics
    for metric in metrics:
        metric.reset_states()
    
