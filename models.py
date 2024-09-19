
#from tensorflow_addons.losses import SigmoidFocalCrossEntropy
#depreciated, use keras.FocalLoss instead
from keras_cv.losses import FocalLoss

import tensorflow as tf

# Define the metrics.
train_loss = tf.keras.metrics.Mean(name="train_loss")
validation_loss = tf.keras.metrics.Mean(name="validation_loss")

train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
validation_accuracy = tf.keras.metrics.CategoricalAccuracy(name="validation_accuracy")

class FedKDModel:

    def __init__(self, trainable=False, arch='mobilenet', dense_layers=[128, 64], dropouts=[0.5, 0.5], name='model', compiled=False, lr=0.001, num_classes=2):
        """
        Initialize the object.
        """

        self.name = name
        self.trainable = trainable
        self.arch = arch
        self.dense_layers = dense_layers
        self.dropouts = dropouts
        self.compiled = compiled
        self.lr = lr
        self.num_classes = num_classes

    def get_model(self, input_shape=(48, 48, 3)):

        base_model = None

        # Load the base model without the top layers.
        if self.arch == 'mobilenet':
            base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
        elif self.arch == 'vgg16':
            base_model = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
        elif self.arch == 'mobilenetv1':
            base_model = tf.keras.applications.MobileNet(input_shape=input_shape, include_top=False, weights='imagenet')
        elif self.arch == 'mobilenetv3large':
            base_model = tf.keras.applications.MobileNetV3Large(input_shape=input_shape, include_top=False, weights='imagenet')
        elif self.arch == 'mobilenetv3small':
            base_model = tf.keras.applications.MobileNetV3Small(input_shape=input_shape, include_top=False, weights='imagenet')
        else:
            pass

        # Set the model trainablity attribute.
        base_model.trainable = self.trainable

        # Create the input layer.
        inputs = tf.keras.Input(shape=input_shape)

        # Initialize the base model with the input layer.
        x = base_model(inputs, training=self.trainable)

        # Add the Dense and Dropout layers to the base model.
        for i in range(len(self.dense_layers)):
            x = tf.keras.layers.Dense(self.dense_layers[i], activation='relu')(x)

            if i < len(self.dropouts):
                x = tf.keras.layers.Dropout(self.dropouts[i])(x)

        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs, outputs, name=self.name)

        if self.compiled:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                #loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=False),
                loss=FocalLoss(from_logits=False),
                metrics=['accuracy']
            )

        return model

class  DistillerModel(tf.keras.Model):

    def __init__(self, trained_teacher, student, temperature=5., alpha=0.5, beta=0.25):
        """
        Initialize the object.
        """

        super(DistillerModel, self).__init__()
        self.trained_teacher = trained_teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def train_step(self, data):
        """
        This method defines the training step.
        """

        # Get the images features and labels.
        images, labels = data

        # Get the teacher predictions.
        teacher_logits = self.trained_teacher(images)
        #print(teacher_logits.shape())


        with tf.GradientTape() as tape:

            # Get the student predictions.
            student_logits = self.student(images)
            #print(student_logits.shape())

            # Compute the knowledge distillation loss.
            loss = self.kd_loss(student_logits, teacher_logits, labels, self.temperature, self.alpha, self.beta)
            print("The loss for this step is {loss}") 

        # Initalize the gradient tape on the trainable parameters.
        gradients = tape.gradient(loss, self.student.trainable_variables)

        # Modify the gradients as described in the https://arxiv.org/abs/1503.02531
        gradients = [gradient * (self.temperature**2) for gradient in gradients]

        # Apply the gradients using the optimizer.
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        # Backpropagation.
        train_loss.update_state(loss)
        train_accuracy.update_state(labels, tf.nn.softmax(student_logits))

        # Fetch the training loss and accuracy.
        _train_loss, _train_accuracy = train_loss.result(), train_accuracy.result()

        # Reset the training loss, and accuracy states.
        train_loss.reset_states(), train_accuracy.reset_states()

        return {"train_loss": _train_loss, "train_accuracy": _train_accuracy}

    def test_step(self, data):
        """
        This method defines the testing/validation step.
        """

        # Get the images features and labels.
        images, labels = data

        # Get the teacher predictions.
        teacher_logits = self.trained_teacher(images)

        # Get the student predictions.
        student_logits = self.student(images, training=False)

        # Compute the knowledge distillation loss.
        loss = self.kd_loss(student_logits, teacher_logits, labels, self.temperature, self.alpha, self.beta)

        # Backpropagation.
        validation_loss.update_state(loss)
        validation_accuracy.update_state(labels, tf.nn.softmax(student_logits))

        # Fetch the validation loss and accuracy.
        _validation_loss, _validation_accuracy = validation_loss.result(), validation_accuracy.result()

        # Reset the validation loss, and accuracy states.
        validation_loss.reset_states(), validation_accuracy.reset_states()

        return {"val_loss": _validation_loss, "val_accuracy": _validation_accuracy}


    def kd_loss(self, student_logits, teacher_logits, true_labels, temperature, alpha, beta):
        """
        Returns the Knowledge Distillation loss.
        """

        # Computes teacher probabilities.
        teacher_probs = teacher_logits / temperature

        # Computes student probabilities.
        student_probs = student_logits / temperature

        # Computes the Knowledge Distillation Loss.
        kd_loss = tf.keras.losses.categorical_crossentropy(teacher_probs, student_probs, from_logits=False)

        # Computes the Cross entropy loss using actual/true labels and student predictions.
        ce_loss = tf.keras.losses.categorical_crossentropy(true_labels, student_logits, from_logits=False)

        # Computes the total loss.
        total_loss = (alpha * kd_loss) + (beta * ce_loss)
        total_loss = total_loss / (alpha + beta)

        return total_loss


