
import numpy as np
import tensorflow as tf
from PIL import Image


class CNNPilot:
    def __init__(self, model_path):
        graph_def = tf.GraphDef()

        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())

        self.graph = tf.Graph()

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name="")

        self.input = self.graph.get_tensor_by_name(
            "camera_input:0"
        )
        self.output = self.graph.get_tensor_by_name(
            "steering_output:0"
        )

        self.session = tf.Session(graph=self.graph)

    def run(self, image):
        if image is None:
            return 0.0, 0.0

        image = Image.fromarray(
            np.asarray(image).astype(np.uint8)
        ).convert("RGB")

        image = image.resize((64, 64))

        array = np.asarray(
            image, dtype=np.float32
        ) / 255.0

        array = np.expand_dims(array, axis=0)

        prediction = self.session.run(
            self.output,
            feed_dict={self.input: array}
        )

        steering = float(prediction[0][0])
        steering = np.clip(steering, -1.0, 1.0)

        throttle = 0.0  # Safety: disabled initially

        return steering, throttle

    def shutdown(self):
        self.session.close()

