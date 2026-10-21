import cv_bridge
import rclpy
import sensor_msgs.msg
from smarty_utils.enums import NodeState
from smarty_utils.smarty_node import SmartyNode
from timing import timer

from ros2_example_package.example_model import model


class ROS2ExampleNode(SmartyNode):
    """ROS2 Example Node."""

    def __init__(self):
        """Initialize the ROS2ExampleNode."""
        super().__init__(
            "ros2_example_node",
            "ros2_example_package",
            node_parameters={
                # Subscriber topics
                "image_subscriber": "/camera/undistorted",
                # Publisher topics
                "debug_image_publisher": "/ros2_example_node/debug_image",
                # Parameters
                "model_config_path": "config/model.yaml",
                "state": NodeState.ACTIVE.value,
                "example_value": 123,
                "debug": False,
            },
            subscribed_topics={
                "image_subscriber": (
                    sensor_msgs.msg.Image,
                    self.image_callback,
                    1,
                ),
            },
            published_topics={
                "debug_image_publisher": (sensor_msgs.msg.Image, 1),
            },
        )

        # Get the parameters from the yaml file
        self.get_logger().info(f"Package Path: {self.package_path}")

        # Create required objects
        self.cv_bridge = cv_bridge.CvBridge()

        self.model = model.ExampleModel(
            root=self.package_path, model_config_path=self.model_config_path
        )

        self.get_logger().info("ROS2ExampleNode initialized")

    @property
    def model_config_path(self) -> str:
        """Return the model config path."""
        return self.get_parameter("model_config_path").value  # type: ignore

    @property
    def example_value(self) -> int:
        """Return an example value for debug image generation."""
        return self.get_parameter("example_value").value  # type: ignore

    def _reset(self) -> None:
        """Reset the node to its initial state."""
        del self.model
        self.model = model.ExampleModel(
            root=self.package_path, model_config_path=self.model_config_path
        )

    def image_callback(self, msg: sensor_msgs.msg.Image):
        """Executed by the ROS2 system whenever a new image is received."""
        # Execute the prediction
        self.execute_prediction(msg)

    @timer.Timer(name="total", filter_strength=40)
    def execute_prediction(self, msg: sensor_msgs.msg.Image):
        """
        Execute the prediction.

        Arguments:
            msg -- The image message.
        """
        with timer.Timer(name="msg_transport", filter_strength=40):
            # The image has to be retrieved from the message
            image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="8UC1")

        with timer.Timer(name="prediction", filter_strength=40):
            # Get the result from the model
            result = self.model.predict(image)

        with timer.Timer(name="publish", filter_strength=40):
            # Publish the result as the custom ROS2 message defined in this package
            print(f"Prediction result: {result}")

        if self._debug:
            with timer.Timer(name="debug", filter_strength=40):
                # Create the debug image. Here it is just the image filled with
                # the example value.
                debug_image = image.copy()
                debug_image.fill(self.example_value)

                self.debug_image_publisher.publish(  # type: ignore
                    self.cv_bridge.cv2_to_imgmsg(image, encoding="8UC1")
                )

        timer.Timer(logger=self.get_logger().info).print()  # type: ignore


def main(args=None):
    """
    Main function to start the LaneDetectionNode.

    Keyword Arguments:
        args -- Launch arguments (default: {None})
    """
    rclpy.init(args=args)
    node = ROS2ExampleNode()

    # We have 2 options on how to run the node:
    # 1. Let the node idle in the background with 'rclpy.spin(node)' if we want to let
    #   subscriber callback function handle the execution of our code.
    #   TODO: is it possible in this way, that our callback gets executed multiple times
    #       in parallel?
    # 2. Run the node in a while loop that waits for incoming messages and then executes
    #   our code. This makes sure that always the latest message is processed and never
    #   multiple messages in parallel. It should be used if the processing of the
    #   message/execution of our code takes longer than the time between incoming #
    #   messages.

    try:
        use_wait_for_message = True
        if use_wait_for_message:
            while rclpy.ok():
                rclpy.spin_once(node)
        else:
            rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        # Shutdown if not already done by the ROS2 launch system
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
