from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    rviz_default_path = "/home/ngin/pcl_detect_ws/src/pedestrian_detection/rviz/config.rviz"
    dataFolder_default = "packard-poster-session-2019-03-20_1"

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=['-d', LaunchConfiguration('rvizconfig')]
    )

    jrdb_publisher = Node(
        package="jrdb_publishers",
        executable="jrdb_publisher",
        parameters=[{"dataFolder": LaunchConfiguration('dataFolder')}]
    )

    # pedestrian_detector = Node(
    #     package='pedestrian_detection',
    #     executable='pedestrian_detector'
    # )

    pedestrian_detector = Node(
        package='pedestrian_detection',
        executable='detect_with_score'
    )

    print(LaunchConfiguration('numDir').parse)

    return LaunchDescription([
        DeclareLaunchArgument(
            name="rvizconfig", default_value=rviz_default_path
        ),
        DeclareLaunchArgument(
            name="dataFolder", default_value=dataFolder_default
        ),
        rviz2,
        jrdb_publisher,
        pedestrian_detector
    ])

