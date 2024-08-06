from setuptools import setup
from glob import glob
import os

package_name = 'pedestrian_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, 'utils'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ngin',
    maintainer_email='nguyentrungnguyen20202004@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pedestrian_detector = pedestrian_detection.pedestrian_detection:main',
            'detect_with_score = pedestrian_detection.detectWscore:main',
        ],
    },
)
