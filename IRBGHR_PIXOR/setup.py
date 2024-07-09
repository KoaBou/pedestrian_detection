from setuptools import setup
import os

package_name = 'IRBGHR_PIXOR'

packages = [package_name]
packages.append(os.path.join(package_name, 'datasets') )
packages.append(os.path.join(package_name, 'losses')) 
packages.append(os.path.join(package_name, 'models')) 
packages.append(os.path.join(package_name, 'utils_1'))
packages.append(os.path.join(package_name, 'models', 'backbones'))
packages.append(os.path.join(package_name, 'models', 'heads'))
packages.append(os.path.join(package_name, 'models', 'torchscript'))


setup(
    name=package_name,
    version='0.0.0',
    packages=packages,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
        ],
    },
)
