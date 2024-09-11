from setuptools import setup
from setuptools import find_packages
from glob import glob
import os
package_name = 'ultralytics_ros'


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.py'))),
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hao',
    maintainer_email='xuhao6815@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracker_node.py = ultralytics_ros.tracker_node:main',
        ],
    },
)
