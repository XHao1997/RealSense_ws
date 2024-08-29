from setuptools import find_packages, setup

package_name = 'data_collection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'collect_rgb = data_collection.collect_rgb:main',
            'telekey = data_collection.telekey:main'
        ],
    },
)