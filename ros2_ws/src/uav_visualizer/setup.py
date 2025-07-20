# 必须与您GitHub上成功的版本完全一致
from setuptools import find_packages, setup

package_name = 'uav_visualizer'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zhz',
    maintainer_email='geantendormi76@gmail.com',
    description='A package to visualize UAV perception results.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'visualizer_node = uav_visualizer.visualizer_node:main',
        ],
    },
)