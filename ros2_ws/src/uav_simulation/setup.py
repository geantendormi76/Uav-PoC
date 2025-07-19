from setuptools import find_packages, setup

package_name = 'uav_simulation'

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
    maintainer_email='geartendormi76@gmail.com',
    description='A package to simulate drone sensors and data.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'video_publisher_node = uav_simulation.video_publisher_node:main',
        ],
    },
)