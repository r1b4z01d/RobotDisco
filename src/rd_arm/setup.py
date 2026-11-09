from glob import glob

from setuptools import find_packages, setup

package_name = 'rd_arm'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.xacro') + glob('config/*.srdf') + glob('config/*.xrdf')),
        ('share/' + package_name + '/motions', glob('motions/*.json')),
    ],
    install_requires=['setuptools', 'pyyaml', 'numpy'],
    zip_safe=True,
    tests_require=['pytest'], extras_require={'test': ['pytest']},
    maintainer='Bryan Ribas',
    maintainer_email='bryanribas@gmail.com',
    description='Robot Disco arm and hand',
    license='CC-BY-NC-SA-4.0',
    entry_points={'console_scripts': [
        'hand = rd_arm.hand_node:main',
        'arm_power = rd_arm.arm_power:main',
        'wave_server = rd_arm.wave_server:main',
        'stream = rd_arm.stream:main',
        'fake_hand = rd_arm.fake_hand:main',
        'fake_dashboard = rd_arm.fake_dashboard:main',
        'description_publisher = rd_arm.description_publisher:main',
    ]},
)
