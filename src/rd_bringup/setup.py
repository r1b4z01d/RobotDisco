from glob import glob

from setuptools import find_packages, setup

package_name = 'rd_bringup'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/services', glob('services/*.conf')),
    ],
    install_requires=['setuptools', 'pyyaml'],
    zip_safe=True,
    tests_require=['pytest'],
    maintainer='Bryan Ribas',
    maintainer_email='bryanribas@gmail.com',
    description='Robot Disco top-level launch files, the robot.yaml config loader, supervisor units',
    license='CC-BY-NC-SA-4.0',
    entry_points={'console_scripts': ['rd_config = rd_bringup.config:main']},
)
