from glob import glob

from setuptools import find_packages, setup

package_name = 'rd_sim'

setup(
    name=package_name, version='2.0.0', packages=find_packages(exclude=['test']),
    data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                ('share/' + package_name, ['package.xml']),
                ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
                ('share/' + package_name + '/config', glob('config/*.yaml')),
                ('share/' + package_name + '/scripts', glob('scripts/*.py'))],
    install_requires=['setuptools', 'pyyaml'], zip_safe=True, tests_require=['pytest'], extras_require={'test': ['pytest']},
    maintainer='Bryan Ribas', maintainer_email='bryanribas@gmail.com', description='Robot Disco in Isaac Sim', license='CC-BY-NC-SA-4.0',
    entry_points={'console_scripts': ['export_urdf = rd_sim.export_urdf:main', 'arm_mirror = rd_sim.arm_mirror:main']},
)
