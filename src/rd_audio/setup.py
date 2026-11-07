from glob import glob

from setuptools import find_packages, setup

package_name = 'rd_audio'

setup(
    name=package_name, version='2.0.0', packages=find_packages(exclude=['test']),
    data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                ('share/' + package_name, ['package.xml']),
                ('share/' + package_name + '/launch', glob('launch/*.launch.py'))],
    install_requires=['setuptools', 'numpy'], zip_safe=True,
    maintainer='Bryan Ribas', maintainer_email='bryanribas@gmail.com', description='Robot Disco audio', license='CC-BY-NC-SA-4.0',
    entry_points={'console_scripts': ['say_server = rd_audio.say_server:main', 'kokoro_worker = rd_audio.kokoro_worker:main']},
)
