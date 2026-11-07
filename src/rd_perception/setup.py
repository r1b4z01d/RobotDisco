from glob import glob

from setuptools import find_packages, setup

package_name = 'rd_perception'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    tests_require=['pytest'], extras_require={'test': ['pytest']},
    maintainer='Bryan Ribas',
    maintainer_email='bryanribas@gmail.com',
    description='Robot Disco sensors',
    license='CC-BY-NC-SA-4.0',
    entry_points={'console_scripts': [
        'scan_range_offset = rd_perception.scan_range_offset:main',
        'tag_adapter = rd_perception.tag_adapter:main',
        'fake_tags = rd_perception.fake_tags:main',
        'image_rotate = rd_perception.image_rotate:main',
        'fake_cameras = rd_perception.fake_cameras:main',
    ]},
)
