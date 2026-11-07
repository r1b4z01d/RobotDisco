from glob import glob

from setuptools import find_packages, setup

package_name = 'rd_face'

setup(
    name=package_name, version='2.0.0', packages=find_packages(exclude=['test']),
    data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                ('share/' + package_name, ['package.xml']),
                ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
                ('share/' + package_name + '/config', glob('config/*.yaml')),
                ('share/' + package_name + '/models', glob('models/*.onnx') + glob('models/README.md'))],
    install_requires=['setuptools', 'numpy'], zip_safe=True,
    maintainer='Bryan Ribas', maintainer_email='bryanribas@gmail.com', description='Robot Disco face', license='CC-BY-NC-SA-4.0',
    entry_points={'console_scripts': ['face = rd_face.face_node:main', 'faceid_worker = rd_face.faceid_worker:main']},
)
