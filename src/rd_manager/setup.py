import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'rd_manager'


def tree(src, dst):
    out = []
    for root, _dirs, files in os.walk(src):
        if files:
            out.append((os.path.join(dst, os.path.relpath(root, src)), [os.path.join(root, f) for f in files]))
    return out


setup(
    name=package_name, version='2.0.0', packages=find_packages(exclude=['test']),
    data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                ('share/' + package_name, ['package.xml']),
                ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
                ('share/' + package_name + '/routes', glob('routes/*.yaml'))] + tree('static', 'share/' + package_name + '/static'),
    install_requires=['setuptools', 'numpy', 'pyyaml'], zip_safe=True,
    maintainer='Bryan Ribas', maintainer_email='bryanribas@gmail.com', description='Robot Disco web manager', license='CC-BY-NC-SA-4.0',
    entry_points={'console_scripts': ['manager = rd_manager.main:main']},
)
