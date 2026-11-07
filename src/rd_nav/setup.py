from glob import glob

from setuptools import find_packages, setup

package_name = 'rd_nav'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/maps', glob('maps/*.pgm') + glob('maps/*.yaml') + glob('maps/active_map')),
        ('share/' + package_name + '/landmarks', glob('landmarks/*.yaml')),
    ],
    install_requires=['setuptools', 'pyyaml', 'numpy'],
    zip_safe=True,
    maintainer='Bryan Ribas',
    maintainer_email='bryanribas@gmail.com',
    description='Robot Disco navigation',
    license='CC-BY-NC-SA-4.0',
    entry_points={'console_scripts': [
        'go_to_landmark = rd_nav.go_to_landmark:main',
        'tag_corrector = rd_nav.tag_corrector:main',
        'rd_keepout = rd_nav.keepout:main',
        'scan_relay = rd_nav.scan_relay:main',
    ]},
)
