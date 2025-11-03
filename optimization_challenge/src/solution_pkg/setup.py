from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'solution_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        
        *[(os.path.join('share', package_name, 'urdf'), [f])
          for f in glob('urdf/**/*', recursive=True)
          if os.path.isfile(f)],

        *[(os.path.join('share', package_name, 'meshes'), [f])
          for f in glob('meshes/**/*', recursive=True)
          if os.path.isfile(f)],
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nishant',
    maintainer_email='nishant1832000@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'solution_node = solution_pkg.solution_node:main'
        ],
    },
)
