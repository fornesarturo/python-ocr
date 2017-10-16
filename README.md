# Python OCR
* [fornesarturo](https://github.com/fornesarturo/)
* [hermesespinola](https://github.com/hermesespinola/)

## Libraries / Frameworks

* OpenCV
* Tesseract
* numpy
* imutils

## Install

For *numpy* and *imutils*:
```bash
$ pip install numpy
$ pip install imutils
```

For *OpenCV 3.3.0*:

### First get the needed libraries

Build-tools:
```bash
$ sudo apt-get install build-essential cmake pkg-config
```

Image libraries:
```bash
$ sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
```

Video libraries:
```bash
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
$ sudo apt-get install libxvidcore-dev libx264-dev
```

OpenCV GUI for image display:
```bash
$ sudo apt-get install libgtk-3-dev
```

Libraries used to optimize operations in OpenCV:
```bash
$ sudo apt-get install libatlas-base-dev gfortran
```

Python dev tools:
```bash
$ sudo apt-get install python2.7-dev python3.5-dev
```

### Download OpenCV Source

Get both OpenCV and OpenCV contrib source:
```bash
$ cd ~
$ wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.3.0.zip
$ unzip opencv.zip
$ wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip
$ unzip opencv_contrib.zip
```

Get Python's virtual environments:

```bash
$ sudo pip install virtualenv virtualenvwrapper
```

Add the following lines to your *.bashrc*, *.zshrc*, etc.:

```bash
$ export WORKON_HOME=$HOME/.virtualenvs
$ source /usr/local/bin/virtualenvwrapper.sh
```

Reload your terminal session by closing and opening or:

```bash
$ source ~/.bashrc
# or
$ source ~/.zshrc
```

Set-up a Python virtual environment:

```bash
$ mkvirtualenv cv -p python3
$ workon cv # to enter the virtual environment
$ pip install numpy
```

Configure the build

```bash
$ workon cv
$ cd ~/opencv-3.3.0/
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.1.0/modules \
    -D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python \
    -D BUILD_EXAMPLES=ON ..
```

Compile OpenCV:

```bash
$ make -j4 # substitute the 4 with your no. of cores
```

If there were no errors run:

```bash
$ sudo make install
$ sudo ldconfig
```

Link and be done:

```bash
$ ls -l /usr/local/lib/python3.5/site-packages/
# you should see the following file listed:
# cv2.cpython-35m-x86_64-linux-gnu.so
$ cd ~/.virtualenvs/cv/lib/python3.5/site-packages/
$ ln -s /usr/local/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so cv2.so
```

Test OpenCV by opening a Python terminal with:

```bash
$ cd ~
$ workon cv
$ python3
```

Then run the following Python code:

```python
import cv2
cv2.__version__
# Output: '3.3.0'
```

Clean-up:

```bash
$ cd ~
$ rm -rf opencv-3.3.0 opencv_contrib-3.3.0 opencv.zip opencv_contrib.zip
```
