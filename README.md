# 该项目基于Adafruit_CircuitPython_PCA9685修改，能在linux上使用python驱动pca9685模块

- python版本应不低于```3.6.9```，若python版本恰为```3.6.9```，应安装```typing_extensions```模块
- 使用方法示例：```pca9865_simpletest.py```
- 使用者需将```pca9685.py与smbus.py文件拷贝到自己的项目中```

- 使用者应知晓自己的i2c总线编号，下面给出部分设备的总线编号
  - nx bus_number:1,8
  - nano bus_number:0,1 (存疑)
  - 也可以在接好线后，通过实例化PCA9685类尝试各个i2c总线，若不存在pca9685模块，则会报错```ValueError: No I2C device at address: 0x40```,下面为示例代码：
  ```
      from adafruit_pca9685 import PCA9685
      from adafruit_pca9685 import I2C
      PCA9685(I2C(${bus_number}))
  ```
  - 可通过```ls /dev/i2c-*```命令查看当前存在的i2c总线
  


## Adafruit_CircuitPython_PCA9685项目地址：[click_here](https://github.com/adafruit/Adafruit_CircuitPython_PCA9685)
