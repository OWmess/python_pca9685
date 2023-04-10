import time
from pca9685 import PCA9685
from pca9685 import I2C
# Create the I2C bus interface.
# nx bus_number:1,8
# nano bus_number:0,1 (?)
i2c_bus =I2C(8)
# Create a simple PCA9685 class instance.
pca = PCA9685(i2c_bus)

# Set the PWM frequency to 60hz.
pca.frequency = 60

# Set the PWM duty cycle for channel zero to 50%. duty_cycle is 16 bits to match other PWM objects
# but the PCA9685 will only actually give 12 bits of resolution.
# 以下代码为：将0，1通道 占空比设置为100%,50%,0%,时间间隔为2s
while(True):
    pca.channels[1].duty_cycle = 0xFFFF
    pca.channels[0].duty_cycle = 0xFFFF
    time.sleep(2)
    pca.channels[1].duty_cycle = 0x7FFF
    pca.channels[0].duty_cycle = 0x7FFF
    time.sleep(2)
    pca.channels[1].duty_cycle = 0x0000
    pca.channels[0].duty_cycle = 0x0000
    time.sleep(2)