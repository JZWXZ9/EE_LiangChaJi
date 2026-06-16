from machine import Pin
from machine import UART
from machine import FPIOA
import time

# 实例化FPIOA
fpioa = FPIOA()

# 为IO分配相应的硬件功能
fpioa.set_function(34, FPIOA.GPIO34)
fpioa.set_function(35, FPIOA.GPIO35)
fpioa.set_function(40,FPIOA.UART1_TXD)
fpioa.set_function(41,FPIOA.UART1_RXD)
fpioa.set_function(44,FPIOA.UART2_TXD)
fpioa.set_function(45,FPIOA.UART2_RXD)

# 构造UART对象
key0 = Pin(34, Pin.IN, pull=Pin.PULL_UP, drive=7)
key1 = Pin(35, Pin.IN, pull=Pin.PULL_UP, drive=7)
uart1 = UART(UART.UART1, baudrate=115200, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)
uart2 = UART(UART.UART2, baudrate=115200, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)

while True:
    if key0.value() == 0:
        time.sleep_ms(20)
        if key0.value() == 0:
            # UART发送数据
            uart1.write("ABC")
            while key0.value() == 0:
                pass
    elif key1.value() == 0:
        time.sleep_ms(20)
        if key1.value() == 0:
            # UART发送数据
            uart2.write("B")
            while key1.value() == 0:
                pass

    # UART接收数据
    data = uart1.read(128)
    if data != None:

        print(data.decode())

    # UART接收数据
    data = uart2.read(128)
    if data != None:
        if data == b'ABC':
            print(1)
        print(data)
