import time
import struct
import array
from types import TracebackType
from typing import Union, Optional, Type, Tuple, Any
from typing_extensions import Protocol, TypeAlias  # Safety import for Python 3.7
import smbus

class I2C:
    """I2C class,通过smbus，实现对i2c设备的操作
    :param bus_num: i2c的总线编号，如：/dev/I2C-1 ，即 1
    """

    MASTER = 0
    SLAVE = 1
    _baudrate = None
    _mode = None
    _i2c_bus = None

    # pylint: disable=unused-argument
    def __init__(self, bus_num, mode=MASTER, baudrate=None):
        if mode != self.MASTER:
            raise NotImplementedError("Only I2C Master supported!")
        _mode = self.MASTER

        # if baudrate != None:
        #    print("I2C frequency is not settable in python, ignoring!")
        #创建一个smbus实例
        try:
            self._i2c_bus = smbus.SMBus(bus_num)
        except FileNotFoundError:
            raise RuntimeError(
                "I2C Bus #%d not found, check if enabled in config!" % bus_num
            ) from RuntimeError

    # pylint: enable=unused-argument

    def scan(self):
        """Try to read a byte from each address, if you get an OSError
        it means the device isnt there"""
        found = []
        for addr in range(0, 0x80):
            try:
                self._i2c_bus.read_byte(addr)
            except OSError:
                continue
            found.append(addr)
        return found

    # pylint: disable=unused-argument
    def writeto(self, address, buffer, *, start=0, end=None, stop=True):
        """Write data from the buffer to an address"""
        if end is None:
            end = len(buffer)
        self._i2c_bus.write_bytes(address, buffer[start:end])

    def readfrom_into(self, address, buffer, *, start=0, end=None, stop=True):
        """Read data from an address and into the buffer"""
        if end is None:
            end = len(buffer)

        readin = self._i2c_bus.read_bytes(address, end - start)
        for i in range(end - start):
            buffer[i + start] = readin[i]

    # pylint: enable=unused-argument

    def writeto_then_readfrom(
        self,
        address,
        buffer_out,
        buffer_in,
        *,
        out_start=0,
        out_end=None,
        in_start=0,
        in_end=None,
        stop=False,
    ):
        """Write data from buffer_out to an address and then
        read data from an address and into buffer_in
        """
        if out_end is None:
            out_end = len(buffer_out)
        if in_end is None:
            in_end = len(buffer_in)
        if stop:
            # To generate a stop in linux, do in two transactions
            self.writeto(address, buffer_out, start=out_start, end=out_end, stop=True)
            self.readfrom_into(address, buffer_in, start=in_start, end=in_end)
        else:
            # To generate without a stop, do in one block transaction
            readin = self._i2c_bus.read_i2c_block_data(
                address, buffer_out[out_start:out_end], in_end - in_start
            )
            for i in range(in_end - in_start):
                buffer_in[i + in_start] = readin[i]


ReadableBuffer: TypeAlias = Union[
    array.array,
    bytearray,
    bytes,
    memoryview,
    "rgbmatrix.RGBMatrix",
    "ulab.numpy.ndarray",
]


WriteableBuffer: TypeAlias = Union[
    array.array,
    bytearray,
    memoryview,
    "rgbmatrix.RGBMatrix",
    "ulab.numpy.ndarray",
]
class I2CDevice:
    """
    Represents a single I2C device.

    :param I2C i2c: The I2C bus the device is on
    :param int device_address: The 7 bit device address
    :param bool probe: Probe for the device upon object creation, default is true

    """

    def __init__(self, i2c: I2C, device_address: int, probe: bool = True) -> None:

        self.i2c = i2c
        self.device_address = device_address

        if probe:
            self.__probe_for_device()

    def readinto(
        self, buf: WriteableBuffer, *, start: int = 0, end: Optional[int] = None
    ) -> None:
        """
        Read into ``buf`` from the device. The number of bytes read will be the
        length of ``buf``.

        If ``start`` or ``end`` is provided, then the buffer will be sliced
        as if ``buf[start:end]``. This will not cause an allocation like
        ``buf[start:end]`` will so it saves memory.

        :param ~WriteableBuffer buffer: buffer to write into
        :param int start: Index to start writing at
        :param int end: Index to write up to but not include; if None, use ``len(buf)``
        """
        if end is None:
            end = len(buf)
        self.i2c.readfrom_into(self.device_address, buf, start=start, end=end)

    def write(
        self, buf: ReadableBuffer, *, start: int = 0, end: Optional[int] = None
    ) -> None:
        """
        Write the bytes from ``buffer`` to the device, then transmit a stop
        bit.

        If ``start`` or ``end`` is provided, then the buffer will be sliced
        as if ``buffer[start:end]``. This will not cause an allocation like
        ``buffer[start:end]`` will so it saves memory.

        :param ~ReadableBuffer buffer: buffer containing the bytes to write
        :param int start: Index to start writing from
        :param int end: Index to read up to but not include; if None, use ``len(buf)``
        """
        if end is None:
            end = len(buf)
        self.i2c.writeto(self.device_address, buf, start=start, end=end)

    # pylint: disable-msg=too-many-arguments
    def write_then_readinto(
        self,
        out_buffer: ReadableBuffer,
        in_buffer: WriteableBuffer,
        *,
        out_start: int = 0,
        out_end: Optional[int] = None,
        in_start: int = 0,
        in_end: Optional[int] = None
    ) -> None:
        """
        Write the bytes from ``out_buffer`` to the device, then immediately
        reads into ``in_buffer`` from the device. The number of bytes read
        will be the length of ``in_buffer``.

        If ``out_start`` or ``out_end`` is provided, then the output buffer
        will be sliced as if ``out_buffer[out_start:out_end]``. This will
        not cause an allocation like ``buffer[out_start:out_end]`` will so
        it saves memory.

        If ``in_start`` or ``in_end`` is provided, then the input buffer
        will be sliced as if ``in_buffer[in_start:in_end]``. This will not
        cause an allocation like ``in_buffer[in_start:in_end]`` will so
        it saves memory.

        :param ~ReadableBuffer out_buffer: buffer containing the bytes to write
        :param ~WriteableBuffer in_buffer: buffer containing the bytes to read into
        :param int out_start: Index to start writing from
        :param int out_end: Index to read up to but not include; if None, use ``len(out_buffer)``
        :param int in_start: Index to start writing at
        :param int in_end: Index to write up to but not include; if None, use ``len(in_buffer)``
        """
        if out_end is None:
            out_end = len(out_buffer)
        if in_end is None:
            in_end = len(in_buffer)

        self.i2c.writeto_then_readfrom(
            self.device_address,
            out_buffer,
            in_buffer,
            out_start=out_start,
            out_end=out_end,
            in_start=in_start,
            in_end=in_end,
        )

    # pylint: enable-msg=too-many-arguments

    def __enter__(self) -> "I2CDevice":
        # while not self.i2c.try_lock():
        #     time.sleep(0)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[type]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        # self.i2c.unlock()
        return False

    def __probe_for_device(self) -> None:
        """
        Try to read a byte from an address,
        if you get an OSError it means the device is not there
        or that the device does not support these means of probing
        """
        # while not self.i2c.try_lock():
        #     time.sleep(0)
        try:
            self.i2c.writeto(self.device_address, b"")
        except OSError:
            # some OS's dont like writing an empty bytesting...
            # Retry by reading a byte
            try:
                result = bytearray(1)
                self.i2c.readfrom_into(self.device_address, result)
            except OSError:
                # pylint: disable=raise-missing-from
                raise ValueError("No I2C device at address: 0x%x" % self.device_address)

                # pylint: enable=raise-missing-from
        finally:
            pass
            # self.i2c.unlock()

class I2CDeviceDriver(Protocol):
    """Describes classes that are drivers utilizing `I2CDevice`"""

    i2c_device: I2CDevice
class Struct:
    """
    Arbitrary structure register that is readable and writeable.

    Values are tuples that map to the values in the defined struct.  See struct
    module documentation for struct format string and its possible value types.

    :param int register_address: The register address to read the bit from
    :param str struct_format: The struct format string for this register.
    """

    def __init__(self, register_address: int, struct_format: str) -> None:
        self.format = struct_format
        self.buffer = bytearray(1 + struct.calcsize(self.format))
        self.buffer[0] = register_address

    def __get__(
        self,
        obj: Optional[I2CDeviceDriver],
        objtype: Optional[Type[I2CDeviceDriver]] = None,
    ) -> Tuple:
        with obj.i2c_device as i2c:
            i2c.write_then_readinto(self.buffer, self.buffer, out_end=1, in_start=1)
        return struct.unpack_from(self.format, memoryview(self.buffer)[1:])

    def __set__(self, obj: I2CDeviceDriver, value: Tuple) -> None:
        struct.pack_into(self.format, self.buffer, 1, *value)
        with obj.i2c_device as i2c:
            i2c.write(self.buffer)


class UnaryStruct:
    """
    Arbitrary single value structure register that is readable and writeable.

    Values map to the first value in the defined struct.  See struct
    module documentation for struct format string and its possible value types.

    :param int register_address: The register address to read the bit from
    :param str struct_format: The struct format string for this register.
    """

    def __init__(self, register_address: int, struct_format: str) -> None:
        self.format = struct_format
        self.address = register_address

    def __get__(
        self,
        obj: Optional[I2CDeviceDriver],
        objtype: Optional[Type[I2CDeviceDriver]] = None,
    ) -> Any:
        buf = bytearray(1 + struct.calcsize(self.format))
        buf[0] = self.address
        with obj.i2c_device as i2c:
            i2c.write_then_readinto(buf, buf, out_end=1, in_start=1)
        return struct.unpack_from(self.format, buf, 1)[0]

    def __set__(self, obj: I2CDeviceDriver, value: Any) -> None:
        buf = bytearray(1 + struct.calcsize(self.format))
        buf[0] = self.address
        struct.pack_into(self.format, buf, 1, value)
        with obj.i2c_device as i2c:
            i2c.write(buf)


class _BoundStructArray:
    """
    Array object that `StructArray` constructs on demand.

    :param object obj: The device object to bind to. It must have a `i2c_device` attribute
    :param int register_address: The register address to read the bit from
    :param str struct_format: The struct format string for each register element
    :param int count: Number of elements in the array
    """

    def __init__(
        self,
        obj: I2CDeviceDriver,
        register_address: int,
        struct_format: str,
        count: int,
    ) -> None:
        self.format = struct_format
        self.first_register = register_address
        self.obj = obj
        self.count = count

    def _get_buffer(self, index: int) -> bytearray:
        """Shared bounds checking and buffer creation."""
        if not 0 <= index < self.count:
            raise IndexError()
        size = struct.calcsize(self.format)
        # We create the buffer every time instead of keeping the buffer (which is 32 bytes at least)
        # around forever.
        buf = bytearray(size + 1)
        buf[0] = self.first_register + size * index
        return buf

    def __getitem__(self, index: int) -> Tuple:
        buf = self._get_buffer(index)
        with self.obj.i2c_device as i2c:
            i2c.write_then_readinto(buf, buf, out_end=1, in_start=1)
        return struct.unpack_from(self.format, buf, 1)  # offset=1

    def __setitem__(self, index: int, value: Tuple) -> None:
        buf = self._get_buffer(index)
        struct.pack_into(self.format, buf, 1, *value)
        with self.obj.i2c_device as i2c:
            i2c.write(buf)

    def __len__(self) -> int:
        return self.count

class StructArray:
    """
    Repeated array of structured registers that are readable and writeable.

    Based on the index, values are offset by the size of the structure.

    Values are tuples that map to the values in the defined struct.  See struct
    module documentation for struct format string and its possible value types.

    .. note:: This assumes the device addresses correspond to 8-bit bytes. This is not suitable for
      devices with registers of other widths such as 16-bit.

    :param int register_address: The register address to begin reading the array from
    :param str struct_format: The struct format string for this register.
    :param int count: Number of elements in the array
    """

    def __init__(self, register_address: int, struct_format: str, count: int) -> None:
        self.format = struct_format
        self.address = register_address
        self.count = count
        self.array_id = "_structarray{}".format(register_address)

    def __get__(
        self,
        obj: Optional[I2CDeviceDriver],
        objtype: Optional[Type[I2CDeviceDriver]] = None,
    ) -> _BoundStructArray:
        # We actually can't handle the indexing ourself due to data descriptor limits. So, we return
        # an object that can instead. This object is bound to the object passed in here by its
        # initializer and then cached on the object itself. That way its lifetime is tied to the
        # lifetime of the object itself.
        if not hasattr(obj, self.array_id):
            setattr(
                obj,
                self.array_id,
                _BoundStructArray(obj, self.address, self.format, self.count),
            )
        return getattr(obj, self.array_id)

class PWMChannel:
    """A single PCA9685 channel that matches the :py:class:`~pwmio.PWMOut` API.

    :param PCA9685 pca: The PCA9685 object
    :param int index: The index of the channel
    """

    def __init__(self, pca: "PCA9685", index: int):
        self._pca = pca
        self._index = index

    @property
    def frequency(self) -> float:
        """The overall PWM frequency in Hertz (read-only).
        A PWMChannel's frequency cannot be set individually.
        All channels share a common frequency, set by PCA9685.frequency."""
        return self._pca.frequency

    @frequency.setter
    def frequency(self, _):
        raise NotImplementedError("frequency cannot be set on individual channels")

    @property
    def duty_cycle(self) -> int:
        """16 bit value that dictates how much of one cycle is high (1) versus low (0). 0xffff will
        always be high, 0 will always be low and 0x7fff will be half high and then half low."""
        pwm = self._pca.pwm_regs[self._index]
        if pwm[0] == 0x1000:
            return 0xFFFF
        return pwm[1] << 4

    @duty_cycle.setter
    def duty_cycle(self, value: int) -> None:
        if not 0 <= value <= 0xFFFF:
            raise ValueError(f"Out of range: value {value} not 0 <= value <= 65,535")

        if value == 0xFFFF:
            self._pca.pwm_regs[self._index] = (0x1000, 0)
        else:
            # Shift our value by four because the PCA9685 is only 12 bits but our value is 16
            value = (value + 1) >> 4
            self._pca.pwm_regs[self._index] = (0, value)


class PCAChannels:  # pylint: disable=too-few-public-methods
    """Lazily creates and caches channel objects as needed. Treat it like a sequence.

    :param PCA9685 pca: The PCA9685 object
    """

    def __init__(self, pca: "PCA9685") -> None:
        self._pca = pca
        self._channels = [None] * len(self)

    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int) -> PWMChannel:
        if not self._channels[index]:
            self._channels[index] = PWMChannel(self._pca, index)
        return self._channels[index]


class PCA9685:
    """
    Initialise the PCA9685 chip at ``address`` on ``i2c_bus``.

    The internal reference clock is 25mhz but may vary slightly with environmental conditions and
    manufacturing variances. Providing a more precise ``reference_clock_speed`` can improve the
    accuracy of the frequency and duty_cycle computations. See the ``calibration.py`` example for
    how to derive this value by measuring the resulting pulse widths.

    :param ~busio.I2C i2c_bus: The I2C bus which the PCA9685 is connected to.
    :param int address: The I2C address of the PCA9685.
    :param int reference_clock_speed: The frequency of the internal reference clock in Hertz.
    """

    # Registers:
    mode1_reg = UnaryStruct(0x00, "<B")
    mode2_reg = UnaryStruct(0x01, "<B")
    prescale_reg = UnaryStruct(0xFE, "<B")
    pwm_regs = StructArray(0x06, "<HH", 16)

    def __init__(
        self,
        i2c_bus: I2C,
        *,
        address: int = 0x40,
        reference_clock_speed: int = 25000000,
    ) -> None:
        #from adafruit_bus_device
        self.i2c_device = I2CDevice(i2c_bus, address)
        self.channels = PCAChannels(self)
        """Sequence of 16 `PWMChannel` objects. One for each channel."""
        self.reference_clock_speed = reference_clock_speed
        """The reference clock speed in Hz."""

        self.reset()

    def reset(self) -> None:
        """Reset the chip."""
        self.mode1_reg = 0x00  # Mode1

    @property
    def frequency(self) -> float:
        """The overall PWM frequency in Hertz."""
        prescale_result = self.prescale_reg
        if prescale_result < 3:
            raise ValueError(
                "The device pre_scale register (0xFE) was not read or returned a value < 3"
            )
        return self.reference_clock_speed / 4096 / prescale_result

    @frequency.setter
    def frequency(self, freq: float) -> None:
        prescale = int(self.reference_clock_speed / 4096.0 / freq + 0.5)
        if prescale < 3:
            raise ValueError("PCA9685 cannot output at the given frequency")
        old_mode = self.mode1_reg  # Mode 1
        self.mode1_reg = (old_mode & 0x7F) | 0x10  # Mode 1, sleep
        self.prescale_reg = prescale  # Prescale
        self.mode1_reg = old_mode  # Mode 1
        time.sleep(0.005)
        # Mode 1, autoincrement on, fix to stop pca9685 from accepting commands at all addresses
        self.mode1_reg = old_mode | 0xA0

    def __enter__(self) -> "PCA9685":
        return self

    def __exit__(
        self,
        exception_type: Optional[Type[type]],
        exception_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.deinit()

    def deinit(self) -> None:
        """Stop using the pca9685."""
        self.reset()
