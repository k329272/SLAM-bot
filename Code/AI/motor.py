# motor.py
# L298N motor control using pigpio, encoder reading, and PID velocity control.
# Designed for Raspberry Pi Zero 2 W.

import time
import threading
import pigpio
import math

class PID:
    def __init__(self, kp, ki, kd, integrator_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integrator = 0.0
        self.last_err = 0.0
        self.last_t = None
        self.integrator_limit = integrator_limit

    def reset(self):
        self.integrator = 0.0
        self.last_err = 0.0
        self.last_t = None

    def update(self, error):
        t = time.time()
        if self.last_t is None:
            dt = 1e-3
        else:
            dt = max(1e-4, t - self.last_t)
        self.last_t = t

        self.integrator += error * dt
        if self.integrator_limit:
            self.integrator = max(-self.integrator_limit, min(self.integrator_limit, self.integrator))

        derivative = (error - self.last_err) / dt
        self.last_err = error

        return self.kp * error + self.ki * self.integrator + self.kd * derivative

class Encoder:
    def __init__(self, pi, gpio_pin, callback_edge=pigpio.EITHER_EDGE):
        self.pi = pi
        self.gpio = gpio_pin
        self._count = 0
        self._lock = threading.Lock()
        self._cb = pi.callback(self.gpio, callback_edge, self._cbf)

    def _cbf(self, gpio, level, tick):
        # level: 0, 1, or 2 if watchdog
        if level == 2:
            return
        with self._lock:
            # Simple single-channel counting. If quadrature use two pins and decode.
            if level == 1:
                self._count += 1
            else:
                self._count -= 1

    def read_and_reset(self):
        with self._lock:
            c = self._count
            self._count = 0
            return c

    def get_count(self):
        with self._lock:
            return self._count

class Motor:
    def __init__(self, pi, in1_pin, in2_pin, pwm_pin, encoder_pin, pwm_freq=1000, reversed=False):
        self.pi = pi
        self.in1 = in1_pin
        self.in2 = in2_pin
        self.pwm = pwm_pin
        self.reversed = reversed

        # set pins as outputs
        pi.set_mode(self.in1, pigpio.OUTPUT)
        pi.set_mode(self.in2, pigpio.OUTPUT)
        pi.set_mode(self.pwm, pigpio.OUTPUT)
        pi.set_PWM_frequency(self.pwm, pwm_freq)
        pi.set_PWM_range(self.pwm, 1000)

        self.encoder = Encoder(pi, encoder_pin)
        self.rpm = 0.0
        self._last_encoder_time = time.time()

        # wheel / motor params (user must configure)
        self.counts_per_rev = 20      # encoder pulses per motor rev (adjust)
        self.gear_ratio = 50         # gearbox ratio (adjust to your DFROBOT TT)
        self.wheel_diameter_m = 0.06  # wheel diameter
        self._vel_lock = threading.Lock()
        self.target_rpm = 0.0

        # PID for velocity (RPM)
        self.pid = PID(kp=0.6, ki=0.2, kd=0.01, integrator_limit=100)

        self._stop_flag = False
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()

    def set_target_rpm(self, rpm):
        with self._vel_lock:
            self.target_rpm = -rpm if self.reversed else rpm

    def get_target_rpm(self):
        with self._vel_lock:
            return self.target_rpm

    def _set_pwm_direction(self, pwm_value):
        # pwm_value in [-1000, 1000]
        if pwm_value > 0:
            self.pi.write(self.in1, 1)
            self.pi.write(self.in2, 0)
            self.pi.set_PWM_dutycycle(self.pwm, min(1000, int(pwm_value)))
        elif pwm_value < 0:
            self.pi.write(self.in1, 0)
            self.pi.write(self.in2, 1)
            self.pi.set_PWM_dutycycle(self.pwm, min(1000, int(-pwm_value)))
        else:
            # stop
            self.pi.write(self.in1, 0)
            self.pi.write(self.in2, 0)
            self.pi.set_PWM_dutycycle(self.pwm, 0)

    def _compute_rpm(self, counts, dt):
        # counts are encoder ticks per dt. Convert to wheel RPM.
        if dt <= 0:
            return 0.0
        # motor revs = counts / counts_per_rev
        motor_revs = counts / float(self.counts_per_rev)
        motor_rpm = (motor_revs / dt) * 60.0
        # convert motor rpm to wheel rpm via gear ratio
        wheel_rpm = motor_rpm / float(self.gear_ratio)
        return wheel_rpm

    def _control_loop(self):
        last_time = time.time()
        while not self._stop_flag:
            t0 = time.time()
            counts = self.encoder.read_and_reset()
            dt = max(1e-3, t0 - last_time)
            last_time = t0
            measured_rpm = self._compute_rpm(counts, dt)
            self.rpm = measured_rpm

            target = self.get_target_rpm()
            error = target - measured_rpm
            pwm_out = self.pid.update(error)

            # scale PID output into PWM range
            pwm_val = max(-1000.0, min(1000.0, pwm_out * 50.0))  # tune multiplier
            self._set_pwm_direction(pwm_val)

            time.sleep(0.02)

    def stop(self):
        self._stop_flag = True
        self._set_pwm_direction(0)
