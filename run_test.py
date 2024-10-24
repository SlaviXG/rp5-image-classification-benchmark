import configparser
import datetime
import json
import os.path
import subprocess
import time
from queue import Queue
from threading import Event

from mqtt_system_governor import color_log
from mqtt_system_governor.commander import BaseCommander

# !!!
# Make sure that jsonify = True and save_feedback = False inside mqtt_system_governor/config.ini
# !!!

# Parameters
RASPBERRY_CLIENT_ID = "client1"
MAX_TEMPERATURE_DIFFERENCE = 30
ITERATION_TESTING_TIME = 15
LOGGER_STARTING_COMMAND = "sudo ./fnirsi-usb-power-data-logger/fnirsi_logger.py"
COMMAND_FEEDBACK_FILE = "command_feedback.txt"
LOGGER_OUTPUT_FILE = "data_logger.txt"
START_OPERATOR_COMMAND = "python3 mqtt_system_governor/operator.py --config=mqtt_system_governor/config.ini"

FEEDBACK_QUEUE = Queue()


def save_feedback_to_file(feedback: str):
    with open(COMMAND_FEEDBACK_FILE, 'a') as f:
        f.write(feedback + '\n')


def get_current_time():
    return datetime.datetime.now().replace(microsecond=0).isoformat()


class StressRaspberry:
    class Commander(BaseCommander):
        def __init__(self, broker, port, command_loader_topic, response_topic, jsonify):
            super().__init__(broker, port, command_loader_topic, response_topic, jsonify)
            self._client.on_message = self.on_message

        def on_message(self, client, userdata, msg):
            feedback = msg.payload.decode()
            if self._jsonify:
                try:
                    feedback = json.loads(feedback)
                    if feedback['client_id'] == RASPBERRY_CLIENT_ID:
                        save_feedback_to_file(json.dumps(feedback))
                        FEEDBACK_QUEUE.put(feedback)
                        print(f"{get_current_time()} -- Feedback received and saved (command: {feedback['command']})")
                        # print(f"{json.dumps(feedback, indent=2)}")
                except json.JSONDecodeError as e:
                    print(f"(!) -- {get_current_time()} --Please make sure that feedback was sent in a JSON format\n{feedback}")
            else:
                print(f"(!) -- {get_current_time()} --Please set the `jsonify` option to True inside mqtt_system_governor/config.ini")

    def __init__(self):
        self.commander = self.init_commander(os.path.join('mqtt_system_governor', 'config.ini'))
        self.command_queue = Queue()
        self.fill_command_queue()
        self.operator_process = None
        self._power_data_logger_process = None
        self._save_logger_output = Event()
        self._awaiting_for_feedback = Event()
        self.clear_command_feedback_file()

    def init_commander(self, config_path: os.path):
        config = configparser.ConfigParser()
        config.read(config_path)
        broker = os.getenv('MQTT_BROKER') or config['mqtt']['broker']
        port = int(config['mqtt']['port'])
        command_loader_topic = config['mqtt']['command_loader_topic']
        response_topic = config['mqtt']['response_topic']
        jsonify = config.getboolean('commander', 'jsonify')

        return self.Commander(broker, port, command_loader_topic, response_topic, jsonify)

    def fill_command_queue(self):
        models = ["RESNET50", "MOBILENETV3", "RESNET18", "RESNEXT", "SQUEEZENET"]

        # Fractional FPS values from 1/10 to 1
        for model in models:
            for i in range(10, 0, -1):  # iterate from 10 down to 1
                fps = 1 / i
                self.command_queue.put(f"python run_model.py {model} {fps:.3f} ./images")
            
            # Integer FPS values from 1 to 10
            for i in range(2, 11):  # iterate from 1 to 10
                fps = i
                self.command_queue.put(f"python run_model.py {model} {fps} ./images")

            # Repeat for accelerated models
            for i in range(10, 0, -1):
                fps = 1 / i
                self.command_queue.put(f"python run_model.py {model} {fps:.3f} ./images --accelerate")
            
            for i in range(2, 11):
                fps = i
                self.command_queue.put(f"python run_model.py {model} {fps} ./images --accelerate")

        color_log.log_info(f"{get_current_time()} -- Command queue has been filled")

    def start_power_data_logger(self):
        self._power_data_logger_process = subprocess.Popen(LOGGER_STARTING_COMMAND,
                                                           stdout=subprocess.PIPE,
                                                           stderr=subprocess.PIPE,
                                                           text=True, shell=True)
        
    def start_operator(self):
        with open(os.devnull, 'w') as devnull:
            self.operator_process = subprocess.Popen(START_OPERATOR_COMMAND, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
            color_log.log_info(f"{get_current_time()} -- Operator process has been started")
            return self.operator_process

    def terminate_processes(self):
        if self.operator_process and self.operator_process.poll() is None:
            self.operator_process.terminate()
            color_log.log_info(f"{get_current_time()} -- Terminated operator process")

        if self._power_data_logger_process and self._power_data_logger_process.poll() is None:
            self._power_data_logger_process.terminate()
            color_log.log_info(f"{get_current_time()} -- Terminated power data logger process")

    @staticmethod
    def clear_command_feedback_file():
        open(COMMAND_FEEDBACK_FILE, 'w').close()


    @staticmethod
    def parse_logger_output_line(line):
        keys = ["timestamp", "sample_in_packet", "voltage_V", "current_A", "dp_V", "dn_V", "temp_C_ema", "energy_Ws",
                "capacity_As"]
        values = line.split()
        values = [float(value) for value in values]
        result = dict(zip(keys, values))
        return result

    def run(self):
        operator_process = self.start_operator()
        while True:
            operator_output = operator_process.stdout.readline()
            if operator_output == '' and operator_process.poll() is not None:
                break
            if operator_output:
                print(operator_output.strip())
                if 'Registered clients:' in operator_output:
                    break

        self.commander.connect()

        # Set additional metrics
        first_line_passed = False
        logger_started = False
        logger_stopped = False
        initial_temperature = None
        current_command = None
        previous_cooling_time = time.time()

        color_log.log_info(f"{get_current_time()} -- Starting benchmarking...")

        with open(LOGGER_OUTPUT_FILE, 'w') as f:
            self.start_power_data_logger()
            while True:
                # Read power data logger output
                logger_output = self._power_data_logger_process.stdout.readline()
                if first_line_passed and logger_output == '' and self._power_data_logger_process.poll() is not None:
                    color_log.log_error(f"{get_current_time()} -- The logger process was stopped.")
                    logger_stopped = True
                    break
                if logger_output.strip():
                    if self._save_logger_output.is_set():
                        f.write(logger_output)
                        f.flush()
                else:
                    # Wait for logger to start giving the output out
                    if not logger_started:
                        continue
                    # Exit the loop after benchmarking
                    elif not self._awaiting_for_feedback.is_set() and self.command_queue.empty():
                        break

                try:
                    # Make a dictionary out of the output
                    logger_output = self.parse_logger_output_line(logger_output)
                except Exception as e:
                    # Occurs when trying to convert string to float, making sure to pass the first logger output line
                    f.write(logger_output)
                    first_line_passed = True
                    color_log.log_info(f"{get_current_time()} -- First output line passed: ({e})")
                    continue

                # Set initial temperature if it wasn't set
                if initial_temperature is None:
                    initial_temperature = logger_output['temp_C_ema']
                    color_log.log_warning(f"{get_current_time()} -- Initial temperature: {initial_temperature} C")
                    logger_started = True

                # If not waiting for feedback
                if not self._awaiting_for_feedback.is_set():
                    # Check if there is a need to cool down the device
                    if logger_output['temp_C_ema'] - initial_temperature > MAX_TEMPERATURE_DIFFERENCE:
                        self._save_logger_output.clear()
                        if time.time() - previous_cooling_time > 60:
                            color_log.log_warning(
                                f"(!) -- {get_current_time()} -- Overheat -- Temporarily cooling down -- "
                                f"Current temperature: {logger_output['temp_C_ema']} C -- Max. expected temperature: "
                                f"{initial_temperature + MAX_TEMPERATURE_DIFFERENCE} C")
                            previous_cooling_time = time.time()
                    else:
                        # Exit if there are no commands left:
                        if self.command_queue.empty():
                            break

                        # Get the command from the queue
                        current_command = self.command_queue.get()
                        if current_command is None:
                            break

                        # Start saving output
                        self._save_logger_output.set()

                        # Send the command
                        self.commander.send_command(RASPBERRY_CLIENT_ID, current_command)
                        self._awaiting_for_feedback.set()

                # If waiting for feedback
                else:
                    if not FEEDBACK_QUEUE.empty():
                        feedback = FEEDBACK_QUEUE.get(timeout=0)
                        # Check if the command matches with the last one sent:
                        if feedback['client_id'] == RASPBERRY_CLIENT_ID and feedback['command'] == current_command:
                            self._awaiting_for_feedback.clear()
                            self._save_logger_output.clear()
                            FEEDBACK_QUEUE.task_done()
                    else:
                        # Still waiting for feedback
                        pass

        if not logger_stopped:
            color_log.log_info(f"{get_current_time()} -- Benchmarking finished")
            self.terminate_processes()
        else:
            self.terminate_processes()


if __name__ == '__main__':
    stress_raspberry = StressRaspberry()
    try:
        stress_raspberry.run()
    except KeyboardInterrupt:
        stress_raspberry.terminate_processes()
        print("\nBenchmarking terminated by user.")

