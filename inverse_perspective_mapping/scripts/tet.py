
import serial
import time

def send_command_and_receive_response(ser, command):
    ser.write(command.encode())  # Send command to Arduino
    response = ser.readline().decode().strip()  # Read response from Arduino
    return response

# Initialize serial connection
ser = serial.Serial('/dev/ttyACM0', 115200)  # Change '/dev/ttyUSB0' to the appropriate port
time.sleep(5)  # Wait for Arduino to initialize

# List of commands to send
commands = [
    "set_home>\\n",
    "Command2\n",
    "Command3\n",
    # Add more commands as needed
]

# Loop through each command and send it to Arduino
for command in commands:
    print(f"Sending command to Arduino: {command.strip()}")
    response = send_command_and_receive_response(ser, command)
    print(f"Response: {response}")

# Close serial connection
ser.close()
