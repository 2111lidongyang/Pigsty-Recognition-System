import random

import paho.mqtt.client as mqtt
from django.test import TestCase

# Create your tests here.
class MqttSubscriber:
    def __init__(self, mqtt_broker, mqtt_port, mqtt_keepalive, client_id, topic):
        self.client = mqtt.Client(client_id, protocol=mqtt.MQTTv31)  # 指定使用旧的回调API版本
        self.topic = topic
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(mqtt_broker, mqtt_port, mqtt_keepalive)
        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT successfully!")
            self.client.subscribe(self.topic)
        else:
            print("Failed to connect, return code {0}".format(rc))

    def on_message(self, client, userdata, message):
        print(f"Received message: {message.payload.decode()}")  # 处理接收到的消息的逻辑

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

def getdetectdata():
    client_id = f'python-mqtt-subscribe-{random.randint(0, 1000)}'  # 可自定义，但要注意客户端id不能重复
    mqtt_broker = 'broker.emqx.io'
    mqtt_port = 1883
    mqtt_keepalive = 600
    topic = 'ljy/test'

    subscriber = MqttSubscriber(mqtt_broker, mqtt_port, mqtt_keepalive, client_id, topic)
    try:
        while True:
            # 在这里添加处理接收到的消息的逻辑，例如：
            message = input("Enter your message: ")
            subscriber.publish_message('收到')

    except BaseException as e:
        print('error:', str(e))
    finally:
        subscriber.disconnect()


getdetectdata()