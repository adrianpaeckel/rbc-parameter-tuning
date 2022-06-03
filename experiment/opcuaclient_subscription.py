#########################################################################################################################
# Name: opc ua client
# Version: 0.1

# Activities:                                           Author:                         Date:
# Initial comment                                       RK                              20190409
# Add toggle module                                     RK                              20190411
# Add computer name to the client                       RK                              20190411
# Changed order to subscribe/ publish                   RK                              20190411
# Moved the try statement inside the for loop           RK                              20190425
# Changed from time to datetime to show milliseconds    RK                              20190508

########################################################################################################################
from opcua import Client, ua
from opcua.ua import UaStatusCodeError
from opcua.common import ua_utils

import pandas as pd
import datetime
import logging
import socket

"""Initialize logger"""
logger = logging.getLogger('opc ua client')


def toggle() -> bool:
    """Toggle function

    The watchdog has to toggle every 5 seconds
    otherwise the connection will be refused
    """
    return datetime.datetime.now().second % 10 < 5

class SubHandler(object):
    """Subscription Handler.

    To receive events from server for a subscription
    data_change and event methods are called directly from receiving thread.
    Do not do expensive, slow or network operation there. Create another
    thread if you need to do such a thing. You have to define here
    what to do with the received date.
    """

    def __init__(self):
        self.df_Read = pd.DataFrame(data={'node': [], 'value': []}).astype('object')
        self.json_Read = self.df_Read.to_json()

    def datachange_notification(self, node, val, _):
        try:
            df_New = pd.DataFrame(data={'node': [], 'value': []}).astype('object')
            df_New.at[0, 'node'] = node.nodeid.to_string()
            df_New.at[0, 'value'] = val
            self.df_Read = self.df_Read.merge(df_New, on=list(self.df_Read), how='outer')
            self.df_Read.drop_duplicates(subset=['node'], inplace=True, keep='last')
            self.json_Read = self.df_Read.to_json()
            logger.info('read %s %s' % (node, val))
        except Exception as e:
            logger.error(e)


    def event_notification(self, event):
        logger.info("Python: New event", event)

class opcua(object):
    def __init__(self,  url='opc.tcp://ehub.nestcollaboration.ch:49320',
                        application_uri='Python client',
                        product_uri='Research client',
                        user='JustforTest',
                        password='JustforTest'):
        """Initialized the opc ua client

        Initializing the opc ua client with the default parameter. In the later stage the client contains an
        encryption certificate as well"""

        self.client = Client(url=url, timeout=4) # You have to enter the url of the opc ua server e.g "opc.tcp://ehub.nestcollaboration.ch:49320"
        self.client.set_user(user)  # You have to enter your User name*
        self.client.set_password(password)  # You have to enter your password*
        self.client.application_uri = f"{application_uri}:{socket.gethostname()}:{user}"# You have to enter the uri according to the name or path of your certificate and key*
        self.client.product_uri =f"{product_uri}:{socket.gethostname()}:{user}"# You have to enter the uri according to the name or path of your certificate and key*
        ##self.self.client.set_security_string("Basic128Rsa15,SignAndEncrypt,uaexpert.der,uaexpert_key.pem")  # You have to enter the name or path of your certificate and key*

        self.client.secure_channel_timeout =600000 #300000 Attempt to get rid of the broken pipe error. Right now there the channel gets checked every hour
        self.client.session_timeout = 30000 #15000 Attempt to get rid of the broken pipe error. Right now there the session gets checked every hour
        self.client.name = user
        self.client.description = self.client.name
        self.handler = SubHandler()
        self.bInitPublish = False

    def connect(self):
        """Connect

        Connect the client to the server and load the specific type definitions"""
        try:
            self.client.connect()
            self.client.load_type_definitions()  # load definition of server specific structures/extension objects
            logger.info('OPC UA Connection to server established')
            return True
        except Exception as e:
            logger.error(e)
            return False

    def disconnect(self):
        """Disconnect"""
        try:
            self.client.disconnect()
            logging.info("Server disconnected.")
        except UaStatusCodeError as e:
            logging.error(f"Server disconnected with error: {e}")



    def subscribe(self,json_Read):
        """Subscribe

        Subscribe to all values you want to read"""
        self.df_Read =pd.read_json(json_Read)
        nodelistRead = []
        for index, row in self.df_Read.iterrows():
            nodelistRead.append(self.client.get_node(row['node']))

        try:
            self.sub = self.client.create_subscription(period=0, handler=self.handler)
            self.sub.subscribe_data_change(nodelistRead)
            logger.info('OPC UA Subscription requested')
        except Exception as e:
            logging.error(e)


    def publish(self,json_Write):
        """Publish

        All values you want to write will be sent to the opc ua server"""
        self.df_Write= pd.read_json(json_Write)
        if not self.bInitPublish:
            self._nodeobjects = [self.client.get_node(node)
                                 for node in self.df_Write['node'].tolist()]
            try:
                self._datatypes = [nodeObject.get_data_type_as_variant_type()
                                   for nodeObject in self._nodeobjects]
                self.bInitPublish = True
                logging.info("Publishing initialized.")
            except UaStatusCodeError as e:
                logging.error(f"UaStatusCodeError while initializing publishing!: {e}")
                raise TimeoutError


        try:
            self._ua_values = [ua.DataValue(ua.Variant(ua_utils.string_to_val(str(value), datatype), datatype))
                               for value, datatype in zip(self.df_Write['value'].tolist(), self._datatypes)]
            self.client.set_values(nodes=self._nodeobjects, values=self._ua_values)
            [logger.info('write %s %s' % (nodeobject, value))
                for nodeobject, value in zip(self._nodeobjects, self._ua_values)]
        except UaStatusCodeError as e:
            logging.error(f"UaStatusCodeError: {e} happened while publishing!")
            raise TimeoutError


    def reset_sub(self,json_Read):
        try:
            print('Resetting server connection')
            self.connect()
            self.subscribe(json_Read=json_Read)
        except Exception as e:
            print('Reset ERROR',e)
            raise Exception
