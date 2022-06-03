
import requests
from requests_ntlm import HttpNtlmAuth ## https://github.com/brandond/requests-negotiate-sspi/blob/master/README.md
import pandas as pd
import time
import logging

"Define logger"
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger('rest client')

class client(object):
    def __init__(self, username, password, domain='nest.local', url='https://visualizer.nestcollaboration.ch/Backend/api/v1/datapoints/'):
        self.username = username
        self.password = password
        self.domain = domain
        self.url = url
        logger.debug('Client initialized')


    def read(self, df_data=pd.DataFrame(columns=[]),startDate='2016-10-01',endDate='2018-10-02'):
        """

        :rtype: object
        """
        s = requests.Session()
        s.auth = HttpNtlmAuth(self.domain + "\\" + self.username, self.password)
        r = s.get(url=self.url)
        if r.status_code != requests.codes.ok:
            logger.debug(r.status_code)
        else:
            logger.debug('Login successful')

            df_result = pd.DataFrame({'timestamp': []})
            for column in df_data:
                try:
                    df = pd.DataFrame(data=s.get(url=self.url + column + '/timeline?startDate=' + startDate + '&endDate=' + endDate).json())
                    df.rename(columns={'value': column}, inplace=True)
                    df['timestamp'] = df['timestamp'].astype('datetime64[m]')
                    df_result = pd.merge(df_result, df, how='outer', on='timestamp')
                except Exception as e:
                    logger.error(e)

            df_result.set_index('timestamp', inplace=True)
            logger.debug('Data acquired')
            return df_result





