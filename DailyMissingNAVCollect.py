from urllib import urlencode
import urllib2
from bs4 import BeautifulSoup
from datetime import datetime
import csv
import time
import datetime
from django.db import connection
import numpy as np
import collections

arr = [1,3,5,10]
url = "http://www.amfiindia.com/modules/NavHistoryCompare"
fo = open("foo.txt", "wb")

def dictfetchall(cursor):
###    "Return all rows from a cursor as a dict"
    print cursor.description
    columns = [col[0] for col in cursor.description]
    return [
        dict (zip(columns, row))
        for row in cursor.fetchall()
    ]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

cursor = connection.cursor()
cursor.execute( "select DateOfPurchase, b.MF_ID, c.HID from (select distinct Asset, DateOfPurchase from Customer_Master where AssetType='MF')a, (select distinct MF_ID, MF_NM from MF_DAILY) b , (select distinct MF_ID, HID from HOUSE_FUND_MAP) c where a.Asset = MF_NM and b.MF_ID = c.MF_ID " )
row = dictfetchall(cursor)
for rowCtr in range(len(row)):
        # cant take today since it will load with a days lag
            frmDt1 =  row[rowCtr]['DateOfPurchase']
            if frmDt1.weekday() == 5:
                 frmDt1 = frmDt1 -  datetime.timedelta(days=1)
            elif frmDt1.weekday() == 6:
                 frmDt1 = frmDt1 -  datetime.timedelta(days=2)

            tdt = frmDt1 + datetime.timedelta(days=1)
            values = {'mfID' : str(row[rowCtr]['HID']), 'scID' : str(row[rowCtr]['MF_ID']) , 'fDate' : frmDt1.strftime("%d-%b-%Y"), 'tDate' : tdt.strftime("%d-%b-%Y") }
            print (values)
            data = urlencode(values)
            req = urllib2.Request(url, data)
            response = urllib2.urlopen(req)

            the_page = response.read()
            soup = BeautifulSoup( the_page )
            MFDTLS = soup.findAll('th', {'class': 'txt-lft'})
            NAVDTLS = soup.findAll('td')
            if len(NAVDTLS)>6 and is_number( str(NAVDTLS[5].text) ) and is_number( str(NAVDTLS[1].text) ) and float(str(NAVDTLS[1].text))>0 :
                #growth = pow( (float(NAVDTLS[5].text)/ float(NAVDTLS[1].text)) , ( 1/float(   ) ) ) - 1
                print ( MFDTLS[0].text+'#'+MFDTLS[1].text+'#'+datetime.datetime.strptime( NAVDTLS[4].text, '%d-%b-%Y %H:%M:%S').strftime('%Y-%m-%d')+'#'+NAVDTLS[1].text+'#'+datetime.datetime.strptime( NAVDTLS[8].text, '%d-%b-%Y %H:%M:%S').strftime('%Y-%m-%d')+'#'+NAVDTLS[5].text+'#'+datetime.datetime.now().strftime('%Y-%m-%d')+"\n"  )


