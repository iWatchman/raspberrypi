import requests
import datetime

def do_this():

    #server_addr = "https://test-project-156600.appspot.com/api/reportEvent"
    server_addr = "http://c7c4d9d1.ngrok.io/api/reportEvent"

    with open('./vids/test.mp4', 'rb') as f:
        bin_data = f.read()
    #clip = './vids/test.mp4'

    now_date = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    print(now_date)
    camName = "camera 2"

    clip = "test.mp4"

    these_header = {'content-disposition': 'form-data'}
    these_files = {'videoClip': ('test.mp4', bin_data, 'video/mp4')}

    #data={'videoClip': clip, 'date': now_date, 'cameraName': camName}
    r = requests.post(server_addr,data={'date': now_date, 'cameraName': camName},headers = these_header,files=these_files)
    print(r.status_code, r.reason)
    print(r.headers)
    print(r.text[:300] + '...')

if __name__ == '__main__':
    do_this()
