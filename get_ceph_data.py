import json
import requests
import io
import os
class ReadDataFailError(Exception):pass
def better_latex_sentense_string(latex_string:str):
    if latex_string is None:return None
    latex_string = latex_string.replace('\n'," ")
    latex_string = multispaces_into_singlespace(latex_string)
    return latex_string.strip()

def build_client():
    print(f"we will building ceph client...................")
    from petrel_client.client import Client  # 安装完成后才可导入
    client = Client(conf_path="~/petreloss.conf") # 实例化Petrel Client，然后就可以调用下面的APIs   
    print(f"done..................")
    return client

def check_path_exists(path,client):
    if "s3" in path:
        return client.contains(path)
    elif path.startswith('http'):
        assert 'get_data' in path, "please use get_data flag for data path"
        response = requests.get(path.replace('get_data','checkfile'))
        if response.status_code == 200:
            status = response.json()["status"]
            return status
        else:
            return False
    else:
        return os.path.exists(path)

def check_lock_exists(path, client):
    if "s3" in path:
        raise NotImplementedError("s3 lock not implemented")
    elif path.startswith('http'):
        assert 'get_data' in path, "please use get_data flag for data path"
        response = requests.get(path.replace('get_data','checklock'))
        if response.status_code == 200:
            status = response.json()["status"]
            return status
        else:
            return False
    else:
        raise NotImplementedError("please donot use lock lock")
        return os.path.exists(path)



def read_json_from_path(path, client):
    if "s3" in path:
        buffer = client.get(path).decode('utf-8')
        if path.endswith('.json'):
            return json.loads(buffer)
        else:
            return {'content':str(buffer)}
    elif path.startswith('http'):
        response = requests.get(path)
        if response.status_code == 200:
            content = response.json()["content"]
            if path.endswith('.json'):
                content = json.loads(content)
            elif path.endswith('.md'):
                content = {'content':content}
            return content
        else:
            return None
    else:
        with open(path,'r') as f:
            data = json.load(f)
            return data

def write_json_to_path(data, path, client):
    if "s3" in path:
        byte_object = json.dumps(data).encode('utf-8')
        with io.BytesIO(byte_object) as f:
            client.put(path, f)
    else:
        assert not path.startswith('http'), "why you want to save the file to a online path?"
        thedir = os.path.dirname(path)
        os.makedirs(thedir, exist_ok=True)
        with open(path,'w') as f:
            json.dump(data, f)

if __name__ == "__main__":
    

    import re
    arxivid= "quant-ph_0004003"
    client = None #build_client()
    onlinepath = "" #"uparxive:s3://uparxive/json"
    for datapath in ["http://10.140.52.123:8000/get_data"]:
        if "s3:" in datapath or "s3:" in onlinepath:
            if client is None:
                client = build_client()

        if not arxivid.endswith('.json'):
            arxivid    = os.path.split(arxivid)[-1]  #<<--- get the arxiv id
        else:
            filepath = arxivid ## assume you
            arxivid  = os.path.split(arxivid)[-1][:-5]
        match = re.search(r"\d{4}", arxivid)
        if match:
            # Print the matched pattern
            date = match.group()
        metadataname = os.path.join(date, arxivid , 'metadata.json')
        arxiv_name   = os.path.join(date, arxivid, 'uparxive', arxivid+'.json')
        filepath     = os.path.join(datapath  , arxiv_name)
        metadatapath = os.path.join(datapath  , metadataname)
        data = read_json_from_path(filepath, client)
        status = check_path_exists(filepath, client)
        print(status)
        status = check_lock_exists(filepath, client)
        print(status)
        # if "s3:" in onlinepath:
        #     assert not arxivid.endswith('.json'), "please provide the path end with the arxivdi rather the .json file"
        #     targetpath = os.path.join(onlinepath, arxivid, 'uparxive', arxivid+'.json')
        # else:
        #     assert not datapath.startswith('http'), "why the output dir is a online path???"
        #     if not arxivid.endswith('.json'):
        #         targetpath = os.path.join(datapath, arxivid, 'uparxive', arxivid+'.json')
        #     else:
        #         targetpath = arxivid[:-5] + '.json'

    
        

        # print(data)
        # print("============================================")
        # if "s3" not in datapath:
        #     write_json_to_path(data, targetpath, client)