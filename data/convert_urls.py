import pdb
import sys
import json
import os
LOCAL_PATH_PREFIX = "file:///data/private/zqh/embodied/tdw/resources"

inp_file = open(sys.argv[1])
out_file = open(sys.argv[2], "w")
download_commands = []
for line in inp_file.readlines():
    commands = json.loads(line.strip())
    for command in commands:
        # if 'url' in command:
        #     print(os.path.isfile(command['url'][7:]), command['url'][7:])
        if 'url' in command and "amazonaws.com" in command['url']:
            new_url = command['url'].split("/")[-1]
            new_url = f"{LOCAL_PATH_PREFIX}/{new_url}"
            if not os.path.isfile(new_url[7:]):
                if command['url'] not in download_commands:
                    download_commands.append(command['url'])
                    out_file.write(f"wget {command['url']}\n")
        #     command['url'] = new_url
    # commands = json.dumps(commands)
    # out_file.write(commands+"\n")
