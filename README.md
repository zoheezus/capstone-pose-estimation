# Pose Estimation Capstone

Sending webcam frames to our local server, which will then forward them to an AWS image processing gpu server. The AWS server will then send the results (which will include poses with their corresponding names) back to our local server, so that we can display them in the browser. <br/>

Zohaib Afridi <br/>
JMU Capstone Fall 2020 <br/>

## PowerShell
```powershell
docker-machine start capstone
docker-machine regenerate-certs capstone
docker-machine env capstone | Invoke-Expression
docker-machine ssh capstone
exit
```

## Docker
```docker
docker build -t zoheezus/capstone .
docker push zoheezus/capstone
docker run -p 8089:8089 zoheezus/capstone
```