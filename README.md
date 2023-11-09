# ai-server

To start the server:

```
cd nart-ai-server
git pull
sudo sysctl -w net.ipv4.ip_unprivileged_port_start=0
conda activate stable
gunicorn main:app -b 0.0.0.0:80 --timeout 180
```
