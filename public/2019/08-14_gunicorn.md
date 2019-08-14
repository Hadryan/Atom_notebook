- [￼￼￼gunicorn](https://gunicorn.org)
- [benoitc/gunicorn](https://github.com/benoitc/gunicorn)
- [jonashaag/bjoern](https://github.com/jonashaag/bjoern)
- [pallets/werkzeug](https://github.com/pallets/werkzeug)
- [Docs » Gunicorn - WSGI server](http://docs.gunicorn.org/en/stable/)
- [How to Deploy Python WSGI Apps Using Gunicorn HTTP Server Behind Nginx](https://www.digitalocean.com/community/tutorials/how-to-deploy-python-wsgi-apps-using-gunicorn-http-server-behind-nginx)
- [How To Serve Flask Applications with Gunicorn and Nginx on Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-18-04)

# nginx
  - [Nginx支持Socket转发过程详解](https://www.cnblogs.com/knowledgesea/p/6497783.html)
  - [nginx 配置转发到其他多台服务器](https://www.cnblogs.com/gjack/p/8315603.html)
  - [Flask部署| gunicorn、nginx部署flask项目，并用supervisor来管理进程](https://www.cnblogs.com/xmxj0707/p/8452881.html)
  ```sh
  server {
      listen       8000;
      server_name  flasktest;
      location / {
              proxy_pass http://127.0.0.1:8000;
      }

  }
  ```
