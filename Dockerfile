FROM python:3.12-slim

RUN apt-get -y update 


# set display port to avoid crash
ENV DISPLAY=:99

# upgrade pip
RUN pip install pip==24.0


RUN python -m pip install pip==24.0

COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

COPY . .

CMD ["/bin/bash", "+x", "/entrypoint.sh"]
