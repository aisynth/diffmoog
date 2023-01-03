FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install openssh-server -y
RUN apt-get install rsync -y

RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/' -i /etc/ssh/sshd_config
RUN sed 's/session\s*required\s*pam_loginuid.so/session optional pam_loginuid.so/' -i /etc/pam.d/sshd
EXPOSE 22

WORKDIR /storage/

CMD ["/usr/sbin/sshd", "-D"]
