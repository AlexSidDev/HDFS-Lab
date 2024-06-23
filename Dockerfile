FROM bitnami/spark:latest

#EXPOSE 8078
USER root
#RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install scikit-learn pandas matplotlib psutil pyarrow

USER 1001

COPY ./application/* /opt/bitnami/spark/

CMD ["bin/spark-class", "org.apache.spark.deploy.master.Master"]