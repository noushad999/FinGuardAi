# syntax=docker/dockerfile:1.7
# Spark Structured Streaming image: official Apache Spark + Kafka source + S3A connector.
FROM apache/spark:3.5.6-python3

USER root
ARG MAVEN=https://repo1.maven.org/maven2
# Optional build secret "extra_ca" for TLS-intercepting proxies (see docs/deployment.md).
RUN --mount=type=secret,id=extra_ca,required=false \
    set -eu; CA=""; if [ -s /run/secrets/extra_ca ]; then CA="--cacert /run/secrets/extra_ca"; \
      export PIP_CERT=/run/secrets/extra_ca; fi; \
    cd /opt/spark/jars; \
    for jar in \
      org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.6/spark-sql-kafka-0-10_2.12-3.5.6.jar \
      org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.6/spark-token-provider-kafka-0-10_2.12-3.5.6.jar \
      org/apache/kafka/kafka-clients/3.4.1/kafka-clients-3.4.1.jar \
      org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar \
      org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar \
      com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar; do \
      curl -fsSL $CA -O "$MAVEN/$jar"; \
    done; \
    pip3 install --no-cache-dir psycopg2-binary==2.9.9
COPY streaming/spark_jobs/ /opt/finguard/streaming/
RUN mkdir -p /tmp/spark-ivy && chown -R spark /opt/finguard /tmp/spark-ivy
USER spark
WORKDIR /opt/finguard
CMD ["/opt/spark/bin/spark-submit", "--master", "local[2]", "--driver-memory", "1g", \
     "/opt/finguard/streaming/transactions_stream.py"]
