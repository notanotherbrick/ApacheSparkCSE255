export PATH="/home/shire/anaconda2/bin:$PATH"
export SPARK_PATH="/home/shire/Documents/spark"

#export PATH="/home/kishan/anaconda2/bin:$PATH"
#export SPARK_PATH=/home/kishan/Desktop/spark

export PYSPARK_DRIVER_PYTHON="jupyter"
export PYSPARK_DRIVER_PYTHON_OPTS="notebook"
# Uncomment next line if the default python on your system is python3
# export PYSPARK_PYTHON=python3
cd "$1"
$SPARK_PATH/bin/pyspark --master local[2]
