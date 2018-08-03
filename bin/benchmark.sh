#!/bin/bash
CURR_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=$(cd $(dirname $0)/../; pwd)
mkdir -p $CURR_DIR/libs

cp $PROJ_DIR/data/arr.txt .

echo "Run python compiler and dump library ..."
python $CURR_DIR/benchmark.py $CURR_DIR/libs

echo "Run java code ..."
CLASSPATH=$CLASSPATH:$PROJ_DIR/target/*:$PROJ_DIR/target/classes/lib/*
java -cp $CLASSPATH \
  -Dlog4j.configuration=file://$PROJ_DIR/conf/log4j.properties \
  me.yzhi.tvm4j.example.Benchmark $CURR_DIR/libs

echo "Run diff over out_py.txt and out_java.txt"
diff out_py.txt out_java.txt