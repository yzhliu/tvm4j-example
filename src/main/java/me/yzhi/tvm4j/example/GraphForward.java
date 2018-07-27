package me.yzhi.tvm4j.example;

import ml.dmlc.tvm.Module;
import ml.dmlc.tvm.NDArray;
import ml.dmlc.tvm.TVMContext;
import ml.dmlc.tvm.contrib.GraphModule;
import ml.dmlc.tvm.contrib.GraphRuntime;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Scanner;

public class GraphForward {
  public static void main(String[] args) throws IOException {
    String loadingDir = args[0];
    Module libmod = Module.load(loadingDir + File.separator + "deploy_lib.so");
    String graphJson = new Scanner(new File(loadingDir + File.separator + "deploy_graph.json"))
        .useDelimiter("\\Z").next();
    byte[] params = readBytes(loadingDir + File.separator + "deploy_param.params");

    TVMContext ctx = TVMContext.cpu();

    GraphModule graph = GraphRuntime.create(graphJson, libmod, ctx);

    graph.loadParams(params).setInput("data", RandomInput()).run();

    NDArray output = NDArray.empty(new long[]{1, 1000});
    graph.getOutput(0, output);

    float[] outputArr = output.asFloatArray();
    for (int i = 0; i < 10; ++i) {
      System.out.println(outputArr[i]);
    }

    System.out.println("Done.");
  }

  public static byte[] readBytes(String filename) throws IOException {
    File file = null;
    FileInputStream fileStream = new FileInputStream(file = new File(filename));
    byte[] arr = new byte[(int) file.length()];
    fileStream.read(arr, 0, arr.length);
    return arr;
  }

  private static NDArray RandomInput() {
    float[] arr = new float[1*3*224*224];
    for (int i = 0; i < arr.length; ++i) {
      arr[i] = (float) Math.random();
    }
    NDArray nd = NDArray.empty(new long[]{1, 3, 224, 224});
    nd.copyFrom(arr);
    return nd;
  }
}
